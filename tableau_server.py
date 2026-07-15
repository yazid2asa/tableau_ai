"""Tableau Server API client — using official tableauserverclient (TSC).

Handles PAT auth, workbook publish, view URL retrieval, and datasource
schema fetching via Metadata API (GraphQL).
"""

import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import Optional

import tableauserverclient as TSC

from config import settings
from schemas import DataSourceMetadata, FieldInfo, FieldType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TSC Server instance — cached, reused across calls
# ---------------------------------------------------------------------------
_server: Optional[TSC.Server] = None
_signed_in: bool = False
_cached_project_luid: Optional[str] = None

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _get_server() -> TSC.Server:
    """Get or create the TSC Server instance."""
    global _server
    if _server is None:
        server_url = settings.tableau_server_url.rstrip("/")
        _server = TSC.Server(server_url, use_server_version=True)
        _server.add_http_options({"verify": True})
    return _server


def _ensure_signed_in() -> TSC.Server:
    """Ensure the TSC server is signed in via PAT. Returns the server."""
    global _signed_in
    server = _get_server()
    if not _signed_in:
        auth = TSC.PersonalAccessTokenAuth(
            settings.tableau_pat_name,
            settings.tableau_pat_secret,
            site_id=settings.tableau_site_id,
        )
        server.auth.sign_in(auth)
        _signed_in = True
        logger.info(
            "Signed in to Tableau Server: %s (site: %s)",
            settings.tableau_server_url, settings.tableau_site_id,
        )
    return server


def _sign_out_and_reset():
    """Sign out and reset cached state (for 401 retry)."""
    global _server, _signed_in, _cached_project_luid
    if _server and _signed_in:
        try:
            _server.auth.sign_out()
        except Exception:
            pass
    _server = None
    _signed_in = False
    _cached_project_luid = None


def _is_auth_error(exc: Exception) -> bool:
    s = str(exc)
    return any(tok in s for tok in ("401", "403", "Unauthorized", "Forbidden"))


# ---------------------------------------------------------------------------
# Backward-compatible async wrappers (called by main.py)
# TSC is synchronous — we wrap with asyncio.to_thread()
# ---------------------------------------------------------------------------


async def signin() -> tuple[str, str]:
    """Sign in via PAT -> returns (token, site_id). Backward-compatible."""
    def _do():
        server = _ensure_signed_in()
        return server.auth_token, server.site_id
    return await asyncio.to_thread(_do)


# Keep for backward compatibility with test-publish endpoint
async def _authed_request(method: str, url: str, **kwargs):
    """Backward-compatible authed request using httpx (for GraphQL only)."""
    import httpx
    server = await asyncio.to_thread(_ensure_signed_in)
    token = server.auth_token
    headers = kwargs.pop("headers", {})
    headers["X-Tableau-Auth"] = token
    headers.setdefault("Accept", "application/json")
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, headers=headers, **kwargs)
    return resp


def _base_url() -> str:
    """Return the REST API base URL."""
    server = _get_server()
    return server.baseurl


# ---------------------------------------------------------------------------
# Project LUID resolution
# ---------------------------------------------------------------------------


def _resolve_project_luid_sync(configured_id: str) -> str:
    """Resolve project LUID. Uses TSC to list projects."""
    global _cached_project_luid
    if _cached_project_luid:
        return _cached_project_luid

    server = _ensure_signed_in()

    # Fetch all projects
    all_projects, _ = server.projects.get()

    # If configured value is a valid UUID, check if it exists
    if configured_id and _UUID_RE.match(configured_id):
        for p in all_projects:
            if p.id == configured_id:
                _cached_project_luid = configured_id
                logger.info("Using configured project: '%s' (LUID: %s)", p.name, configured_id)
                return _cached_project_luid
        logger.warning("Configured project LUID %s not found on Server", configured_id)
    elif configured_id:
        logger.warning("TABLEAU_DEFAULT_PROJECT_ID '%s' is not a valid UUID — auto-detecting", configured_id)

    # Auto-detect: prefer "default" project
    if not all_projects:
        raise RuntimeError("No projects found on Tableau Server")

    default = next((p for p in all_projects if p.name.lower() == "default"), None)
    chosen = default or all_projects[0]
    _cached_project_luid = chosen.id
    logger.info("Auto-selected project: '%s' (LUID: %s)", chosen.name, _cached_project_luid)
    return _cached_project_luid


# ---------------------------------------------------------------------------
# Workbook publish (TSC)
# ---------------------------------------------------------------------------


async def publish_workbook(twb_path: str, project_id: str, overwrite: bool = True) -> str:
    """Publish a .twb to Tableau Server/Cloud -> returns workbook LUID.

    Uses TSC's native publish which handles multipart formatting correctly.
    """
    def _do():
        server = _ensure_signed_in()
        resolved_project_id = _resolve_project_luid_sync(project_id)

        mode = TSC.Server.PublishMode.Overwrite if overwrite else TSC.Server.PublishMode.CreateNew

        wb_item = TSC.WorkbookItem(resolved_project_id)
        wb_item.name = os.path.splitext(os.path.basename(twb_path))[0]
        wb_item.show_tabs = True

        logger.info(
            "Publishing workbook via TSC: name=%s, project=%s, mode=%s, file=%s",
            wb_item.name, resolved_project_id, mode, twb_path,
        )

        try:
            published = server.workbooks.publish(
                wb_item,
                twb_path,
                mode=mode,
                skip_connection_check=True,
            )
            logger.info("Published workbook: LUID=%s, name=%s", published.id, published.name)
            return published.id
        except Exception as e:
            logger.error("TSC publish failed: %s", e)
            # On auth error, reset and retry once
            if _is_auth_error(e):
                logger.info("Auth expired, re-signing in and retrying publish")
                _sign_out_and_reset()
                server = _ensure_signed_in()
                resolved_project_id = _resolve_project_luid_sync(project_id)
                wb_item = TSC.WorkbookItem(resolved_project_id)
                wb_item.name = os.path.splitext(os.path.basename(twb_path))[0]
                wb_item.show_tabs = True
                published = server.workbooks.publish(
                    wb_item, twb_path, mode=mode, skip_connection_check=True,
                )
                return published.id
            raise

    return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# View URL retrieval (TSC)
# ---------------------------------------------------------------------------


async def get_view_url(workbook_luid: str) -> str:
    """Get the first view URL for a published workbook."""
    def _do():
        server = _ensure_signed_in()
        wb = server.workbooks.get_by_id(workbook_luid)
        server.workbooks.populate_views(wb)
        if not wb.views:
            raise ValueError(f"No views found for workbook {workbook_luid}")
        view = wb.views[0]
        # TSC returns content_url as "WorkbookName/sheets/SheetName"
        # Browser URL format is "views/WorkbookName/SheetName" (no /sheets/)
        content_url = view.content_url.replace("/sheets/", "/")
        server_url = settings.tableau_server_url.rstrip("/")
        site_path = f"/site/{settings.tableau_site_id}" if settings.tableau_site_id else ""
        return f"{server_url}/#{site_path}/views/{content_url}"

    return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# Datasource content URL (TSC)
# ---------------------------------------------------------------------------


async def get_datasource_content_url(luid: str) -> str:
    """Get the contentUrl for a published datasource by LUID."""
    def _do():
        server = _ensure_signed_in()
        ds = server.datasources.get_by_id(luid)
        return ds.content_url

    return await asyncio.to_thread(_do)


async def get_dimension_members(datasource_luid: str, field_caption: str,
                                limit: int = 500) -> list[str]:
    """Return the distinct string members of a dimension via the VizQL Data Service.

    Used to validate/correct LLM-generated filter VALUES against what actually
    exists in the datasource (FIX-054). Querying a single dimension field returns
    its distinct domain. Returns [] on any failure (auth, VDS disabled, unknown
    field) so the caller degrades gracefully to "no correction" rather than erroring.
    """
    import httpx

    def _do() -> list[str]:
        server = _ensure_signed_in()
        token = server.auth_token
        base = settings.tableau_server_url.rstrip("/")
        url = f"{base}/api/v1/vizql-data-service/query-datasource"
        body = {
            "datasource": {"datasourceLuid": datasource_luid},
            "query": {"fields": [{"fieldCaption": field_caption}]},
        }
        headers = {
            "X-Tableau-Auth": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
        except Exception as exc:
            logger.warning("VDS member query transport error for %s.%s: %s",
                           datasource_luid, field_caption, exc)
            return []
        if resp.status_code != 200:
            # On auth expiry, reset+retry once (mirror publish_workbook).
            if _is_auth_error(Exception(str(resp.status_code))) or resp.status_code in (401, 403):
                _sign_out_and_reset()
                server2 = _ensure_signed_in()
                headers["X-Tableau-Auth"] = server2.auth_token
                try:
                    resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
                except Exception:
                    return []
            if resp.status_code != 200:
                logger.warning("VDS member query HTTP %s for %s.%s: %s",
                               resp.status_code, datasource_luid, field_caption, resp.text[:200])
                return []
        try:
            rows = resp.json().get("data", [])
        except Exception:
            return []
        members: list[str] = []
        for row in rows:
            # single-field query → one key per row; take its value
            for v in row.values():
                if v is not None and str(v) != "":
                    members.append(str(v))
                break
            if len(members) >= limit:
                break
        return members

    return await asyncio.to_thread(_do)


# VizQL Data Service aggregate functions accepted by query-datasource.
_VDS_AGG_FUNCTIONS = {"SUM", "AVG", "MEDIAN", "COUNT", "COUNTD", "MIN", "MAX"}


async def query_datasource_aggregate(
    datasource_luid: str,
    measure_caption: str,
    aggregation: str = "SUM",
    group_by: str | None = None,
    filters: list[dict] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Aggregate query via the VizQL Data Service — the data behind `query_data`
    (direct factual answers in chat, no chart generated).

    Same endpoint/auth/retry pattern as get_dimension_members (proven by FIX-054).
    ``filters`` is a list of ``{"field": caption, "values": [...]}`` dicts,
    emitted as VDS SET filters. Returns the raw row dicts (e.g.
    ``[{"Region": "Ouest", "SUM(Sales)": 1234.5}]``; a single row when group_by
    is None) or ``[]`` on any failure so the caller degrades to a friendly
    "couldn't fetch" message instead of erroring the chat turn.
    """
    import httpx

    agg = (aggregation or "SUM").upper()
    if agg not in _VDS_AGG_FUNCTIONS:
        agg = "SUM"

    fields: list[dict] = []
    if group_by:
        fields.append({"fieldCaption": group_by})
    fields.append({"fieldCaption": measure_caption, "function": agg})

    body: dict = {
        "datasource": {"datasourceLuid": datasource_luid},
        "query": {"fields": fields},
    }
    vds_filters = []
    for f in (filters or []):
        fld = f.get("field")
        vals = f.get("values") or ([f["value"]] if f.get("value") is not None else [])
        if fld and vals:
            vds_filters.append({
                "field": {"fieldCaption": fld},
                "filterType": "SET",
                "values": [str(v) for v in vals],
                "exclude": bool(f.get("exclude", False)),
            })
    if vds_filters:
        body["query"]["filters"] = vds_filters

    def _do() -> list[dict]:
        server = _ensure_signed_in()
        token = server.auth_token
        base = settings.tableau_server_url.rstrip("/")
        url = f"{base}/api/v1/vizql-data-service/query-datasource"
        headers = {
            "X-Tableau-Auth": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
        except Exception as exc:
            logger.warning("VDS aggregate query transport error for %s.%s: %s",
                           datasource_luid, measure_caption, exc)
            return []
        if resp.status_code != 200:
            if resp.status_code in (401, 403):
                _sign_out_and_reset()
                server2 = _ensure_signed_in()
                headers["X-Tableau-Auth"] = server2.auth_token
                try:
                    resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
                except Exception:
                    return []
            if resp.status_code != 200:
                logger.warning("VDS aggregate query HTTP %s for %s.%s: %s",
                               resp.status_code, datasource_luid, measure_caption,
                               resp.text[:200])
                return []
        try:
            rows = resp.json().get("data", [])
        except Exception:
            return []
        return rows[:limit] if isinstance(rows, list) else []

    return await asyncio.to_thread(_do)


async def query_dimension_pairs(
    datasource_luid: str,
    field_a: str,
    field_b: str | None = None,
    filters: list[dict] | None = None,
    limit: int = 2000,
) -> list[dict]:
    """Distinct dimension values (or value pairs) via the VizQL Data Service.

    One field → its (optionally filtered) distinct domain; two fields → the
    distinct ``(a, b)`` combinations. This is the mapping primitive behind the
    cross-datasource ``query_data`` join (C1b): the dimension-owning datasource
    answers "which linking values match this filter?" (e.g. the vehicle_ids of
    brand Ford) or "which group does each linking value belong to?" (vehicle_id
    → brand). Same endpoint / auth-retry pattern as ``query_datasource_aggregate``
    (proven by FIX-054). ``filters`` uses the same ``{"field", "values",
    "exclude"}`` shape. Returns ``[]`` on any failure so the caller degrades to
    a friendly message instead of erroring the chat turn.
    """
    import httpx

    fields = [{"fieldCaption": field_a}]
    if field_b:
        fields.append({"fieldCaption": field_b})
    body: dict = {
        "datasource": {"datasourceLuid": datasource_luid},
        "query": {"fields": fields},
    }
    vds_filters = []
    for f in (filters or []):
        fld = f.get("field")
        vals = f.get("values") or ([f["value"]] if f.get("value") is not None else [])
        if fld and vals:
            vds_filters.append({
                "field": {"fieldCaption": fld},
                "filterType": "SET",
                "values": [str(v) for v in vals],
                "exclude": bool(f.get("exclude", False)),
            })
    if vds_filters:
        body["query"]["filters"] = vds_filters

    def _do() -> list[dict]:
        server = _ensure_signed_in()
        token = server.auth_token
        base = settings.tableau_server_url.rstrip("/")
        url = f"{base}/api/v1/vizql-data-service/query-datasource"
        headers = {
            "X-Tableau-Auth": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
        except Exception as exc:
            logger.warning("VDS pair query transport error for %s.%s/%s: %s",
                           datasource_luid, field_a, field_b, exc)
            return []
        if resp.status_code != 200:
            if resp.status_code in (401, 403):
                _sign_out_and_reset()
                server2 = _ensure_signed_in()
                headers["X-Tableau-Auth"] = server2.auth_token
                try:
                    resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
                except Exception:
                    return []
            if resp.status_code != 200:
                logger.warning("VDS pair query HTTP %s for %s.%s/%s: %s",
                               resp.status_code, datasource_luid, field_a, field_b,
                               resp.text[:200])
                return []
        try:
            rows = resp.json().get("data", [])
        except Exception:
            return []
        return rows[:limit] if isinstance(rows, list) else []

    return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# Metadata API (GraphQL) — Datasource schemas
# TSC has native metadata.query() support
# ---------------------------------------------------------------------------

_GRAPHQL_FIELDS_FRAGMENT = """
      fields {
        name
        isHidden
        ... on ColumnField {
          dataType
          role
          upstreamColumns { name }
        }
        ... on CalculatedField {
          dataType
          role
        }
        ... on DatasourceField {
          remoteField {
            ... on ColumnField {
              dataType
              role
              upstreamColumns { name }
            }
          }
        }
      }
"""

_GRAPHQL_ALL_DATASOURCES = (
    "{ publishedDatasourcesConnection { nodes { luid name"
    + _GRAPHQL_FIELDS_FRAGMENT
    + "} } }"
)

_GRAPHQL_DATASOURCES_BY_LUID = (
    'query($luids: [String!]!) { publishedDatasourcesConnection(filter: {luid: {in: $luids}}) { nodes { luid name'
    + _GRAPHQL_FIELDS_FRAGMENT
    + "} } }"
)


def _map_graphql_datatype(dt: str) -> FieldType:
    """Map Tableau Metadata API dataType to our FieldType enum."""
    mapping = {
        "STRING": FieldType.STRING,
        "INTEGER": FieldType.INTEGER,
        "INT": FieldType.INTEGER,
        "REAL": FieldType.FLOAT,
        "FLOAT": FieldType.FLOAT,
        "DATE": FieldType.DATE,
        "DATETIME": FieldType.DATETIME,
        "BOOLEAN": FieldType.BOOLEAN,
    }
    return mapping.get(dt.upper(), FieldType.STRING)


def _map_graphql_role(role: str) -> str:
    """Map Tableau Metadata API role to dimension/measure."""
    if role and role.upper() == "MEASURE":
        return "measure"
    return "dimension"


# Tableau auto-generates these internal "fields" alongside the real data columns.
# They appear in the Metadata API response but are not user-data and would confuse
# the LLM and the field-validation error messages if surfaced.
_FILE_EXTENSION_FIELD = re.compile(
    r"\.(csv|xlsx?|xls|tsv|txt|json|parquet|hyper|tde|tdsx|tds)(\s*\(count\))?$",
    re.IGNORECASE,
)
_INTERNAL_FIELD_NAMES = {
    "measure names", "measure values", "number of records",
    "latitude (generated)", "longitude (generated)",
}


def _is_junk_field(name: str) -> bool:
    """True for Tableau-internal auto-generated 'fields' that aren't real columns."""
    if not name:
        return True
    low = name.strip().lower()
    if low in _INTERNAL_FIELD_NAMES:
        return True
    if _FILE_EXTENSION_FIELD.search(low):
        return True
    return False


def _parse_datasource_nodes(nodes: list[dict]) -> list[DataSourceMetadata]:
    """Convert GraphQL datasource nodes to DataSourceMetadata list.

    Filters out hidden fields (isHidden=true) and Tableau-internal auto-generated
    fields (Measure Names/Values, <filename>.csv [(Count)], Number of Records,
    generated lat/long) — neither is directly usable in TWB column references via
    sqlproxy and they pollute the LLM prompt + error messages.
    """
    result = []
    for node in nodes:
        fields = []
        for f in node.get("fields", []):
            # Skip hidden fields — they can't be referenced in TWB via sqlproxy
            if f.get("isHidden", False):
                continue
            # Skip Tableau-internal auto-generated fields
            if _is_junk_field(f.get("name", "")):
                continue
            dt = f.get("dataType", "")
            role = f.get("role", "")
            upstream = f.get("upstreamColumns") or []
            if not dt and f.get("remoteField"):
                rf = f["remoteField"]
                dt = rf.get("dataType", "")
                role = rf.get("role", "")
                if not upstream:
                    upstream = rf.get("upstreamColumns") or []
            # Skip fields with no dataType (often internal/generated)
            if not dt:
                continue
            # The physical/upstream column name is the real sqlproxy binding name.
            # GraphQL `name` is the caption (e.g. "Sub Category"); the published
            # datasource binds by the physical name (e.g. "Sub_Category"). Keep it
            # only when it differs, so generation binds to the right field.
            caption = f["name"]
            physical = upstream[0].get("name") if upstream and upstream[0].get("name") else None
            # local_name is still captured (so the post-save sqlproxy safety net and
            # future opt-in physical-binding flows can consult it), but the field
            # registration in twb_generator now binds by caption — see the comment
            # there. We only set local_name when the physical name differs from the
            # caption; otherwise leave it None.
            local_name = physical if physical and physical != caption else None

            fields.append(FieldInfo(
                name=caption,
                type=_map_graphql_datatype(dt),
                role=_map_graphql_role(role or "DIMENSION"),
                local_name=local_name,
            ))
        result.append(DataSourceMetadata(
            datasource_name=node["name"],
            fields=fields,
            luid=node["luid"],
        ))
    return result


async def get_all_datasource_schemas() -> list[DataSourceMetadata]:
    """Fetch all published datasource schemas via Metadata API GraphQL."""
    def _do():
        server = _ensure_signed_in()
        result = server.metadata.query(_GRAPHQL_ALL_DATASOURCES)
        nodes = (
            result.get("data", {})
            .get("publishedDatasourcesConnection", {})
            .get("nodes", [])
        )

        # Apply filter if configured
        ds_filter = settings.tableau_datasource_filter.strip()
        if ds_filter:
            allowed = {s.strip() for s in ds_filter.split(",")}
            nodes = [
                n for n in nodes if n["name"] in allowed or n["luid"] in allowed
            ]
        elif len(nodes) > 20:
            logger.warning(
                "Found %d datasources on Server, hard-capping at 20. "
                "Set TABLEAU_DATASOURCE_FILTER to limit.",
                len(nodes),
            )
            nodes = nodes[:20]

        return _parse_datasource_nodes(nodes)

    return await asyncio.to_thread(_do)


async def get_multiple_datasource_schemas(
    luids: list[str],
) -> list[DataSourceMetadata]:
    """Fetch schemas for specific datasources by LUID."""
    def _do():
        server = _ensure_signed_in()
        result = server.metadata.query(_GRAPHQL_DATASOURCES_BY_LUID, {"luids": luids})
        nodes = (
            result.get("data", {})
            .get("publishedDatasourcesConnection", {})
            .get("nodes", [])
        )
        return _parse_datasource_nodes(nodes)

    return await asyncio.to_thread(_do)
