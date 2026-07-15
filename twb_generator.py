import re
import uuid
from datetime import date, timedelta
from pathlib import Path

from lxml import etree
from twilize.twb_editor import TWBEditor
from twilize.validator import TWBValidationError

from schemas import VizIntent, DataSourceMetadata, FieldInfo, FieldType, CalculatedField
from config import settings

# VizIntent.viz_type → twilize mark_type
VIZ_TYPE_TO_MARK: dict[str, str] = {
    "bar_chart": "Bar",
    "line_chart": "Line",
    "pie": "Pie",
    "scatter": "Scatterplot",
    "area": "Area",
    "heatmap": "Heatmap",
    "treemap": "Tree Map",
    
    "text": "Text",
    # FIX-045: the valid Tableau mark primitive is "GanttBar" (no space). Both
    # "Gantt Bar" (twilize's old default) and "Gantt" are rejected by Tableau's
    # native parser at publish ("Invalid primitive type ..."). Verified empirically.
    "gantt": "GanttBar",
}

# FieldType.value → twilize datatype
_DATATYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "float": "real",
    "date": "date",
    "datetime": "date",
    "boolean": "boolean",
}

# role → twilize field_type
_FIELD_TYPE_MAP: dict[str, str] = {
    "dimension": "nominal",
    "measure": "quantitative",
}


def _normalize(name: str) -> str:
    """Lowercase and strip spaces, underscores, hyphens for fuzzy field matching."""
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _clear_template_fields(ed: TWBEditor) -> None:
    """Remove all pre-loaded template fields from the datasource XML and field registry.

    The blank twilize template ships with Superstore data: ~48 columns, 22 metadata-records,
    aliases, object-graph, and relation references to "Orders". If the user connects their
    own data, all of these ghost artifacts must be stripped or they appear as NULL fields
    and phantom "Orders (Total)" entries in Tableau.
    """
    ds = ed._datasource

    # 1. Remove all <column> elements from the datasource
    for col in list(ds.findall("column")):
        ds.remove(col)

    # 2. Clear <metadata-records> inside the connection (creates ghost fields like "Orders (Total)")
    for conn in ds.findall(".//connection"):
        for mr in list(conn.findall("metadata-records")):
            conn.remove(mr)

    # 3. Clear <aliases> (template-specific field aliases)
    for aliases in list(ds.findall("aliases")):
        ds.remove(aliases)

    # 4. Clear <object-graph> (template logical table references)
    for og in list(ds.findall("object-graph")):
        ds.remove(og)

    # 5. Clear <semantic-values> (template semantic hints)
    for sv in list(ds.findall("semantic-values")):
        ds.remove(sv)

    # 6. Unregister all pre-loaded fields from the registry
    for field in list(ed.field_registry.all_fields()):
        try:
            ed.field_registry.unregister(field.display_name)
        except Exception:
            pass


def _populate_field_registry(
    ed: TWBEditor,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
) -> None:
    """Register fields in the in-memory field registry only (no XML changes).

    Safe for existing workbooks — does not remove any <column> elements.
    Used by add_sheet_to_existing() to avoid destroying the primary datasource.
    """
    fields_to_register: list[dict] | None = None

    if metadata and metadata.fields:
        fields_to_register = [
            {"name": f.name, "local_name": f.local_name, "type": f.type.value, "role": f.role}
            for f in metadata.fields
        ]

    if fields_to_register:
        # Deduplicate by display name — Metadata API can return the same field
        # twice when it appears in multiple logical tables, and a second
        # registration with a different role/datatype produces a phantom
        # `!` field on the shelves (Tableau can't decide which to use).
        seen_names: set[str] = set()
        for f in fields_to_register:
            if f["name"] in seen_names:
                continue
            seen_names.add(f["name"])
            datatype = _DATATYPE_MAP.get(f["type"], "string")
            role = f["role"] if f["role"] in ("dimension", "measure") else "dimension"
            # Date/datetime dimensions must be ordinal, not nominal
            if datatype == "date" and role == "dimension":
                field_type = "ordinal"
            else:
                field_type = _FIELD_TYPE_MAP.get(role, "nominal")
            # Bind by the PHYSICAL name (sqlproxy binding key) — Tableau references
            # columns internally by the upstream/remote name (e.g. `trip_id`,
            # `region_base`), even when the caption is prettified ("Trip Id",
            # "Region Base"). Emitting the caption form here ([Region Base]) when
            # the actual column is [region_base] creates a phantom duplicate
            # field with a red `!` and leaves the chart blank. The physical name
            # comes from GraphQL `upstreamColumns.name` and is carried in
            # FieldInfo.local_name; when missing (caption == physical) we fall
            # back to the name itself.
            binding = f.get("local_name") or f["name"]
            ed.field_registry.register(
                display_name=f["name"],
                local_name=f"[{binding}]",
                datatype=datatype,
                role=role,
                field_type=field_type,
            )
    else:
        # No metadata — register only viz intent fields
        ed.field_registry.register(viz.x_field, f"[{viz.x_field}]", "string", "dimension", "nominal")
        if viz.y_field:
            ed.field_registry.register(viz.y_field, f"[{viz.y_field}]", "real", "measure", "quantitative")
        if viz.color_field:
            ed.field_registry.register(viz.color_field, f"[{viz.color_field}]", "string", "dimension", "nominal")


def _register_fields(
    ed: TWBEditor,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
) -> None:
    """Clear template artifacts and register fields. For NEW workbooks only."""
    _clear_template_fields(ed)
    _populate_field_registry(ed, viz, metadata)


def _is_preaggregated_calc(viz: VizIntent, field_name: str) -> bool:
    """Check if field_name is a calculated field whose formula already contains aggregations."""
    if not field_name or not viz.calculated_fields:
        return False
    _AGG_FUNCS = ("SUM(", "AVG(", "COUNT(", "COUNTD(", "MIN(", "MAX(",
                  "RUNNING_SUM(", "TOTAL(", "RANK(", "LOOKUP(", "ATTR(",
                  "MEDIAN(", "STDEV(", "VAR(", "PERCENTILE(", "WINDOW_")
    for cf in viz.calculated_fields:
        if cf.name == field_name:
            upper = cf.formula.upper()
            return any(fn in upper for fn in _AGG_FUNCS)
    return False


def _agg_wrap(viz: VizIntent, field_name: str, agg: str) -> str:
    """Wrap field_name in aggregation, or return bare name for pre-aggregated calc fields.

    Tableau cannot nest aggregations (e.g. SUM(SUM([Profit])/SUM([Sales])) → blank chart).
    For calc fields that already contain aggregation functions, we pass the bare field name
    so twilize generates derivation='User' — Tableau treats this as already-aggregated.
    """
    if _is_preaggregated_calc(viz, field_name):
        return field_name
    return f"{agg}({field_name})"


def _build_chart_kwargs(viz: VizIntent, agg: str) -> dict:
    """Return kwargs dict for configure_chart() based on viz type."""
    vtype = viz.viz_type

    y_agg = _agg_wrap(viz, viz.y_field, agg)

    if vtype == "pie":
        # PieChartBuilder: color=dimension, wedge_size=measure. Columns/rows ignored.
        return {"color": viz.x_field, "wedge_size": y_agg, "label": viz.x_field}

    if vtype == "scatter":
        # Circle mark: x on columns, y on rows
        return {"columns": [viz.x_field], "rows": [y_agg], "color": viz.color_field or None}

    if vtype == "area":
        # Same shelf layout as line_chart
        return {"columns": [f"YEAR({viz.x_field})"], "rows": [y_agg], "color": viz.color_field or None}

    if vtype == "heatmap":
        # Square mark: two dimensions as shelves, measure as color intensity
        color_agg = _agg_wrap(viz, viz.color_field or viz.y_field, agg)
        return {"columns": [viz.x_field], "rows": [viz.y_field], "color": color_agg}

    if vtype == "treemap":
        # Tree Map: columns/rows auto-cleared by twilize; size=measure, detail=dimension
        return {"size": y_agg, "detail": viz.x_field, "color": viz.color_field or viz.x_field}

    if vtype == "line_chart":
        return {"columns": [f"YEAR({viz.x_field})"], "rows": [y_agg], "color": viz.color_field or None}

    if vtype == "text":
        # Text table: rows=row dimension, columns=optional column dimension, label=measure
        cols = [viz.color_field] if viz.color_field else []
        return {"columns": cols, "rows": [viz.x_field], "label": y_agg}

    if vtype == "gantt":
        # Gantt Bar: columns=start date, rows=task/category, size=duration measure
        # color_field holds the start date field; x_field is the task dimension
        start_date = viz.color_field or viz.x_field
        return {"columns": [start_date], "rows": [viz.x_field], "size": y_agg}

    # bar_chart and anything else
    return {
        "columns": [viz.x_field],
        "rows": [y_agg],
        "color": viz.color_field or None,
        "sort_descending": y_agg if viz.sort == "descending" else None,
    }


def resolve_filter_display_type(filter_spec, metadata=None):
    """Resolve the display type for a filter card based on field type and cardinality.

    Mutates filter_spec in place: sets display_type and show_filter_card.
    """
    # top_n / bottom_n → no visible card
    if filter_spec.op in ("top_n", "bottom_n"):
        filter_spec.show_filter_card = False
        filter_spec.display_type = None
        return

    # Look up field info from metadata
    field_info = None
    if metadata and "fields" in metadata:
        for f in metadata["fields"]:
            if f.get("name") == filter_spec.field:
                field_info = f
                break

    if field_info is None:
        # Unknown field → safe default
        filter_spec.display_type = "dropdown_search"
        return

    field_type = field_info.get("type", "string")
    field_role = field_info.get("role", "dimension")

    # Date fields → relative_date
    if field_type in ("date", "datetime"):
        filter_spec.display_type = "relative_date"
        return

    # Numeric measures → slider_range
    if field_role == "measure" and field_type in ("int", "float", "real", "number", "integer"):
        filter_spec.display_type = "slider_range"
        return

    # Dimensions — estimate distinct values
    distinct_count = field_info.get("distinct_count")

    # eq with exclude → single_value_list
    if filter_spec.op == "eq" and hasattr(filter_spec, 'exclude') and getattr(filter_spec, 'exclude', False):
        filter_spec.display_type = "single_value_list"
        return

    if distinct_count is not None:
        if distinct_count <= 5:
            filter_spec.display_type = "single_value_list"
        elif distinct_count <= 15:
            filter_spec.display_type = "dropdown"
        else:
            filter_spec.display_type = "dropdown_search"
    else:
        filter_spec.display_type = "dropdown_search"


def _build_filters_for_twilize(viz: VizIntent) -> list[dict] | None:
    """Convert VizIntent FilterSpec list to twilize filter dicts."""
    if not viz.filters:
        return None

    agg = (viz.aggregation or "SUM").upper()
    twilize_filters = []
    for f in viz.filters:
        op = f.op
        field = f.field

        if op in ("top_n", "bottom_n"):
            # FIX-047: the rank-by measure must be AGGREGATED. Passing a bare field
            # name made twilize emit `NONE([field])` as the order expression, which
            # Tableau cannot rank by — the published view either errored on query
            # ("problem querying the data") or returned every row unfiltered. Wrap
            # `by` in the chart aggregation so twilize emits `SUM([field])`.
            by_field = f.by or viz.y_field
            by_expr = _agg_wrap(viz, by_field, agg) if by_field else None
            default_n = 10 if op == "top_n" else 5
            twilize_filters.append({
                "column": field,
                "top": int(f.value) if f.value is not None else default_n,
                "by": by_expr,
                # bottom_n = the N SMALLEST: twilize always emits end="top", so an
                # ASC ordering takes the lowest-ranked N (verified on the server).
                "direction": "DESC" if op == "top_n" else "ASC",
            })
        elif op in ("gt", "gte"):
            twilize_filters.append({
                "column": field,
                "type": "quantitative",
                "min": str(f.value if f.value is not None else 0),
            })
        elif op in ("lt", "lte"):
            twilize_filters.append({
                "column": field,
                "type": "quantitative",
                "max": str(f.value if f.value is not None else 0),
            })
        elif op == "between":
            twilize_filters.append({
                "column": field,
                "type": "quantitative",
                "min": str(f.min if f.min is not None else 0),
                "max": str(f.max if f.max is not None else 0),
            })
        elif op == "eq":
            val = f.value
            if isinstance(val, bool):
                val = str(val).lower()
            elif val is not None:
                val = str(val)
            twilize_filters.append({
                "column": field,
                "values": [val] if val is not None else [],
            })
        elif op == "in":
            twilize_filters.append({
                "column": field,
                "values": f.values or [],
            })
        elif op in ("neq", "not_in"):
            # Exclusion: twilize has no native exclude filter. Emit the same empty
            # categorical placeholder as not_null; the post-save pass
            # _apply_exclude_filters rewrites it to except(all-members, excluded)
            # — the form Tableau uses for an exclude filter (FIX-055).
            twilize_filters.append({
                "column": field,
                "values": [],
            })
        elif op == "date_range":
            # Explicit date span ("de mars à juin 2025") → quantitative date range,
            # the exact form the `year` op already uses (verified on the Server).
            spec: dict = {"column": field, "type": "quantitative"}
            if f.date_min:
                spec["min"] = f"#{f.date_min}#"
            if f.date_max:
                spec["max"] = f"#{f.date_max}#"
            if len(spec) > 2:  # at least one bound — otherwise it's a no-op, skip
                twilize_filters.append(spec)
        elif op == "year":
            # Year → quantitative date range over the full calendar year. Defaults
            # to the current year (not a hardcoded 2024) when the LLM omits a value.
            val = int(f.value) if f.value is not None else date.today().year
            twilize_filters.append({
                "column": field,
                "type": "quantitative",
                "min": f"#{val}-01-01#",
                "max": f"#{val}-12-31#",
            })
        elif op in ("quarter", "month"):
            # FIX-048: month/quarter no longer hardcode year=2024 (which silently
            # returned the wrong year's data or nothing). A bare "Q1"/"June" has no
            # year, so filter on the DATE PART — QUARTER()/MONTH() — which is
            # unambiguous and matches all years. twilize parses these date-part
            # expressions into [qr:field:ok] / [mn:field:ok] discrete instances.
            part = int(f.value) if f.value is not None else 1
            fn = "QUARTER" if op == "quarter" else "MONTH"
            twilize_filters.append({
                "column": f"{fn}({field})",
                "values": [str(part)],
            })
        elif op in ("last_n_days", "last_n_months"):
            # FIX-049: was a no-op (values:[] → "show all"). Emit a concrete
            # quantitative date range ending today so the data is actually
            # restricted. (A true auto-updating relative-date filter is a v1
            # follow-up; a concrete window already guarantees the data changes.)
            n = int(f.value) if f.value is not None else (30 if op == "last_n_days" else 6)
            today = date.today()
            start = today - timedelta(days=n) if op == "last_n_days" else _subtract_months(today, n)
            twilize_filters.append({
                "column": field,
                "type": "quantitative",
                "min": f"#{start.isoformat()}#",
                "max": f"#{today.isoformat()}#",
            })
        elif op == "not_null":
            # Placeholder categorical filter; rewritten to a real "exclude null"
            # filter post-save by _apply_not_null_filters (FIX-050). twilize has no
            # native non-null filter, and an empty categorical is a no-op on its own.
            twilize_filters.append({
                "column": field,
                "values": [],
            })

    return twilize_filters if twilize_filters else None


def _subtract_months(d: date, months: int) -> date:
    """Return the date `months` months before `d` (clamping the day to month length)."""
    total = (d.year * 12 + (d.month - 1)) - months
    year, month = divmod(total, 12)
    month += 1
    # clamp day (e.g. Mar 31 - 1 month → Feb 28/29)
    for day in (d.day, 30, 29, 28):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return date(year, month, 28)


def _show_filter_cards(ed, viz, metadata=None):
    """Add visible filter cards to the worksheet for filters with show_filter_card=True.

    Injects <card type="filter"> elements into <windows>/<window>/<cards>/<edge name="right">.
    """
    if not viz.filters:
        return

    import lxml.etree as ET

    worksheet_name = viz.title or "Sheet 1"

    # Collect filter column references from the worksheet's <view>/<filter> elements
    filter_columns = {}
    for ws in ed.root.iter("worksheet"):
        if ws.get("name") != worksheet_name:
            continue
        for filt in ws.iter("filter"):
            col = filt.get("column", "")
            parts = col.rsplit(".", 1)
            if len(parts) == 2:
                inner = parts[1].strip("[]")
                segments = inner.split(":")
                if len(segments) >= 2:
                    field_name = segments[1]
                    filter_columns[field_name] = col

    # Find the <window> matching our worksheet
    window_elem = None
    for window in ed.root.iter("window"):
        if window.get("name") == worksheet_name:
            window_elem = window
            break

    if window_elem is None:
        return

    # Find <cards>/<edge name="right">
    cards_elem = window_elem.find("cards")
    if cards_elem is None:
        return

    right_edge = None
    for edge in cards_elem.findall("edge"):
        if edge.get("name") == "right":
            right_edge = edge
            break

    if right_edge is None:
        return

    for fspec in viz.filters:
        if not fspec.show_filter_card or not fspec.display_type:
            continue

        col_ref = filter_columns.get(fspec.field)
        if not col_ref:
            continue

        strip = ET.SubElement(right_edge, "strip", attrib={"size": "160"})
        ET.SubElement(strip, "card", attrib={
            "type": "filter",
            "param": col_ref,
        })


_TABLEAU_DATATYPES = {"string", "integer", "real", "boolean", "date", "datetime"}
_DATATYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string",
    "int": "integer", "int64": "integer", "long": "integer", "bigint": "integer",
    "float": "real", "double": "real", "number": "real", "numeric": "real", "decimal": "real",
    "bool": "boolean",
    "timestamp": "datetime",
}


def _normalize_calc_datatype(raw: str | None) -> str:
    """Tableau's XML loader requires lowercase {string|integer|real|boolean|date|datetime}.
    LLM often returns 'STRING', 'float', 'int', etc. — coerce or fall back to 'real'."""
    lower = (raw or "").strip().lower()
    if lower in _TABLEAU_DATATYPES:
        return lower
    return _DATATYPE_ALIASES.get(lower, "real")


def _detect_cross_datasource_calc_fields(
    viz: VizIntent,
    primary_metadata: DataSourceMetadata | None,
    secondary_metadata: DataSourceMetadata | None,
) -> list[str]:
    """Return names of calc fields whose formulas reference secondary-datasource-only fields.

    Pure read — does NOT modify viz. Called upstream (main.py) to surface a
    helpful clarification before generate_twb() is invoked (FIX-044).
    """
    if not viz.calculated_fields or not secondary_metadata or not secondary_metadata.fields:
        return []
    if not primary_metadata or not primary_metadata.fields:
        return []

    primary_names: set[str] = set()
    for f in primary_metadata.fields:
        primary_names.add(f.name.lower())
        if f.local_name:
            primary_names.add(f.local_name.lower())

    secondary_names: set[str] = set()
    for f in secondary_metadata.fields:
        secondary_names.add(f.name.lower())
        if f.local_name:
            secondary_names.add(f.local_name.lower())

    secondary_only = secondary_names - primary_names
    bad: list[str] = []
    for cf in viz.calculated_fields:
        refs = re.findall(r'\[([^\]]+)\]', cf.formula)
        if any(r.lower() in secondary_only for r in refs):
            bad.append(cf.name)
    return bad


def _filter_cross_datasource_calc_fields(
    viz: VizIntent,
    primary_metadata: DataSourceMetadata | None,
    secondary_metadata: DataSourceMetadata | None,
) -> VizIntent:
    """Drop calculated fields whose formulas reference secondary-datasource-only fields.

    Tableau cannot resolve a reference to a secondary datasource field inside a
    primary-datasource calc field formula. The workbook opens with
    "The calculation contains errors" for any such field (FIX-044).

    This is a safety net — SYSTEM_PROMPT already forbids cross-DS formulas,
    but the LLM occasionally ignores that rule when it sees fields from both
    datasources in scope.
    """
    if not viz.calculated_fields or not secondary_metadata or not secondary_metadata.fields:
        return viz
    if not primary_metadata or not primary_metadata.fields:
        return viz

    primary_names: set[str] = set()
    for f in primary_metadata.fields:
        primary_names.add(f.name.lower())
        if f.local_name:
            primary_names.add(f.local_name.lower())

    secondary_names: set[str] = set()
    for f in secondary_metadata.fields:
        secondary_names.add(f.name.lower())
        if f.local_name:
            secondary_names.add(f.local_name.lower())

    secondary_only = secondary_names - primary_names

    import logging
    _log = logging.getLogger(__name__)

    valid = []
    for cf in viz.calculated_fields:
        refs = re.findall(r'\[([^\]]+)\]', cf.formula)
        bad = [r for r in refs if r.lower() in secondary_only]
        if bad:
            _log.warning(
                "FIX-044: dropping calc field '%s' — formula references secondary-only "
                "fields %s. Cross-datasource calc fields are not supported by Tableau.",
                cf.name, bad,
            )
        else:
            valid.append(cf)

    if len(valid) == len(viz.calculated_fields):
        return viz
    return viz.model_copy(update={"calculated_fields": valid})


def _inject_calculated_fields(ed: TWBEditor, viz: VizIntent) -> None:
    """Add calculated fields to the datasource using twilize's native API."""
    for cf in (viz.calculated_fields or []):
        datatype = _normalize_calc_datatype(cf.datatype)
        role = (cf.role or "measure").strip().lower() or "measure"
        if datatype == "boolean":
            role = "dimension"
        ed.add_calculated_field(
            field_name=cf.name,
            formula=cf.formula,
            datatype=datatype,
            role=role,
        )


def _extract_hostname(url: str) -> str:
    """Extract hostname from URL, stripping protocol and trailing slash.

    Tableau sqlproxy connections require bare hostname (no https://).
    'https://prod-uk-a.online.tableau.com/' → 'prod-uk-a.online.tableau.com'
    """
    host = url.rstrip("/")
    if "://" in host:
        host = host.split("://", 1)[1]
    return host


def _apply_server_datasource(ed: TWBEditor, datasource_name: str, content_url: str) -> None:
    """Wire TWBEditor to a published datasource on Tableau Server/Cloud.

    Uses twilize's set_tableauserver_connection() then patches the XML
    to match the exact sqlproxy format from the user's working reference .twb:

      <repository-location site="{site}" id="{contentUrl}"
          path="/t/{site}/datasources"
          derived-from="https://{server}/t/{site}/datasources/{contentUrl}?rev="
          revision="1.0"/>
      <connection class="sqlproxy" channel="https" dataserver-permissions="true"
          dbname="{contentUrl}" directory="dataserver" port="443"
          server="{bare_hostname}" username="">
        <relation connection="sqlproxy.xxx" name="sqlproxy" table="[sqlproxy]" type="table"/>
      </connection>
    """
    from config import settings
    hostname = _extract_hostname(settings.tableau_server_url)
    site_id = settings.tableau_site_id.strip()

    # Step 1: Let twilize do the heavy lifting (removes old connections, etc.)
    ed.set_tableauserver_connection(
        server=hostname,
        dbname=content_url,
        username="",
        table_name=datasource_name,
        port="443",
    )

    # Step 2: Update datasource caption to show actual name
    ed._datasource.set("caption", datasource_name)

    # Step 3: Get datasource name for relation connection attribute
    ds_name = ed._datasource.get("name", "")
    # Will be renamed federated.xxx → sqlproxy.xxx via _patch_sqlproxy_names() after save
    sqlproxy_name = ds_name.replace("federated.", "sqlproxy.", 1) if ds_name.startswith("federated.") else ds_name

    # Step 4: Patch repository-location to match reference .twb format
    repo = ed._datasource.find("repository-location")
    if repo is not None:
        if site_id:
            repo.set("site", site_id)
            repo.set("path", f"/t/{site_id}/datasources")
            repo.set("derived-from", f"https://{hostname}/t/{site_id}/datasources/{content_url}?rev=")
        repo.set("id", content_url)
        repo.set("revision", "1.0")

    # Step 5: Patch connection attributes — exact reference format
    conn = ed._datasource.find("connection")
    if conn is not None:
        conn.set("channel", "https")
        conn.set("dataserver-permissions", "true")
        conn.set("directory", "dataserver")
        conn.set("port", "443")

        # Step 6: Fix relation — must be name='sqlproxy' table='[sqlproxy]'
        for rel in conn.findall(".//relation"):
            rel.set("connection", sqlproxy_name)
            rel.set("name", "sqlproxy")
            rel.set("table", "[sqlproxy]")

    # Also fix object-graph relation if present
    for og_rel in ed._datasource.findall(".//object-graph//relation"):
        og_rel.set("connection", sqlproxy_name)
        og_rel.set("name", "sqlproxy")
        og_rel.set("table", "[sqlproxy]")


def _add_secondary_datasource(ed: TWBEditor, ds_name: str, content_url: str) -> None:
    """Add a second published datasource to an existing workbook.

    Uses the same _apply_server_datasource() function as generate_twb() to ensure
    the secondary datasource has identical XML structure to the primary.
    """
    datasources_elem = ed._datasource.getparent()
    if datasources_elem is None:
        return

    # Check if this datasource already exists
    for existing_ds in datasources_elem.findall("datasource"):
        if existing_ds.get("caption") == ds_name:
            # Already added — point twilize editor to it
            ed._datasource = existing_ds
            return

    # Create a temporary TWBEditor to build a properly-wired datasource element
    # using the same battle-tested _apply_server_datasource() function
    tmp_ed = TWBEditor("")
    _clear_template_fields(tmp_ed)
    _apply_server_datasource(tmp_ed, ds_name, content_url)

    # Extract the datasource element and adopt it into the existing workbook
    new_ds = tmp_ed._datasource

    # CRITICAL: Give the secondary datasource a UNIQUE name.
    # TWBEditor("") always creates datasource name="federated.0ahyg8e1xelf3914bag3r0yukuro".
    # The primary datasource already uses that ID (renamed to sqlproxy.xxx).
    # Without a unique name, _patch_sqlproxy_names() creates a name collision.
    unique_id = f"federated.{uuid.uuid4().hex[:20]}"
    old_name = new_ds.get("name", "")
    new_ds.set("name", unique_id)
    # Also fix the relation connection attribute to match the new name
    for rel in new_ds.findall(".//relation"):
        if rel.get("connection") == old_name:
            rel.set("connection", unique_id)

    datasources_elem.append(new_ds)

    # Point twilize's editor to the new datasource so field registration goes there
    ed._datasource = new_ds


def _detect_linking_fields(schema_a, schema_b) -> list[str]:
    """Detect common field names between two datasource schemas for blending.

    Compares field names (lowercase, stripped) between both schemas.
    Returns list of common field names (using schema_a's original casing).
    """
    names_b = {f.name.lower().strip() for f in schema_b.fields}
    return [f.name for f in schema_a.fields if f.name.lower().strip() in names_b]


# Aggregations that cannot be decomposed into stacked per-linking-value
# segments: a blend always computes at the linking-field granularity
# (FIX-062, measured), so a "bar per category" is really a STACK of
# per-linking-value segments — fine for SUM/COUNT (segments add up to the
# true total), wrong for these (averages/minima don't add).
_NON_ADDITIVE_AGGS = {"AVG", "MIN", "MAX", "MEDIAN"}
# Viz types whose value slot is a single aggregated pill the TOTAL wrap supports.
_TOTAL_WRAP_VIZ_TYPES = {"bar_chart", "line_chart", "area", "pie", "treemap", "text"}


def _wrap_nonadditive_blend_measure(
    viz: VizIntent,
    secondary_metadata: DataSourceMetadata | None,
    linking_fields: list[str] | None,
) -> tuple[VizIntent, str | None]:
    """FIX-063: blend + non-additive aggregation grouped by a secondary-owned
    dimension → wrap the measure in ``TOTAL(<agg>([measure]))``.

    ``TOTAL`` re-aggregates the category's underlying rows, so every
    per-linking-value segment carries the TRUE category-level value (a
    trip-weighted average, not a sum of per-vehicle averages); the post-save
    ``_apply_blend_total_table_calc`` then sets the compute-using (along the
    linking field) and turns mark stacking off so the identical segments
    overlap into what reads as ONE mark per category.

    Fires only when: the aggregation is non-additive, at least one shelf
    dimension is secondary-owned (the cross-granularity case), and the
    measure is NOT secondary-owned — a calc formula cannot reference another
    datasource's field (FIX-044), so a secondary-owned measure keeps the
    honest per-linking-value rendering instead.

    Returns the (possibly rewritten) viz and the calc field name (None = not
    applied).
    """
    agg = (viz.aggregation or "SUM").upper()
    if agg not in _NON_ADDITIVE_AGGS or viz.viz_type not in _TOTAL_WRAP_VIZ_TYPES:
        return viz, None
    if not (viz.y_field and secondary_metadata and secondary_metadata.fields
            and linking_fields):
        return viz, None

    linking_lower = {f.lower() for f in linking_fields}
    sec_index = {f.name.lower(): f for f in secondary_metadata.fields}

    cross_dim = any(
        cap and cap.lower() in sec_index
        and cap.lower() not in linking_lower
        and sec_index[cap.lower()].role == "dimension"
        for cap in (viz.x_field, viz.color_field)
    )
    if not cross_dim:
        return viz, None
    if viz.y_field.lower() in sec_index:
        return viz, None  # secondary-owned measure — formula can't cross DSes

    calc_name = f"{viz.y_field} ({agg})"
    if any(cf.name == calc_name for cf in (viz.calculated_fields or [])):
        return viz, calc_name  # already wrapped (idempotent re-entry)
    calc = CalculatedField(
        name=calc_name,
        formula=f"TOTAL({agg}([{viz.y_field}]))",
        datatype="real",
        role="measure",
    )
    return viz.model_copy(update={
        "y_field": calc_name,
        "calculated_fields": list(viz.calculated_fields or []) + [calc],
    }), calc_name


def _apply_blend_total_table_calc(
    twb_path: str,
    calc_name: str,
    linking_fields: list[str],
    primary_metadata: DataSourceMetadata | None,
    secondary_metadata: DataSourceMetadata | None,
) -> None:
    """Post-save (FIX-063): finish the TOTAL(<agg>) wrap on the LAST worksheet.

    1. The calc pill's ``<column-instance derivation='User'>`` gets
       ``<table-calc ordering-field='[<ds>].[none:<link>:nk]'
       ordering-type='Field'/>`` — "compute using the linking field", i.e.
       partition by every other dimension in the view (the category axis).
       XML shape reverse-engineered from the production workbook
       ``Suivi Quotidien Test 07_04 (6).twb``.
    2. ``<breakdown value='off'/>`` in every pane's ``<view>`` — mark
       stacking in workbook XML is the pane-view ``breakdown`` element
       (``StackingMode-ST`` ∈ off/on/auto per Tableau's official
       ``twb_2026.2.0.xsd``; twilize emits ``auto``). ``off`` makes the
       identical per-linking-value segments overlap into a single visible
       mark instead of stacking to N × value.
    """
    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)

    worksheets = root.findall(".//worksheets/worksheet")
    if not worksheets:
        return
    ws = worksheets[-1]

    # Resolve the calc's XML column id ([Calculation_xxx]) via its caption.
    calc_id = None
    for col in root.findall(".//datasources/datasource/column"):
        if col.get("caption") == calc_name and col.find("calculation") is not None:
            calc_id = col.get("name")
            break
    if not calc_id:
        return

    # The compute-using reference: prefer the exact LOD ref FIX-062 injected
    # (it is guaranteed to exist in this worksheet); fall back to the linking
    # instance declared in the dep blocks.
    link_physicals = [
        _physical_field_name(f, primary_metadata, secondary_metadata)
        for f in (linking_fields or [])
    ]
    ordering_ref = None
    for lod in ws.findall(".//panes/pane/encodings/lod"):
        ref = lod.get("column", "")
        if any(f":{p}:" in ref for p in link_physicals):
            ordering_ref = ref
            break
    if ordering_ref is None:
        for dep in ws.findall(".//datasource-dependencies"):
            ds_name = dep.get("datasource", "")
            for ci in dep.findall("column-instance"):
                nm = ci.get("name", "")
                if any(nm == f"[none:{p}:nk]" for p in link_physicals):
                    ordering_ref = f"[{ds_name}].{nm}"
                    break
            if ordering_ref:
                break
    if ordering_ref is None:
        return

    changed = False
    for ci in ws.findall(".//datasource-dependencies/column-instance"):
        if ci.get("column") == calc_id and ci.get("derivation") == "User":
            if ci.find("table-calc") is None:
                etree.SubElement(ci, "table-calc", attrib={
                    "ordering-field": ordering_ref,
                    "ordering-type": "Field",
                })
                changed = True

    # Stack marks off — the pane-view <breakdown> element (NOT a style rule;
    # a <style-rule element='mark'><format attr='stack-marks'/> is silently
    # ignored — measured: the bars kept stacking to N × value).
    for pane in ws.findall(".//panes/pane"):
        pview = pane.find("view")
        if pview is None:
            pview = etree.Element("view")
            pane.insert(0, pview)
        breakdown = pview.find("breakdown")
        if breakdown is None:
            breakdown = etree.SubElement(pview, "breakdown")
        if breakdown.get("value") != "off":
            breakdown.set("value", "off")
            changed = True

    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8"))


def _physical_field_name(caption: str, *schemas: DataSourceMetadata | None) -> str:
    """Resolve a field caption to its physical/binding name (FIX-002 convention).

    Published (sqlproxy) datasources bind references by the physical name
    (``FieldInfo.local_name``), never the GraphQL caption. Searches the given
    schemas in order (first match wins) and falls back to the caption itself
    when no schema knows the field — for never-renamed fields the two are
    identical, so the fallback is safe.
    """
    low = caption.lower()
    for schema in schemas:
        if not schema or not schema.fields:
            continue
        for f in schema.fields:
            if f.name.lower() == low or (f.local_name or "").lower() == low:
                return f.local_name or f.name
    return caption


def _inject_datasource_relationships(
    workbook_root,
    primary_ds_name: str,
    secondary_ds_name: str,
    linking_fields: list[str],
    primary_metadata: DataSourceMetadata | None = None,
    secondary_metadata: DataSourceMetadata | None = None,
) -> None:
    """Inject (or augment) the workbook-level <datasource-relationships> block that
    declares the blend link between two published datasources.

    This is the XML pattern modern Tableau produces (verified against a real
    published workbook): an in-workbook block — sibling of <datasources>,
    <worksheets>, <windows> — that declares the link column on both datasources
    and a directed <datasource-relationship> with a <column-mapping>/<map>.

    Without this block, Tableau Cloud's parser cannot resolve cross-datasource
    field references and rejects the workbook with a 500000 INTERNAL_SERVER_ERROR
    (which Tableau confusingly surfaces as "User does not have permission for
    action."). The deprecated <blended-columns>/<blended-link> per-worksheet
    pattern we used before is NOT how Tableau emits blends today; it leaves the
    relationship undefined and produces the 500000.

    FIX-061: the linking columns MUST be declared by their PHYSICAL name
    (``local_name``, e.g. ``vehicle_id``), not the GraphQL caption
    (``Vehicle Id``) — same rule as every shelf reference (FIX-002). A
    caption-named column doesn't exist in either published datasource, so
    Tableau shows it as a phantom red-``!`` duplicate in the data pane and the
    declared <column-mapping> links two phantoms: the REAL linking field in
    the view (``[none:vehicle_id:nk]``, on Detail via FIX-043) is never
    mapped, the blend stays broken, and a secondary-DS filter silently stops
    filtering. ``primary_metadata`` / ``secondary_metadata`` provide the
    caption→physical lookup per side; stale caption-form entries left by
    earlier turns are purged so re-published session workbooks heal.

    Names use `federated.xxx` here — `_patch_sqlproxy_names()` rewrites them to
    `sqlproxy.xxx` post-save so they match the published-datasource convention.

    For now linking fields default to nominal/categorical (`:nk]`); date/ordinal
    join keys are out of scope and would require a field-type lookup.
    """
    from lxml import etree as _etree

    if not linking_fields:
        return

    rels = workbook_root.find("datasource-relationships")
    if rels is None:
        rels = _etree.SubElement(workbook_root, "datasource-relationships")
        datasources_elem = workbook_root.find("datasources")
        if datasources_elem is not None:
            datasources_elem.addnext(rels)

    # caption → (physical-in-primary, physical-in-secondary)
    physicals: dict[str, tuple[str, str]] = {
        field: (
            _physical_field_name(field, primary_metadata, secondary_metadata),
            _physical_field_name(field, secondary_metadata, primary_metadata),
        )
        for field in linking_fields
    }

    def _purge_stale_caption_refs(caption: str) -> None:
        # FIX-061 healing: earlier turns injected the linking field by its
        # caption — a column that doesn't exist in the published datasource.
        # Remove those phantom declarations (and their mappings) so an
        # accumulated session workbook recovers on its next republish.
        stale_col = f"[{caption}]"
        stale_inst = f"[none:{caption}:nk]"
        for dep in rels.findall("datasource-dependencies"):
            for col in list(dep.findall("column")):
                if col.get("name") == stale_col:
                    dep.remove(col)
            for ci in list(dep.findall("column-instance")):
                if ci.get("name") == stale_inst:
                    dep.remove(ci)
        for rel in rels.findall("datasource-relationship"):
            mapping_el = rel.find("column-mapping")
            if mapping_el is None:
                continue
            for m in list(mapping_el.findall("map")):
                if stale_inst in m.get("key", "") or stale_inst in m.get("value", ""):
                    mapping_el.remove(m)

    def _ensure_dependencies_block(ds_name: str):
        for dep in rels.findall("datasource-dependencies"):
            if dep.get("datasource") == ds_name:
                return dep
        return _etree.SubElement(rels, "datasource-dependencies", datasource=ds_name)

    def _ensure_field_in_dependencies(dep_block, caption: str, physical: str):
        bracketed = f"[{physical}]"
        if not any(col.get("name") == bracketed for col in dep_block.findall("column")):
            _etree.SubElement(dep_block, "column", attrib={
                "caption": caption,
                "datatype": "string",
                "name": bracketed,
                "role": "dimension",
                "type": "nominal",
            })
        inst_name = f"[none:{physical}:nk]"
        if not any(ci.get("name") == inst_name for ci in dep_block.findall("column-instance")):
            _etree.SubElement(dep_block, "column-instance", attrib={
                "column": bracketed,
                "derivation": "None",
                "name": inst_name,
                "pivot": "key",
                "type": "nominal",
            })

    primary_dep = _ensure_dependencies_block(primary_ds_name)
    secondary_dep = _ensure_dependencies_block(secondary_ds_name)
    for field in linking_fields:
        prim_physical, sec_physical = physicals[field]
        if prim_physical != field or sec_physical != field:
            _purge_stale_caption_refs(field)
        _ensure_field_in_dependencies(primary_dep, field, prim_physical)
        _ensure_field_in_dependencies(secondary_dep, field, sec_physical)

    existing_rel = None
    for rel in rels.findall("datasource-relationship"):
        if rel.get("source") == primary_ds_name and rel.get("target") == secondary_ds_name:
            existing_rel = rel
            break

    if existing_rel is None:
        existing_rel = _etree.SubElement(
            rels, "datasource-relationship",
            source=primary_ds_name, target=secondary_ds_name,
        )

    mapping = existing_rel.find("column-mapping")
    if mapping is None:
        mapping = _etree.SubElement(existing_rel, "column-mapping")

    existing_keys = {m.get("key") for m in mapping.findall("map")}
    for field in linking_fields:
        prim_physical, sec_physical = physicals[field]
        key = f"[{primary_ds_name}].[none:{prim_physical}:nk]"
        if key in existing_keys:
            continue
        _etree.SubElement(mapping, "map", attrib={
            "key": key,
            "value": f"[{secondary_ds_name}].[none:{sec_physical}:nk]",
        })


def _apply_blend_datasources(ed: TWBEditor, primary_name: str, primary_content_url: str,
                              secondary_name: str, secondary_content_url: str,
                              linking_fields: list[str],
                              primary_metadata: DataSourceMetadata | None = None,
                              secondary_metadata: DataSourceMetadata | None = None) -> tuple[str, str]:
    """Wire TWBEditor to two published datasources with blending.

    1. Primary datasource via the standard `_apply_server_datasource` path.
    2. Secondary datasource as a sibling <datasource> with a unique federated id.
    3. Workbook-level <datasource-relationships> block declaring the blend.

    Returns ``(primary_ds_name, secondary_ds_name)`` — the pre-patch XML names
    (still ``federated.xxx``); the caller needs them to rewire the worksheet
    so secondary-owned shelf fields point at the secondary block. See FIX-041.
    """
    from lxml import etree as _etree
    from config import settings

    _apply_server_datasource(ed, primary_name, primary_content_url)
    primary_ds_name = ed._datasource.get("name", "")

    workbook = ed.root
    datasources_elem = workbook.find(".//datasources")
    if datasources_elem is None:
        return

    secondary_ds_name = f"federated.{uuid.uuid4().hex[:20]}"
    sec_ds = _etree.SubElement(datasources_elem, "datasource", attrib={
        "name": secondary_ds_name,
        "caption": secondary_name,
        "inline": "true",
    })
    hostname = _extract_hostname(settings.tableau_server_url)
    site_id = settings.tableau_site_id.strip()
    _etree.SubElement(sec_ds, "repository-location", attrib={
        "id": secondary_content_url,
        "path": f"/t/{site_id}/datasources" if site_id else "/datasources",
        "derived-from": (
            f"https://{hostname}/t/{site_id}/datasources/{secondary_content_url}?rev="
            if site_id else f"https://{hostname}/datasources/{secondary_content_url}?rev="
        ),
        "revision": "1.0",
        **({"site": site_id} if site_id else {}),
    })
    sec_conn = _etree.SubElement(sec_ds, "connection", attrib={
        "channel": "https",
        "class": "sqlproxy",
        "dataserver-permissions": "true",
        "dbname": secondary_content_url,
        "directory": "dataserver",
        "port": "443",
        "server": hostname,
        "username": "",
    })
    _etree.SubElement(sec_conn, "relation", attrib={
        "connection": secondary_ds_name.replace("federated.", "sqlproxy.", 1),
        "name": "sqlproxy",
        "table": "[sqlproxy]",
        "type": "table",
    })

    _inject_datasource_relationships(
        workbook, primary_ds_name, secondary_ds_name, linking_fields,
        primary_metadata=primary_metadata,
        secondary_metadata=secondary_metadata,
    )

    return primary_ds_name, secondary_ds_name


def _rewire_worksheet_for_blend(
    ed: TWBEditor,
    viz: VizIntent,
    primary_ds_name: str,
    secondary_ds_name: str,
    secondary_caption: str,
    secondary_metadata: DataSourceMetadata,
    linking_fields: list[str] | None,
    primary_metadata: DataSourceMetadata | None = None,
) -> None:
    """After ``_configure_worksheet`` emits a single-DS chart that puts every
    field under the primary's ``<datasource-dependencies>`` block, surgically
    move the secondary-owned ones into a dedicated secondary block, declare
    the secondary datasource in the worksheet's ``<view><datasources>``, and
    re-prefix every shelf/encoding/filter reference from
    ``[primary].[xxx]`` to ``[secondary].[xxx]``.

    Without this, a blend chart whose dimension lives in the secondary (e.g.
    ``vehicle_type`` from a ``vehicles`` DS combined with ``cargo_weight_kg``
    from a ``trips`` DS, linked on ``vehicle_id``) is wired against the
    primary only — Tableau Cloud silently substitutes a primary-DS field
    (typically the linking key, here ``vehicle_id``) and the chart renders
    on the wrong dimension.

    Runs BEFORE ``ed.save()`` so the names are still ``federated.xxx`` and
    ``_patch_sqlproxy_names`` renames primary + secondary consistently.
    Linking fields (those shared by both DSes and used as the blend key)
    stay on primary by convention.
    """
    from lxml import etree as _etree

    if not secondary_metadata or not secondary_metadata.fields:
        return

    linking_lower = {f.lower() for f in (linking_fields or [])}
    secondary_index = {f.name.lower(): f for f in secondary_metadata.fields}

    # viz_intent fields that should live on secondary
    candidate_field_captions: list[str] = []
    for cap in (viz.x_field, viz.y_field, viz.color_field):
        if cap:
            candidate_field_captions.append(cap)
    for fspec in (viz.filters or []):
        if fspec.field:
            candidate_field_captions.append(fspec.field)

    secondary_owned_physical: set[str] = set()
    for cap in candidate_field_captions:
        low = cap.lower()
        if low in linking_lower:
            continue  # linking field stays on primary
        if low in secondary_index:
            f = secondary_index[low]
            physical = f.local_name or f.name
            secondary_owned_physical.add(physical)

    if not secondary_owned_physical:
        return

    workbook = ed._tree.getroot() if hasattr(ed, "_tree") else ed.root
    worksheets = workbook.findall(".//worksheet")
    if not worksheets:
        return
    worksheet = worksheets[-1]

    view = worksheet.find(".//view")
    if view is None:
        return

    view_ds_elem = view.find("datasources")
    if view_ds_elem is None:
        view_ds_elem = _etree.SubElement(view, "datasources")

    # Lookup primary's caption from the workbook-level <datasources> block
    primary_caption = None
    for ds in workbook.findall(".//datasources/datasource"):
        if ds.get("name") == primary_ds_name:
            primary_caption = ds.get("caption")
            break

    existing_view_names = {ds.get("name") for ds in view_ds_elem.findall("datasource")}
    if primary_ds_name not in existing_view_names:
        _etree.SubElement(view_ds_elem, "datasource", attrib={
            "caption": primary_caption or "primary",
            "name": primary_ds_name,
        })
    if secondary_ds_name not in existing_view_names:
        _etree.SubElement(view_ds_elem, "datasource", attrib={
            "caption": secondary_caption,
            "name": secondary_ds_name,
        })

    # Find primary's <datasource-dependencies> in the view (if any)
    primary_dep = None
    secondary_dep = None
    for dep in view.findall("datasource-dependencies"):
        ds_attr = dep.get("datasource")
        if ds_attr == primary_ds_name:
            primary_dep = dep
        elif ds_attr == secondary_ds_name:
            secondary_dep = dep

    bracketed_secondary = {f"[{p}]" for p in secondary_owned_physical}
    moved_instance_names: list[str] = []

    def _insert_secondary_dep() -> _etree._Element:
        # The secondary <datasource-dependencies> block MUST sit contiguous
        # with primary's (right after it) — Tableau Cloud stops parsing dep
        # blocks at the first non-dep sibling like <aggregation>, so any block
        # placed after that is silently ignored, the secondary's vehicle_id
        # is never seen, and the blend never engages.
        new_block = _etree.Element("datasource-dependencies", datasource=secondary_ds_name)
        if primary_dep is not None:
            primary_dep.addnext(new_block)
        else:
            view.append(new_block)
        return new_block

    if primary_dep is not None:
        if secondary_dep is None:
            secondary_dep = _insert_secondary_dep()
        # Move <column> elements
        for col in list(primary_dep.findall("column")):
            if col.get("name", "") in bracketed_secondary:
                primary_dep.remove(col)
                secondary_dep.append(col)
        # Move <column-instance> elements
        for ci in list(primary_dep.findall("column-instance")):
            if ci.get("column", "") in bracketed_secondary:
                primary_dep.remove(ci)
                secondary_dep.append(ci)
                inst_name = ci.get("name", "")
                if inst_name:
                    moved_instance_names.append(inst_name)
        # Clean empty primary block
        if (len(primary_dep.findall("column")) == 0
                and len(primary_dep.findall("column-instance")) == 0):
            primary_dep.getparent().remove(primary_dep)

    if not moved_instance_names:
        return

    # Declare the linking field(s) inside BOTH primary and secondary worksheet
    # dep blocks, AND inject them on the chart's Detail mark. Tableau Cloud
    # only engages a blend when the linking field is "used" by the worksheet
    # — workbook-level <datasource-relationships> isn't enough on its own.
    # The Detail mark keeps the chart's semantic shape intact (e.g. "per
    # vehicle type" stays 3 bars, not 100) while making the link active so
    # the user doesn't get the "Blending requires at least one field with
    # the same name in each data source" warning and doesn't have to drag
    # the linking field onto a shelf manually.
    if linking_fields:
        # Re-resolve dep blocks (primary may have been removed if it went empty)
        primary_dep = None
        secondary_dep = None
        for dep in view.findall("datasource-dependencies"):
            ds_attr = dep.get("datasource")
            if ds_attr == primary_ds_name:
                primary_dep = dep
            elif ds_attr == secondary_ds_name:
                secondary_dep = dep
        if primary_dep is None:
            primary_dep = _etree.SubElement(view, "datasource-dependencies", datasource=primary_ds_name)
        if secondary_dep is None:
            secondary_dep = _insert_secondary_dep()

        # Resolve each side's physical name from its OWN schema (FIX-061) —
        # the caption→physical mapping can differ between DSes, and the
        # published datasource only resolves the physical form (FIX-002).
        for link_cap in linking_fields:
            sec_physical = _physical_field_name(link_cap, secondary_metadata, primary_metadata)
            prim_physical = _physical_field_name(link_cap, primary_metadata, secondary_metadata)
            for dep, physical in ((primary_dep, prim_physical), (secondary_dep, sec_physical)):
                bracketed = f"[{physical}]"
                already_has_col = any(
                    c.get("name") == bracketed for c in dep.findall("column")
                )
                if not already_has_col:
                    _etree.SubElement(dep, "column", attrib={
                        "datatype": "string",
                        "name": bracketed,
                        "role": "dimension",
                        "type": "nominal",
                    })
                inst_name = f"[none:{physical}:nk]"
                already_has_inst = any(
                    ci.get("name") == inst_name for ci in dep.findall("column-instance")
                )
                if not already_has_inst:
                    _etree.SubElement(dep, "column-instance", attrib={
                        "column": bracketed,
                        "derivation": "None",
                        "name": inst_name,
                        "pivot": "key",
                        "type": "nominal",
                    })

        # Inject the linking field on the Detail mark of every pane in this
        # worksheet so Tableau Cloud actually engages the blend. The XML
        # element for the Detail shelf is <lod>, NOT <detail> — confirmed in
        # twilize/charts/builder_base.py:449-451 where the `detail` keyword
        # argument writes <lod column="...">. Using <detail> silently no-ops.
        # Uses primary's column reference because the blend's direction is
        # primary→secondary (primary drives the aggregation, secondary
        # contributes the dimension via the link).
        #
        # Placement policy for the linking field (FIX-062, MEASURED 2026-07-08):
        # the Detail LOD is injected whenever the worksheet uses ANY secondary
        # field — dimension, filter, OR measure. A "cleaner" variant that
        # skipped the LOD when the secondary contributed only measures (hoping
        # Tableau would re-aggregate the blended measure per primary category)
        # was CONTRADICTED on the real Server: without the linking field in
        # the view the blend never joins, and every category silently shows
        # the GLOBAL aggregate (harness `blend_secondary_measure_avg`:
        # 'single repeated value 241.95 — measure not joined per brand').
        # Do NOT re-attempt — a chart that looks clean but shows the same
        # wrong number everywhere is the worst failure mode. The per-linking-
        # value mark granularity this forces is a Tableau blending constraint,
        # not a placement bug.
        #
        # The only situational exception: the LLM already placed the linking
        # field on an active shelf (cols / rows / color / size / label / lod /
        # tooltip — the user asked for a per-link breakdown) → skip, the blend
        # is engaged through the LLM's own choice and adding it on LOD too
        # would duplicate the field reference. The check is on the instance
        # suffix ":<physical>:" so it matches any aggregation form (none,
        # count, countd, ...) and either DS prefix (primary / secondary).
        panes = worksheet.findall(".//table/panes/pane")
        active_refs: list[str] = []
        for shelf_name in ("cols", "rows"):
            shelf_el = worksheet.find(f".//table/{shelf_name}")
            if shelf_el is not None and shelf_el.text:
                active_refs.append(shelf_el.text)
        for pane in panes:
            enc = pane.find("encodings")
            if enc is not None:
                for child in enc:
                    col_ref = child.get("column", "")
                    if col_ref:
                        active_refs.append(col_ref)

        for pane in panes:
            encodings = pane.find("encodings")
            if encodings is None:
                encodings = _etree.SubElement(pane, "encodings")
            for link_cap in linking_fields:
                # The LOD ref binds to the PRIMARY's instance, so resolve the
                # physical name against the primary schema first (FIX-061).
                physical = _physical_field_name(link_cap, primary_metadata, secondary_metadata)
                inst_suffix = f":{physical}:"
                if any(inst_suffix in ref for ref in active_refs):
                    continue  # LLM already placed it; blend engaged
                link_ref = f"[{primary_ds_name}].[none:{physical}:nk]"
                already_on_lod = any(
                    el.get("column") == link_ref for el in encodings.findall("lod")
                )
                if not already_on_lod:
                    _etree.SubElement(encodings, "lod", column=link_ref)

    replacements = [
        (f"[{primary_ds_name}].{inst}", f"[{secondary_ds_name}].{inst}")
        for inst in moved_instance_names
    ]

    def _rewrite(s: str | None) -> str | None:
        if not s:
            return s
        out = s
        for old, new in replacements:
            if old in out:
                out = out.replace(old, new)
        return out

    for elem in worksheet.iter():
        new_text = _rewrite(elem.text)
        if new_text != elem.text:
            elem.text = new_text
        new_tail = _rewrite(elem.tail)
        if new_tail != elem.tail:
            elem.tail = new_tail
        for attr_name, attr_val in list(elem.attrib.items()):
            new_val = _rewrite(attr_val)
            if new_val != attr_val:
                elem.set(attr_name, new_val)


def _configure_worksheet(ed: TWBEditor, viz: VizIntent) -> None:
    worksheet_name = viz.title
    agg = (viz.aggregation or "SUM").upper()

    # Build twilize-compatible filter list
    tw_filters = _build_filters_for_twilize(viz)

    if viz.viz_type == "kpi":
        # Single aggregated number — Text mark, no shelves, measure on Text encoding
        kpi_agg = _agg_wrap(viz, viz.x_field, agg)
        ed.configure_chart(
            worksheet_name=worksheet_name,
            mark_type="Text",
            columns=[],
            rows=[],
            label=kpi_agg,
            filters=tw_filters,
        )
    elif viz.viz_type == "combo":
        # Dual-axis: Bar (primary) + Line (secondary), two measures on rows.
        # color_1=combo_c colors the bars by the second measure (continuous gradient).
        combo_y = _agg_wrap(viz, viz.y_field, agg)
        combo_c = _agg_wrap(viz, viz.color_field or viz.y_field, agg)
        ed.configure_dual_axis(
            worksheet_name=worksheet_name,
            mark_type_1="Bar",
            mark_type_2="Line",
            columns=[viz.x_field],
            rows=[combo_y, combo_c],
            dual_axis_shelf="rows",
            synchronized=True,
            color_1=combo_c,
            filters=tw_filters,
        )
    else:
        mark_type = VIZ_TYPE_TO_MARK.get(viz.viz_type)
        if mark_type is None:
            raise ValueError(
                f"Unsupported viz type '{viz.viz_type}'. Supported: {sorted(VIZ_TYPE_TO_MARK) + ['combo']}"
            )
        kwargs = _build_chart_kwargs(viz, agg)
        kwargs["filters"] = tw_filters
        ed.configure_chart(worksheet_name=worksheet_name, mark_type=mark_type, **kwargs)


def _inject_filter_slices(twb_path: str) -> None:
    """Post-save: add missing <slices> entries so filters actually restrict query results (FIX-051).

    Tableau needs a <slices><column>ref</column></slices> block in each worksheet's
    <view> for every active <filter>. Without it the filter card shows up in the UI
    but the filter predicate is never sent to the datasource — data is unfiltered.
    twilize emits the <filter> element but omits <slices>, so this step injects it.
    """
    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)
    changed = False

    for view in root.iter("view"):
        filters = view.findall("filter")
        if not filters:
            continue
        filter_cols = [f.get("column") for f in filters if f.get("column")]
        if not filter_cols:
            continue

        slices = view.find("slices")
        if slices is None:
            slices = etree.SubElement(view, "slices")
            changed = True

        existing = {c.text for c in slices.findall("column") if c.text}
        for col_ref in filter_cols:
            if col_ref not in existing:
                col_elem = etree.SubElement(slices, "column")
                col_elem.text = col_ref
                existing.add(col_ref)
                changed = True

    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def _patch_sqlproxy_names(twb_path: str) -> None:
    """Post-save: rename all 'federated.xxx' references to 'sqlproxy.xxx' in the .twb.

    Tableau Cloud uses 'sqlproxy.' prefix for published datasource connections.
    Twilize generates 'federated.' prefix. This must be renamed globally in the
    XML (datasource name, worksheet references, rows/cols shelves, etc.).
    """
    content = Path(twb_path).read_text(encoding="utf-8")
    # Match federated.{random_id} and replace with sqlproxy.{same_id}
    patched = re.sub(r'\bfederated\.([a-z0-9]+)', r'sqlproxy.\1', content)
    Path(twb_path).write_text(patched, encoding="utf-8")


def _fix_field_names_for_sqlproxy(twb_path: str, metadata: DataSourceMetadata | None) -> None:
    """Post-save: rewrite any caption-form field reference to the physical binding name.

    The GraphQL Metadata API returns each field's caption (e.g. "Region Base"), but a
    published (sqlproxy) datasource binds references by the field's physical/internal
    name (e.g. "region_base"). When they differ, a reference written with the caption
    won't resolve — Tableau shows a phantom duplicate field (red ``!``) and a broken
    pill, and the chart renders blank.

    Field registration already binds shelf fields by ``local_name`` (the physical
    name), so this is a safety net for any place twilize emits the caption form
    instead — most importantly ``<filter>`` columns, which are passed through by
    caption. It replaces ``[caption]`` / ``:caption:`` / ``:caption]`` with the
    physical name across every XML context. The caption stays on the published
    datasource, so the data pane still shows the friendly name.
    """
    if not metadata or not metadata.fields:
        return

    content = Path(twb_path).read_text(encoding="utf-8")
    changed = False

    for field in metadata.fields:
        physical = field.local_name
        if not physical or physical == field.name:
            continue
        caption = field.name
        # Replace bracketed refs: [Region Base] → [region_base]
        if f"[{caption}]" in content:
            content = content.replace(f"[{caption}]", f"[{physical}]")
            changed = True
        # Replace column-instance name segments: :Region Base: → :region_base:
        if f":{caption}:" in content:
            content = content.replace(f":{caption}:", f":{physical}:")
            changed = True
        # Replace trailing instance context: :Region Base] → :region_base]
        if f":{caption}]" in content:
            content = content.replace(f":{caption}]", f":{physical}]")
            changed = True

    if changed:
        Path(twb_path).write_text(content, encoding="utf-8")


def _fix_quantitative_date_instances(twb_path: str) -> None:
    """Post-save: repair date dimensions used in quantitative range filters.

    twilize converts a column to a quantitative range filter (year/quarter/month →
    in-range min/max) but only rewrites the instance key suffix ``:nk]`` → ``:qk]``
    (nominal). It never handles ``:ok]`` (ordinal), so a date *dimension* gets an
    instance named ``[none:Order Date:ok]`` stamped with ``type="quantitative"`` — an
    internal contradiction Tableau cannot resolve. The result is a phantom duplicate
    field (red ``!``) in the data pane and a broken (red) filter pill.

    Fix: find every ``<column-instance>`` whose ``type`` is quantitative but whose name
    still ends in ``:ok]``, then rename that instance (and every reference to it —
    filters, rows, cols, encodings) to ``:qk]`` so the key suffix matches the type.
    Mirrors twilize's own ``:nk]`` → ``:qk]`` logic in charts/builder_base.py.

    FIX-046 — the rename MUST be scoped per worksheet. A file-wide text replace
    (the previous implementation) corrupts a *different* worksheet that legitimately
    uses the same date field as an ORDINAL continuous-date axis (``[none:Trip Date:ok]``
    on a line/area/combo sheet). When a later turn adds a quantitative date *filter*
    on a second sheet, the global replace renamed BOTH sheets' instances to ``:qk]`` —
    leaving the first sheet's instance named ``:qk]`` while still typed ``ordinal``.
    Two same-named instances with conflicting types ⇒ Tableau's native parser rejects
    the workbook at publish ("worksheet … could not be parsed; the workbook may be
    malformed"). Scoping the rename to the worksheet (and its matching ``<window>``)
    that actually owns the quantitative instance keeps every other sheet's ordinal
    date axis intact.
    """
    content = Path(twb_path).read_text(encoding="utf-8")
    root = etree.fromstring(content.encode("utf-8"))

    def _apply_renames(subtree, renames: dict[str, str]) -> bool:
        """Rewrite instance-name occurrences in attribute values + element text,
        scoped to a single subtree (one worksheet + its window)."""
        changed = False
        for el in subtree.iter():
            for attr, val in list(el.attrib.items()):
                new = val
                for old, rep in renames.items():
                    if old in new:
                        new = new.replace(old, rep)
                if new != val:
                    el.set(attr, new)
                    changed = True
            for textattr in ("text", "tail"):
                cur = getattr(el, textattr)
                if not cur:
                    continue
                new = cur
                for old, rep in renames.items():
                    if old in new:
                        new = new.replace(old, rep)
                if new != cur:
                    setattr(el, textattr, new)
                    changed = True
        return changed

    # Map worksheet name → matching <window> so a worksheet's filter-card refs
    # (under <windows>/<window>/<cards>) get renamed in lockstep with the sheet.
    windows_by_name = {w.get("name"): w for w in root.iter("window") if w.get("name")}

    any_changed = False
    for ws in root.iter("worksheet"):
        renames: dict[str, str] = {}
        for ci in ws.iter("column-instance"):
            if ci.get("type") == "quantitative":
                name = ci.get("name") or ""
                if name.endswith(":ok]"):
                    renames[name] = name[:-4] + ":qk]"
        if not renames:
            continue
        any_changed |= _apply_renames(ws, renames)
        win = windows_by_name.get(ws.get("name"))
        if win is not None:
            any_changed |= _apply_renames(win, renames)

    if any_changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def _apply_not_null_filters(twb_path: str, viz: VizIntent, metadata: DataSourceMetadata | None) -> None:
    """Post-save: turn each ``not_null`` filter into a real "exclude null" filter.

    twilize has no native non-null filter; ``_build_filters_for_twilize`` emits a
    placeholder empty categorical filter (``function="level-members"`` = all members,
    a no-op). This pass rewrites that placeholder to ``except(all-members, null-member)``
    — i.e. every value EXCEPT null — which is what the user asked for (FIX-050).
    Runs after the sqlproxy/physical-name passes so the filter column is already in
    its final ``[sqlproxy.X].[none:<physical>:nk]`` form.
    """
    nn_fields = [f.field for f in (viz.filters or []) if f.op == "not_null"]
    if not nn_fields:
        return

    # Resolve each requested field to its physical name (the sqlproxy binding key).
    name_to_phys: dict[str, str] = {}
    if metadata and metadata.fields:
        for fld in metadata.fields:
            name_to_phys[fld.name.lower()] = (fld.local_name or fld.name)
    tokens = {name_to_phys.get(fn.lower(), fn).lower() for fn in nn_fields}
    tokens |= {fn.lower() for fn in nn_fields}  # also match by caption, just in case

    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)
    USER_NS = "{http://www.tableausoftware.com/xml/user}"
    changed = False

    for filt in root.iter("filter"):
        if filt.get("class") != "categorical":
            continue
        col = filt.get("column") or ""
        m = re.search(r":([^:\]]+):[a-z]k\]$", col)
        if not m or m.group(1).lower() not in tokens:
            continue
        # instance ref without the [sqlproxy.X]. prefix → the groupfilter <level>
        level = "[" + col.split("].[", 1)[-1] if "].[" in col else col
        for child in list(filt):
            filt.remove(child)
        gf_except = etree.SubElement(filt, "groupfilter")
        gf_except.set("function", "except")
        gf_except.set(f"{USER_NS}ui-marker", "filter")
        gf_all = etree.SubElement(gf_except, "groupfilter")
        gf_all.set("function", "level-members")
        gf_all.set("level", level)
        gf_null = etree.SubElement(gf_except, "groupfilter")
        gf_null.set("function", "member")
        gf_null.set("level", level)
        gf_null.set("member", "%null%")
        changed = True

    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def _apply_exclude_filters(twb_path: str, viz: VizIntent, metadata: DataSourceMetadata | None) -> None:
    """Post-save: turn each ``neq`` / ``not_in`` filter into a real Tableau
    exclude filter (FIX-055).

    twilize has no native exclusion; ``_build_filters_for_twilize`` emits the same
    empty categorical placeholder as ``not_null``. This pass rewrites it to
    ``except(all-members, member…)`` — one excluded member directly, several
    wrapped in a ``union`` — which is the exact groupfilter shape Tableau itself
    writes for an "exclude" categorical filter. Same machinery as FIX-050
    (``_apply_not_null_filters``), generalized from the ``%null%`` member to the
    user's excluded values. String members are quoted the way twilize quotes
    include-members (``"value"``); pure-numeric members stay unquoted (Tableau
    rejects quoted numerics on non-string levels — same lesson as FIX-048).
    Runs after the sqlproxy/physical-name passes so the filter column is already
    in its final ``[sqlproxy.X].[none:<physical>:nk]`` form.
    """
    excludes: dict[str, list] = {}
    for f in (viz.filters or []):
        if f.op == "neq" and f.value is not None:
            excludes.setdefault(f.field.lower(), []).append(f.value)
        elif f.op == "not_in" and f.values:
            excludes.setdefault(f.field.lower(), []).extend(f.values)
    if not excludes:
        return

    # Resolve each requested field to its physical name (the sqlproxy binding key).
    name_to_phys: dict[str, str] = {}
    if metadata and metadata.fields:
        for fld in metadata.fields:
            name_to_phys[fld.name.lower()] = (fld.local_name or fld.name)
    token_to_values: dict[str, list] = {}
    for fname, vals in excludes.items():
        token_to_values[name_to_phys.get(fname, fname).lower()] = vals
        token_to_values[fname] = vals  # also match by caption, just in case

    def _member_attr(v) -> str:
        s = str(v).lower() if isinstance(v, bool) else str(v)
        return s if re.fullmatch(r"-?\d+(\.\d+)?", s) else f'"{s}"'

    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)
    USER_NS = "{http://www.tableausoftware.com/xml/user}"
    changed = False

    for filt in root.iter("filter"):
        if filt.get("class") != "categorical":
            continue
        col = filt.get("column") or ""
        m = re.search(r":([^:\]]+):[a-z]k\]$", col)
        if not m or m.group(1).lower() not in token_to_values:
            continue
        values = token_to_values[m.group(1).lower()]
        level = "[" + col.split("].[", 1)[-1] if "].[" in col else col
        for child in list(filt):
            filt.remove(child)
        gf_except = etree.SubElement(filt, "groupfilter")
        gf_except.set("function", "except")
        gf_except.set(f"{USER_NS}ui-domain", "database")
        gf_except.set(f"{USER_NS}ui-enumeration", "exclusive")
        gf_except.set(f"{USER_NS}ui-marker", "enumerate")
        gf_all = etree.SubElement(gf_except, "groupfilter")
        gf_all.set("function", "level-members")
        gf_all.set("level", level)
        if len(values) == 1:
            gf_mem = etree.SubElement(gf_except, "groupfilter")
            gf_mem.set("function", "member")
            gf_mem.set("level", level)
            gf_mem.set("member", _member_attr(values[0]))
        else:
            gf_union = etree.SubElement(gf_except, "groupfilter")
            gf_union.set("function", "union")
            for v in values:
                gf_mem = etree.SubElement(gf_union, "groupfilter")
                gf_mem.set("function", "member")
                gf_mem.set("level", level)
                gf_mem.set("member", _member_attr(v))
        changed = True

    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def _apply_relative_date_filters(twb_path: str, viz: VizIntent, metadata: DataSourceMetadata | None) -> None:
    """Post-save: upgrade ``last_n_days`` / ``last_n_months`` from the FROZEN
    concrete range (FIX-049) to a LIVE Tableau relative-date filter (FIX-059).

    The concrete range guaranteed the filter restricted data but was computed at
    generation time — a "30 derniers jours" chart went stale the next day. This
    pass rewrites the quantitative range filter that _build_filters_for_twilize
    emitted into ``<filter class='relative-date' first-period='-N' last-period='0'
    period-type='day|month' include-future='false' include-null='false'>`` — the
    documented Tableau shape — so the window re-evaluates at view time.
    The concrete range stays the built-in fallback: if this pass finds nothing
    to rewrite, the workbook still ships with the (frozen but correct) range.
    """
    rel: dict[str, tuple[str, int]] = {}
    for f in (viz.filters or []):
        if f.op in ("last_n_days", "last_n_months"):
            n = int(f.value) if f.value is not None else (30 if f.op == "last_n_days" else 6)
            rel[f.field.lower()] = ("day" if f.op == "last_n_days" else "month", n)
    if not rel:
        return

    name_to_phys: dict[str, str] = {}
    if metadata and metadata.fields:
        for fld in metadata.fields:
            name_to_phys[fld.name.lower()] = (fld.local_name or fld.name)
    tokens: dict[str, tuple[str, int]] = {}
    for fname, spec in rel.items():
        tokens[name_to_phys.get(fname, fname).lower()] = spec
        tokens[fname] = spec

    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)
    changed = False
    for filt in root.iter("filter"):
        if filt.get("class") != "quantitative":
            continue
        col = filt.get("column") or ""
        m = re.search(r":([^:\]]+):[a-z]k\]$", col)
        if not m or m.group(1).lower() not in tokens:
            continue
        period, n = tokens[m.group(1).lower()]
        for child in list(filt):
            filt.remove(child)
        for attr in ("min", "max", "included-values"):
            filt.attrib.pop(attr, None)
        filt.set("class", "relative-date")
        filt.set("first-period", str(-n))
        filt.set("last-period", "0")
        filt.set("period-type", period)
        filt.set("include-future", "false")
        filt.set("include-null", "false")
        changed = True
    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def add_dashboard_to_workbook(twb_path: str, title: str, sheet_titles: list[str]) -> str:
    """Inject a dashboard laying out existing worksheets in a grid (C4 —
    "mets les 3 charts dans un dashboard").

    Mirrors the structure of a real production dashboard (reverse-engineered
    from `Suivi Quotidien Test 07_04 (6).twb`): a ``<dashboards><dashboard>``
    block — sibling of <worksheets> — holding <style/>, <size>, and nested
    <zones> (root layout-basic → layout-flow rows → one named zone per sheet),
    plus a ``<window class='dashboard'>`` entry. Zone coordinates are in
    1/1000 % of the dashboard (100000 = 100%). Replaces an existing dashboard
    of the same name (re-running the command updates the layout).
    Returns the dashboard name used."""
    tree = etree.parse(twb_path)
    root = tree.getroot()

    dashboards = root.find("dashboards")
    if dashboards is None:
        worksheets_el = root.find("worksheets")
        if worksheets_el is None:
            raise ValueError("workbook has no <worksheets> block")
        dashboards = etree.Element("dashboards")
        worksheets_el.addnext(dashboards)
    # Idempotent per name: drop a same-named dashboard (and its window) first.
    for d in list(dashboards.findall("dashboard")):
        if d.get("name") == title:
            dashboards.remove(d)
    windows_el = root.find("windows")
    if windows_el is not None:
        for w in list(windows_el.findall("window")):
            if w.get("class") == "dashboard" and w.get("name") == title:
                windows_el.remove(w)

    dash = etree.SubElement(dashboards, "dashboard")
    dash.set("enable-sort-zone-taborder", "true")
    dash.set("name", title)
    etree.SubElement(dash, "style")
    size = etree.SubElement(dash, "size")
    for attr, val in (("maxheight", "800"), ("maxwidth", "1000"),
                      ("minheight", "800"), ("minwidth", "1000")):
        size.set(attr, val)
    zones = etree.SubElement(dash, "zones")

    n = max(1, len(sheet_titles))
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    zone_id = 1

    def _zone(parent, **attrs):
        nonlocal zone_id
        z = etree.SubElement(parent, "zone")
        z.set("id", str(zone_id))
        zone_id += 1
        for k, v in attrs.items():
            z.set(k.replace("_", "-"), str(v))
        return z

    root_zone = _zone(zones, h=100000, w=100000, x=0, y=0, **{"type_v2": "layout-basic"})
    vflow = _zone(root_zone, h=100000, w=100000, x=0, y=0,
                  param="vert", **{"type_v2": "layout-flow"})
    row_h = 100000 // rows
    first_sheet_zone_id: int | None = None
    for r in range(rows):
        row_sheets = sheet_titles[r * cols:(r + 1) * cols]
        hflow = _zone(vflow, h=row_h, w=100000, x=0, y=r * row_h,
                      param="horz", **{"type_v2": "layout-flow"})
        col_w = 100000 // max(1, len(row_sheets))
        for c, sheet in enumerate(row_sheets):
            z = _zone(hflow, h=row_h, w=col_w, x=c * col_w, y=r * row_h, name=sheet)
            if first_sheet_zone_id is None:
                first_sheet_zone_id = int(z.get("id"))
            zs = etree.SubElement(z, "zone-style")
            fmt = etree.SubElement(zs, "format")
            fmt.set("attr", "margin")
            fmt.set("value", "4")

    if windows_el is not None:
        win = etree.SubElement(windows_el, "window")
        win.set("class", "dashboard")
        win.set("name", title)
        # Tableau's native parser requires a <viewpoint> per sheet placed on the
        # dashboard — without it publish fails with "Dashboard references sheet
        # 'X' which has no visual representation in the workbook" (verified on
        # the real Server). Mirrors the production workbook's shape.
        vps = etree.SubElement(win, "viewpoints")
        for sheet in sheet_titles:
            vp = etree.SubElement(vps, "viewpoint")
            vp.set("name", sheet)
        if first_sheet_zone_id is not None:
            active = etree.SubElement(win, "active")
            active.set("id", str(first_sheet_zone_id))

    tree.write(twb_path, xml_declaration=True, encoding="UTF-8")
    return title


# Tableau date-part instance prefixes (MONTH→mn, QUARTER→qr, etc.). A discrete
# filter on one of these takes a NUMERIC member (e.g. month 6, quarter 1).
_DATEPART_PREFIXES = ("mn", "qr", "my", "qy", "wk", "dy", "wd", "md", "mdy", "hr", "mi", "sc")


def _fix_datepart_filter_members(twb_path: str) -> None:
    """Post-save: unquote the member value of a discrete date-part filter.

    twilize quotes every categorical member (``member="&quot;1&quot;"``). For a
    string dimension that's correct, but for a date-part filter (MONTH/QUARTER →
    ``[mn:field:ok]`` / ``[qr:field:ok]``) Tableau requires a NUMERIC member —
    a quoted ``"1"`` makes Tableau reject the filter at publish ("Error parsing
    filter for field 'QUARTER(...)', ignoring filter") and silently drop it, so
    the data is never restricted (FIX-048). Verified on the server: bare level +
    unquoted numeric member is the form Tableau accepts.
    """
    content = Path(twb_path).read_bytes()
    root = etree.fromstring(content)
    changed = False
    for filt in root.iter("filter"):
        if filt.get("class") != "categorical":
            continue
        col = filt.get("column") or ""
        m = re.search(r"\.\[([a-z]+):[^:\]]+:[a-z]k\]$", col)
        if not m or m.group(1) not in _DATEPART_PREFIXES:
            continue
        for gf in filt.iter("groupfilter"):
            mem = gf.get("member")
            if mem and len(mem) >= 2 and mem[0] == '"' and mem[-1] == '"':
                gf.set("member", mem[1:-1])
                changed = True
    if changed:
        Path(twb_path).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8")
        )


def generate_twb(
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
    blend_secondary_content_url: str | None = None,
    blend_secondary_name: str | None = None,
    blend_linking_fields: list[str] | None = None,
    blend_secondary_metadata: DataSourceMetadata | None = None,
) -> tuple[str, Path]:
    """
    Generate a Tableau .twb file using twilize.
    Returns (filename, absolute_path).
    Raises ValueError for unsupported viz_type or validation failure.

    When blend_* params are provided, the workbook wires a primary + secondary
    published datasource and emits the workbook-level
    <datasource-relationships> block declaring the blend (FIX-041).
    ``blend_secondary_metadata`` carries the secondary's field schema so
    ``_rewire_worksheet_for_blend`` can route shelf references to the right DS.
    """
    ed = TWBEditor("")
    primary_ds_name: str | None = None
    secondary_ds_name: str | None = None
    if server_ds_content_url:
        if blend_secondary_content_url and blend_linking_fields:
            primary_ds_name, secondary_ds_name = _apply_blend_datasources(
                ed,
                server_ds_name or "datasource", server_ds_content_url,
                blend_secondary_name or "secondary", blend_secondary_content_url,
                blend_linking_fields,
                primary_metadata=metadata,
                secondary_metadata=blend_secondary_metadata,
            )
        else:
            _apply_server_datasource(ed, server_ds_name or "datasource", server_ds_content_url)
            primary_ds_name = ed._datasource.get("name", "")

    # FIX-044: drop calc fields that reference secondary-only fields before injection
    if blend_secondary_metadata:
        viz = _filter_cross_datasource_calc_fields(viz, metadata, blend_secondary_metadata)

    # FIX-063: non-additive aggregation grouped by a secondary-owned dimension
    # → TOTAL(<agg>) wrap so each per-linking-value segment carries the TRUE
    # category value (finished post-save by _apply_blend_total_table_calc).
    total_calc_name = None
    if blend_secondary_metadata and blend_linking_fields:
        viz, total_calc_name = _wrap_nonadditive_blend_measure(
            viz, blend_secondary_metadata, blend_linking_fields)

    _register_fields(ed, viz, metadata)
    _inject_calculated_fields(ed, viz)

    ed.add_worksheet(viz.title)
    _configure_worksheet(ed, viz)
    _show_filter_cards(ed, viz, metadata.model_dump() if hasattr(metadata, 'model_dump') and metadata else metadata)

    # Worksheet-level blend rewiring: move secondary-owned fields into the
    # secondary's <datasource-dependencies> block and re-prefix their shelf
    # refs from [primary].[xxx] to [secondary].[xxx]. Runs BEFORE save so the
    # federated.xxx names get renamed consistently by _patch_sqlproxy_names.
    if (secondary_ds_name and primary_ds_name and blend_secondary_metadata
            and blend_linking_fields):
        _rewire_worksheet_for_blend(
            ed, viz,
            primary_ds_name=primary_ds_name,
            secondary_ds_name=secondary_ds_name,
            secondary_caption=blend_secondary_name or "secondary",
            secondary_metadata=blend_secondary_metadata,
            linking_fields=blend_linking_fields,
            primary_metadata=metadata,
        )

    settings.output_dir.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}_{viz.viz_type}.twb"
    out_path = settings.output_dir / filename

    try:
        ed.save(str(out_path), validate=True)
    except TWBValidationError as exc:
        raise ValueError(f"twilize validation failed: {exc}") from exc

    # Rename federated.xxx → sqlproxy.xxx for Tableau Cloud compatibility
    if server_ds_content_url:
        _patch_sqlproxy_names(str(out_path))
        # Fix field names that twilize may have normalized (e.g., hyphens → spaces)
        _fix_field_names_for_sqlproxy(str(out_path), metadata)
    # Repair date dimensions in quantitative range filters (:ok] → :qk])
    _fix_quantitative_date_instances(str(out_path))
    # Ensure each filter has a matching <slices> entry so it actually restricts data
    _inject_filter_slices(str(out_path))
    # Unquote numeric members of discrete date-part (month/quarter) filters
    _fix_datepart_filter_members(str(out_path))
    # Rewrite not_null placeholders into real "exclude null" filters
    _apply_not_null_filters(str(out_path), viz, metadata)
    # Rewrite neq/not_in placeholders into real exclude filters (FIX-055)
    _apply_exclude_filters(str(out_path), viz, metadata)
    # Upgrade last_n_* frozen ranges to live relative-date filters (FIX-059)
    _apply_relative_date_filters(str(out_path), viz, metadata)
    # FIX-063: compute-using + stack-marks-off for the TOTAL(<agg>) blend wrap
    if total_calc_name and blend_linking_fields:
        _apply_blend_total_table_calc(
            str(out_path), total_calc_name, blend_linking_fields,
            metadata, blend_secondary_metadata)

    return filename, out_path


def _deduplicate_sheet_name(twb_path: str, desired_name: str) -> str:
    """Return a worksheet name that doesn't collide with any existing sheet (FIX-052).

    Two <worksheet> elements with the same name make Tableau's native parser reject
    the whole workbook ("worksheet … could not be parsed; the workbook may be
    malformed"). A filtered follow-up that the LLM titles identically to the prior
    chart would collide, so append " (2)", " (3)", … on conflict."""
    try:
        existing = {ws.get("name", "") for ws in etree.parse(twb_path).getroot().findall(".//worksheet")}
    except Exception:
        return desired_name
    if desired_name not in existing:
        return desired_name
    suffix = 2
    while f"{desired_name} ({suffix})" in existing:
        suffix += 1
    return f"{desired_name} ({suffix})"


def add_sheet_to_existing(
    twb_path: str,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
    blend_secondary_content_url: str | None = None,
    blend_secondary_name: str | None = None,
    blend_linking_fields: list[str] | None = None,
    blend_secondary_metadata: DataSourceMetadata | None = None,
) -> str:
    """
    Add a new worksheet to an existing .twb file.

    Strategy: generate the new sheet as a standalone TWB using the battle-tested
    generate_twb(), then merge its worksheet + datasource into the existing workbook
    via pure XML manipulation. This avoids twilize internal state issues with
    multi-datasource workbooks.

    For same-datasource sheets: uses twilize directly (safe, no state conflicts).
    For different-datasource sheets, or any sheet requiring a blend: generate
    standalone TWB and XML-merge into existing.
    """
    needs_blend = bool(blend_secondary_content_url and blend_linking_fields)

    # Deduplicate worksheet name against existing sheets so Tableau never sees
    # two worksheets with the same name (e.g. a filtered variant of a prior chart).
    unique_name = _deduplicate_sheet_name(twb_path, viz.title)
    if unique_name != viz.title:
        viz = viz.model_copy(update={"title": unique_name})

    # Determine if this is a same-datasource or different-datasource sheet
    is_different_ds = False
    if server_ds_content_url and server_ds_name:
        existing_tree = etree.parse(twb_path)
        existing_root = existing_tree.getroot()
        for ds in existing_root.findall(".//datasource[@inline='true']"):
            existing_caption = ds.get("caption", "")
            if existing_caption and existing_caption != server_ds_name:
                is_different_ds = True
            break

    if is_different_ds or needs_blend:
        # DIFFERENT DATASOURCE or BLEND: generate standalone TWB, then XML-merge
        return _merge_new_sheet_into_workbook(
            twb_path, viz, metadata,
            server_ds_content_url, server_ds_name,
            blend_secondary_content_url=blend_secondary_content_url,
            blend_secondary_name=blend_secondary_name,
            blend_linking_fields=blend_linking_fields,
            blend_secondary_metadata=blend_secondary_metadata,
        )
    else:
        # SAME DATASOURCE, no blend: use twilize directly (safe path)
        return _add_sheet_same_datasource(
            twb_path, viz, metadata,
            server_ds_content_url, server_ds_name,
        )


def list_worksheet_titles(twb_path: str) -> list[str]:
    """Return the workbook's worksheet names in document order."""
    try:
        root = etree.parse(twb_path).getroot()
    except Exception:
        return []
    return [w.get("name", "") for w in root.findall(".//worksheets/worksheet")]


def delete_sheet_from_workbook(twb_path: str, sheet_title: str) -> bool:
    """Remove ONE worksheet (and its <window>) from the workbook (C2 —
    natural-language "supprime le chart X").

    Refuses to delete the LAST worksheet — Tableau rejects a workbook without
    any worksheet, so the caller must surface that instead. Returns True when
    the sheet was removed."""
    tree = etree.parse(twb_path)
    root = tree.getroot()
    sheets = root.findall(".//worksheets/worksheet")
    if len(sheets) <= 1:
        return False
    removed = False
    for ws in sheets:
        if ws.get("name") == sheet_title:
            ws.getparent().remove(ws)
            removed = True
    if not removed:
        return False
    for win in root.findall(".//windows/window"):
        if win.get("name") == sheet_title:
            win.getparent().remove(win)
    tree.write(twb_path, xml_declaration=True, encoding="UTF-8")
    return True


def rename_sheet_in_workbook(twb_path: str, old_title: str, new_title: str) -> str | None:
    """Rename a worksheet + its <window> (C2 — "renomme la feuille X en Y").

    The new name goes through the FIX-052 dedup so two sheets never share a
    name (Tableau's parser rejects the workbook otherwise). Returns the actual
    new name used, or None when old_title wasn't found."""
    unique = _deduplicate_sheet_name(twb_path, new_title)
    tree = etree.parse(twb_path)
    root = tree.getroot()
    renamed = False
    for ws in root.findall(".//worksheets/worksheet"):
        if ws.get("name") == old_title:
            ws.set("name", unique)
            renamed = True
    if not renamed:
        return None
    for win in root.findall(".//windows/window"):
        if win.get("name") == old_title:
            win.set("name", unique)
    tree.write(twb_path, xml_declaration=True, encoding="UTF-8")
    return unique


def modify_sheet_in_existing(
    twb_path: str,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
    old_title: str | None = None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
    blend_secondary_content_url: str | None = None,
    blend_secondary_name: str | None = None,
    blend_linking_fields: list[str] | None = None,
    blend_secondary_metadata: DataSourceMetadata | None = None,
) -> str:
    """Replace ONE worksheet in-place within a multi-sheet workbook, keeping the others.

    Used for an in-place "modify" — a pure tweak (filter / sort / chart-type) on the
    current chart. It removes the worksheet titled ``old_title`` (and its ``<window>``),
    then re-adds the regenerated worksheet for ``viz`` via the normal accumulation path.
    The shared datasource and every OTHER worksheet are left untouched, so a modify
    never destroys previously generated charts.
    """
    if old_title:
        tree = etree.parse(twb_path)
        root = tree.getroot()
        removed = False
        for ws in root.findall(".//worksheets/worksheet"):
            if ws.get("name") == old_title:
                ws.getparent().remove(ws)
                removed = True
        for win in root.findall(".//windows/window"):
            if win.get("name") == old_title:
                win.getparent().remove(win)
        if removed:
            tree.write(twb_path, xml_declaration=True, encoding="UTF-8")
    return add_sheet_to_existing(
        twb_path, viz, metadata,
        server_ds_content_url=server_ds_content_url,
        server_ds_name=server_ds_name,
        blend_secondary_content_url=blend_secondary_content_url,
        blend_secondary_name=blend_secondary_name,
        blend_linking_fields=blend_linking_fields,
        blend_secondary_metadata=blend_secondary_metadata,
    )


def _add_sheet_same_datasource(
    twb_path: str,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
) -> str:
    """Add a sheet using the same datasource — safe twilize path."""
    ed = TWBEditor.open_existing(twb_path)

    _populate_field_registry(ed, viz, metadata)
    _inject_calculated_fields(ed, viz)

    ed.add_worksheet(viz.title)
    _configure_worksheet(ed, viz)
    _show_filter_cards(ed, viz, metadata.model_dump() if hasattr(metadata, 'model_dump') and metadata else metadata)

    try:
        ed.save(twb_path, validate=True)
    except TWBValidationError as exc:
        raise ValueError(f"twilize validation failed: {exc}") from exc

    _patch_sqlproxy_names(twb_path)
    _fix_field_names_for_sqlproxy(twb_path, metadata)
    _fix_quantitative_date_instances(twb_path)
    _inject_filter_slices(twb_path)
    _fix_datepart_filter_members(twb_path)
    _apply_not_null_filters(twb_path, viz, metadata)
    _apply_exclude_filters(twb_path, viz, metadata)
    _apply_relative_date_filters(twb_path, viz, metadata)
    return twb_path


def _merge_new_sheet_into_workbook(
    existing_twb_path: str,
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
    blend_secondary_content_url: str | None = None,
    blend_secondary_name: str | None = None,
    blend_linking_fields: list[str] | None = None,
    blend_secondary_metadata: DataSourceMetadata | None = None,
) -> str:
    """Generate a standalone TWB for the new sheet, then XML-merge into existing workbook.

    Strategy:
    1. generate_twb() creates a complete, valid single-sheet TWB (battle-tested)
    2. The new TWB uses the template datasource name (sqlproxy.0ahyg8e1xelf3914bag3r0yukuro)
       which collides with the existing workbook's primary datasource
    3. We rename ALL references in the new TWB to a unique ID via text replacement
       (same proven approach as _patch_sqlproxy_names)
    4. Then extract <datasource>, <worksheet>, <window> and insert into existing workbook
    5. When the new sheet uses a blend, also lift its <datasource-relationships>
       block (or rebuild it) into the merged workbook so Tableau Cloud can resolve
       cross-datasource references — see FIX-041.
    """
    needs_blend = bool(blend_secondary_content_url and blend_linking_fields)

    # Step 1: Generate standalone TWB for the new sheet (with blend if needed)
    _new_filename, new_path = generate_twb(
        viz, metadata,
        server_ds_content_url=server_ds_content_url,
        server_ds_name=server_ds_name,
        blend_secondary_content_url=blend_secondary_content_url,
        blend_secondary_name=blend_secondary_name,
        blend_linking_fields=blend_linking_fields,
        blend_secondary_metadata=blend_secondary_metadata,
    )

    # Step 2: Read the new TWB as TEXT and rename datasource to a unique ID
    # The template always uses the same datasource name; rename for collision-safety.
    new_content = Path(str(new_path)).read_text(encoding="utf-8")
    template_ds_name = "sqlproxy.0ahyg8e1xelf3914bag3r0yukuro"
    unique_ds_name = f"sqlproxy.{uuid.uuid4().hex[:20]}"
    new_content = new_content.replace(template_ds_name, unique_ds_name)

    # Step 2.5: Deduplicate blend secondary datasources across turns.
    # Each blend turn generates a fresh UUID for the secondary, but the secondary
    # points to the same Tableau Server datasource (same dbname/content-url).
    # Appending a new <datasource> element with the same caption but a different
    # UUID on every blend turn accumulates duplicates that Tableau Cloud rejects
    # ("Workbook contains a duplicate data source name or caption").
    # Fix: if the existing workbook already has a secondary with the same dbname,
    # rename the new UUID → existing UUID in new_content so all worksheet
    # references point to the existing element, then skip appending it in Step 4.
    existing_content = Path(existing_twb_path).read_text(encoding="utf-8")
    existing_root_scan = etree.fromstring(existing_content.encode("utf-8"))
    existing_dbname_to_ds_name: dict[str, str] = {}
    existing_ds_names: set[str] = set()
    for ds in existing_root_scan.findall(".//datasources/datasource[@inline='true']"):
        nm = ds.get("name", "")
        if nm:
            existing_ds_names.add(nm)
        conn = ds.find(".//connection[@class='sqlproxy']")
        if conn is not None:
            dbname = conn.get("dbname", "")
            if dbname and nm:
                existing_dbname_to_ds_name[dbname] = nm

    # Step 2.5a: Primary datasource deduplication (all turns, blend or not).
    # Each call to _merge_new_sheet_into_workbook gives the new primary a fresh UUID,
    # but if the same Tableau Server datasource (identified by dbname = content_url)
    # is already in the workbook from a previous turn, appending a second element with
    # the same caption causes Tableau to reject the publish with:
    #   "Workbook contains a duplicate data source name or caption: 'sqlproxy.xxx'"
    # Fix: reuse the existing element's name, rename all new-content references to it,
    # and mark it present so Step 4 skips the append.
    if server_ds_content_url and server_ds_content_url in existing_dbname_to_ds_name:
        existing_primary_nm = existing_dbname_to_ds_name[server_ds_content_url]
        if unique_ds_name != existing_primary_nm:
            new_content = new_content.replace(unique_ds_name, existing_primary_nm)
            unique_ds_name = existing_primary_nm  # keep in sync for Step 7 blend injection
        existing_ds_names.add(existing_primary_nm)

    if needs_blend and existing_dbname_to_ds_name:
        # Scan the new standalone for non-primary sqlproxy datasources
        new_root_scan = etree.fromstring(new_content.encode("utf-8"))
        for new_ds in new_root_scan.findall(".//datasources/datasource[@inline='true']"):
            new_nm = new_ds.get("name", "")
            if not new_nm or new_nm == unique_ds_name:
                continue  # skip primary
            conn = new_ds.find(".//connection[@class='sqlproxy']")
            if conn is None:
                continue
            dbname = conn.get("dbname", "")
            if dbname and dbname in existing_dbname_to_ds_name:
                existing_nm = existing_dbname_to_ds_name[dbname]
                if new_nm != existing_nm:
                    # Rename every occurrence of the new UUID → existing UUID so
                    # the new worksheet's datasource references resolve correctly.
                    new_content = new_content.replace(new_nm, existing_nm)
                # Either way: the datasource is already in the workbook; skip append.
                existing_ds_names.add(existing_nm)

    # Step 3: Parse both TWBs as XML
    existing_root = etree.fromstring(existing_content.encode("utf-8"))
    new_root = etree.fromstring(new_content.encode("utf-8"))

    # Capture the blend secondary before appending datasource nodes below.
    # lxml moves elements between parents, so after Step 4 the standalone root
    # no longer has the datasource nodes we need to rebuild the relationship.
    new_secondary_name = None
    if needs_blend:
        for ds in new_root.findall("./datasources/datasource[@inline='true']"):
            ds_name = ds.get("name", "")
            if ds_name != unique_ds_name and ds_name.startswith(("federated.", "sqlproxy.")):
                new_secondary_name = ds_name
                break

    # Step 4: Copy each new <datasource inline='true'> into the existing workbook.
    # When a blend is involved the standalone has BOTH primary and secondary — copy both.
    # Skip any datasource whose name already exists (deduplication from Step 2.5).
    existing_datasources = existing_root.find(".//datasources")
    if existing_datasources is not None:
        for new_ds in new_root.findall(".//datasources/datasource[@inline='true']"):
            if new_ds.get("name", "") not in existing_ds_names:
                existing_datasources.append(new_ds)

    # Step 4b: calculated-field <column> defs live INSIDE the datasource
    # element. When Step 2.5/2.5a reuses the session's existing datasource
    # element (same dbname → element NOT appended above), the new sheet's calc
    # columns must be copied over, or the new worksheet references an
    # undefined [Calculation_xxx] → red pill (FIX-063 TOTAL wrap, and any
    # LLM calc field on a blend turn).
    for new_ds in new_root.findall(".//datasources/datasource[@inline='true']"):
        nm = new_ds.get("name", "")
        if not nm or nm not in existing_ds_names:
            continue  # element was appended whole in Step 4 (calc cols travel with it)
        existing_el = next(
            (d for d in existing_root.findall(".//datasources/datasource")
             if d.get("name") == nm), None)
        if existing_el is None:
            continue
        have = {c.get("name") for c in existing_el.findall("column")}
        for col in new_ds.findall("column"):
            if col.find("calculation") is not None and col.get("name") not in have:
                existing_el.append(col)

    # Step 5: Copy the new worksheet into the existing workbook
    existing_worksheets = existing_root.find(".//worksheets")
    new_worksheet = new_root.find(".//worksheet")
    if existing_worksheets is not None and new_worksheet is not None:
        existing_worksheets.append(new_worksheet)

    # Step 6: Copy the new window into the existing workbook
    existing_windows = existing_root.find(".//windows")
    new_window = new_root.find(".//windows/window")
    if existing_windows is not None and new_window is not None:
        existing_windows.append(new_window)

    # Step 7: Merge the workbook-level <datasource-relationships> blend declaration.
    # We rebuild it cleanly from the params (so it survives even when the standalone's
    # block uses datasource ids that we just renamed in Step 2).
    if needs_blend and new_secondary_name:
        _inject_datasource_relationships(
            existing_root,
            unique_ds_name, new_secondary_name,
            blend_linking_fields,
            primary_metadata=metadata,
            secondary_metadata=blend_secondary_metadata,
        )

    # Step 8: Write the merged workbook
    merged = etree.tostring(existing_root, xml_declaration=True, encoding="UTF-8")
    Path(existing_twb_path).write_bytes(merged)

    # Clean up the standalone TWB
    try:
        new_path.unlink(missing_ok=True)
    except Exception:
        pass

    return existing_twb_path


def generate_multi_sheet_twb(
    viz_intents: list[VizIntent],
    metadata: DataSourceMetadata | None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
    blend_secondary_content_url: str | None = None,
    blend_secondary_name: str | None = None,
    blend_linking_fields: list[str] | None = None,
) -> tuple[str, Path]:
    """
    Generate a single .twb with one worksheet per VizIntent.
    Returns (filename, absolute_path).
    """
    if not viz_intents:
        raise ValueError("No viz intents to generate")

    ed = TWBEditor("")
    if server_ds_content_url:
        if blend_secondary_content_url and blend_linking_fields:
            # No secondary schema in this signature — the helper falls back to
            # the primary's physical name (linking captions exist in both DSes).
            _apply_blend_datasources(
                ed, server_ds_name or "datasource", server_ds_content_url,
                blend_secondary_name or "secondary", blend_secondary_content_url,
                blend_linking_fields,
                primary_metadata=metadata,
            )
        else:
            _apply_server_datasource(ed, server_ds_name or "datasource", server_ds_content_url)

    # Register all fields up front (metadata covers all intents; fallback iterates each)
    if metadata and metadata.fields:
        _register_fields(ed, viz_intents[0], metadata)
    else:
        for viz in viz_intents:
            _register_fields(ed, viz, None)

    used_names: set[str] = set()
    for viz in viz_intents:
        if viz.viz_type not in ("combo", "kpi") and VIZ_TYPE_TO_MARK.get(viz.viz_type) is None:
            continue  # skip unsupported types silently

        # Deduplicate worksheet names — Tableau requires unique names
        ws_name = viz.title
        if ws_name in used_names:
            suffix = 2
            while f"{ws_name} ({suffix})" in used_names:
                suffix += 1
            ws_name = f"{ws_name} ({suffix})"
            viz = viz.model_copy(update={"title": ws_name})
        used_names.add(ws_name)

        _inject_calculated_fields(ed, viz)
        ed.add_worksheet(viz.title)
        _configure_worksheet(ed, viz)
        _show_filter_cards(ed, viz, metadata.model_dump() if hasattr(metadata, 'model_dump') and metadata else metadata)

    settings.output_dir.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}_workbook.twb"
    out_path = settings.output_dir / filename

    try:
        ed.save(str(out_path), validate=True)
    except TWBValidationError as exc:
        raise ValueError(f"twilize validation failed: {exc}") from exc

    if server_ds_content_url:
        _patch_sqlproxy_names(str(out_path))
        _fix_field_names_for_sqlproxy(str(out_path), metadata)
    _fix_quantitative_date_instances(str(out_path))
    _inject_filter_slices(str(out_path))
    _fix_datepart_filter_members(str(out_path))
    for _viz in viz_intents:
        _apply_not_null_filters(str(out_path), _viz, metadata)
        _apply_exclude_filters(str(out_path), _viz, metadata)
        _apply_relative_date_filters(str(out_path), _viz, metadata)

    return filename, out_path


def patch_twb(
    twb_path: str,
    viz: VizIntent,
    metadata: DataSourceMetadata | None = None,
    server_ds_content_url: str | None = None,
    server_ds_name: str | None = None,
) -> tuple[str, Path]:
    """
    Apply delta changes to an existing .twb by regenerating the worksheet.
    Generates a fresh .twb with the updated intent (preserving server datasource wiring).
    Returns (filename, absolute_path) of the patched file.
    """
    return generate_twb(
        viz, metadata,
        server_ds_content_url=server_ds_content_url,
        server_ds_name=server_ds_name,
    )
