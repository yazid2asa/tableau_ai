import re
import uuid
from pathlib import Path

from lxml import etree
from twilize.twb_editor import TWBEditor
from twilize.validator import TWBValidationError

from schemas import VizIntent, DataSourceMetadata, FieldInfo, FieldType
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
    "gantt": "Gantt Bar",
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

    twilize_filters = []
    for f in viz.filters:
        op = f.op
        field = f.field

        if op in ("top_n",):
            twilize_filters.append({
                "column": field,
                "top": f.value or 10,
                "by": f.by or viz.y_field,
                "direction": "DESC",
            })
        elif op in ("bottom_n",):
            twilize_filters.append({
                "column": field,
                "top": f.value or 5,
                "by": f.by or viz.y_field,
                "direction": "ASC",
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
            twilize_filters.append({
                "column": field,
                "values": [f.value] if f.value is not None else [],
            })
        elif op == "in":
            twilize_filters.append({
                "column": field,
                "values": f.values or [],
            })
        elif op in ("year", "quarter", "month"):
            val = int(f.value) if f.value is not None else 2024
            if op == "year":
                twilize_filters.append({
                    "column": field,
                    "type": "quantitative",
                    "min": f"#{val}-01-01#",
                    "max": f"#{val}-12-31#",
                })
            elif op == "quarter":
                q_start_month = (val - 1) * 3 + 1
                q_end_month = val * 3
                year = int(f.year) if hasattr(f, "year") and f.year else 2024
                twilize_filters.append({
                    "column": field,
                    "type": "quantitative",
                    "min": f"#{year}-{q_start_month:02d}-01#",
                    "max": f"#{year}-{q_end_month:02d}-28#",
                })
            else:  # month
                year = 2024
                twilize_filters.append({
                    "column": field,
                    "type": "quantitative",
                    "min": f"#{year}-{val:02d}-01#",
                    "max": f"#{year}-{val:02d}-28#",
                })
        elif op in ("last_n_days", "last_n_months"):
            twilize_filters.append({
                "column": field,
                "values": [],  # Tableau will show all — relative date needs manual XML
            })
        elif op == "not_null":
            twilize_filters.append({
                "column": field,
                "values": [],
            })

    return twilize_filters if twilize_filters else None


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
        ed.add_calculated_field(
            field_name=cf.name,
            formula=cf.formula,
            datatype=_normalize_calc_datatype(cf.datatype),
            role=(cf.role or "measure").strip().lower() or "measure",
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


def _inject_datasource_relationships(
    workbook_root,
    primary_ds_name: str,
    secondary_ds_name: str,
    linking_fields: list[str],
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

    def _ensure_dependencies_block(ds_name: str):
        for dep in rels.findall("datasource-dependencies"):
            if dep.get("datasource") == ds_name:
                return dep
        return _etree.SubElement(rels, "datasource-dependencies", datasource=ds_name)

    def _ensure_field_in_dependencies(dep_block, field: str):
        bracketed = f"[{field}]"
        for col in dep_block.findall("column"):
            if col.get("name") == bracketed:
                return
        _etree.SubElement(dep_block, "column", attrib={
            "caption": field,
            "datatype": "string",
            "name": bracketed,
            "role": "dimension",
            "type": "nominal",
        })
        _etree.SubElement(dep_block, "column-instance", attrib={
            "column": bracketed,
            "derivation": "None",
            "name": f"[none:{field}:nk]",
            "pivot": "key",
            "type": "nominal",
        })

    primary_dep = _ensure_dependencies_block(primary_ds_name)
    secondary_dep = _ensure_dependencies_block(secondary_ds_name)
    for field in linking_fields:
        _ensure_field_in_dependencies(primary_dep, field)
        _ensure_field_in_dependencies(secondary_dep, field)

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
        key = f"[{primary_ds_name}].[none:{field}:nk]"
        if key in existing_keys:
            continue
        _etree.SubElement(mapping, "map", attrib={
            "key": key,
            "value": f"[{secondary_ds_name}].[none:{field}:nk]",
        })


def _apply_blend_datasources(ed: TWBEditor, primary_name: str, primary_content_url: str,
                              secondary_name: str, secondary_content_url: str,
                              linking_fields: list[str]) -> tuple[str, str]:
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

        def _physical_for(name: str, schema: DataSourceMetadata | None) -> str:
            if not schema:
                return name
            for f in schema.fields:
                if f.name.lower() == name.lower():
                    return f.local_name or f.name
            return name

        # Build a primary schema lookup so the linking physical name matches
        # whatever the primary stores (rare but the caption→physical mapping
        # can differ between DSes).
        for link_cap in linking_fields:
            sec_physical = _physical_for(link_cap, secondary_metadata)
            # On the primary side we don't have primary_metadata here, so use
            # the secondary's physical name as a best guess — they share the
            # same caption, so the physical names are almost always identical.
            prim_physical = sec_physical
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
        # SKIP the LOD injection when the LLM has already placed the linking
        # field on ANY active shelf (cols / rows / color / size / label / lod
        # / tooltip / etc.). In those cases the blend is already engaged
        # through the LLM's own choice and adding it on LOD too would
        # duplicate the field reference. The check is on the instance suffix
        # ":<physical>:" so it matches any aggregation form (none, count,
        # countd, ...) and either DS prefix (primary / secondary).
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
                physical = _physical_for(link_cap, secondary_metadata)
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
    """
    content = Path(twb_path).read_text(encoding="utf-8")
    root = etree.fromstring(content.encode("utf-8"))

    renames: dict[str, str] = {}
    for ci in root.iter("column-instance"):
        if ci.get("type") == "quantitative":
            name = ci.get("name") or ""
            if name.endswith(":ok]"):
                renames[name] = name[:-4] + ":qk]"

    if not renames:
        return

    for old, new in renames.items():
        content = content.replace(old, new)
    Path(twb_path).write_text(content, encoding="utf-8")


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
            )
        else:
            _apply_server_datasource(ed, server_ds_name or "datasource", server_ds_content_url)
            primary_ds_name = ed._datasource.get("name", "")

    # FIX-044: drop calc fields that reference secondary-only fields before injection
    if blend_secondary_metadata:
        viz = _filter_cross_datasource_calc_fields(viz, metadata, blend_secondary_metadata)

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

    return filename, out_path


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
            _apply_blend_datasources(
                ed, server_ds_name or "datasource", server_ds_content_url,
                blend_secondary_name or "secondary", blend_secondary_content_url,
                blend_linking_fields,
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
