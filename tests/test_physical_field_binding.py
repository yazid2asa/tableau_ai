"""Tests for physical-name binding on published (sqlproxy) datasources.

The GraphQL Metadata API returns a field's *caption* (e.g. "Sub Category"), but a
published datasource binds references by the field's *physical* name (e.g.
"Sub_Category"). FieldInfo.local_name carries that physical name; generation must
bind every reference (column def, column-instance, shelves, filters) to it, or
Tableau renders a phantom duplicate field (red "!") and a broken filter pill.
"""
from lxml import etree

import tableau_server
import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, FilterSpec, VizIntent


META = DataSourceMetadata(
    datasource_name="order",
    fields=[
        FieldInfo(name="Sub Category", type=FieldType.STRING, role="dimension", local_name="Sub_Category"),
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),  # caption == physical
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension", local_name="Order_Date"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
    ],
    luid="ds-1",
)


def _viz() -> VizIntent:
    return VizIntent(
        viz_type="bar_chart",
        title="Sales by Sub Category in West",
        x_field="Sub Category",
        y_field="Sales",
        filters=[
            FilterSpec(field="Region", op="eq", value="West"),
            FilterSpec(field="Order Date", op="year", value=2024),
        ],
        aggregation="SUM",
        action="new",
    )


def test_shelf_and_instances_bind_to_physical_name():
    _fn, out = twb_generator.generate_twb(
        _viz(), META, server_ds_content_url="order", server_ds_name="order",
    )
    try:
        content = out.read_text(encoding="utf-8")
        # No caption-form reference survives anywhere
        assert "[Sub Category]" not in content
        assert "[Order Date]" not in content
        # Physical binding for the dimension column def + instance
        assert "[Sub_Category]" in content
        assert "[none:Sub_Category:nk]" in content
        # Date dimension binds physical AND keeps the :qk] consistency fix
        assert "[Order_Date]" in content
        assert "[none:Order_Date:qk]" in content
    finally:
        out.unlink(missing_ok=True)


def test_filter_binds_to_physical_name():
    _fn, out = twb_generator.generate_twb(
        _viz(), META, server_ds_content_url="order", server_ds_name="order",
    )
    try:
        root = etree.parse(str(out)).getroot()
        date_filters = [
            f.get("column", "") for f in root.iter("filter")
            if "Order" in (f.get("column") or "")
        ]
        assert date_filters, "expected a date filter"
        assert all("Order_Date" in c for c in date_filters)
        assert all("Order Date" not in c for c in date_filters)
    finally:
        out.unlink(missing_ok=True)


def test_field_without_local_name_is_unchanged():
    """A field whose caption equals its physical name must not be rewritten."""
    meta = DataSourceMetadata(
        datasource_name="order",
        fields=[
            FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
            FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        ],
        luid="ds-2",
    )
    viz = VizIntent(
        viz_type="bar_chart", title="Sales by Category",
        x_field="Category", y_field="Sales", aggregation="SUM", action="new",
    )
    _fn, out = twb_generator.generate_twb(
        viz, meta, server_ds_content_url="order", server_ds_name="order",
    )
    try:
        content = out.read_text(encoding="utf-8")
        assert "[Category]" in content
        assert "[none:Category:nk]" in content
    finally:
        out.unlink(missing_ok=True)


def test_fix_field_names_maps_caption_to_physical(tmp_path):
    """_fix_field_names_for_sqlproxy rewrites leftover caption forms (e.g. filters)."""
    twb = tmp_path / "f.twb"
    twb.write_text(
        '<workbook><filter class="categorical" '
        'column="[sqlproxy.x].[none:Sub Category:nk]"/></workbook>',
        encoding="utf-8",
    )
    meta = DataSourceMetadata(
        datasource_name="order",
        fields=[FieldInfo(name="Sub Category", type=FieldType.STRING,
                          role="dimension", local_name="Sub_Category")],
    )
    twb_generator._fix_field_names_for_sqlproxy(str(twb), meta)
    out = twb.read_text(encoding="utf-8")
    assert "[none:Sub_Category:nk]" in out
    assert "Sub Category" not in out


def test_parse_datasource_nodes_extracts_physical_name():
    """GraphQL upstreamColumns.name becomes FieldInfo.local_name when it differs."""
    nodes = [{
        "luid": "abc",
        "name": "order",
        "fields": [
            {"name": "Sub Category", "isHidden": False, "dataType": "STRING",
             "role": "DIMENSION", "upstreamColumns": [{"name": "Sub_Category"}]},
            {"name": "Sales", "isHidden": False, "dataType": "REAL",
             "role": "MEASURE", "upstreamColumns": [{"name": "Sales"}]},
        ],
    }]
    result = tableau_server._parse_datasource_nodes(nodes)
    fields = {f.name: f for f in result[0].fields}
    assert fields["Sub Category"].local_name == "Sub_Category"   # differs → captured
    assert fields["Sales"].local_name is None                     # same → not set
