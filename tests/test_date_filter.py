"""Tests for date-dimension range filters on published datasources.

twilize converts a column to a quantitative range filter (year/quarter/month) but
only rewrites the instance key suffix :nk] -> :qk] (nominal), never :ok] (ordinal).
A date dimension used in such a filter therefore gets an instance named
[none:Order Date:ok] stamped type="quantitative" — an internal contradiction that
Tableau cannot resolve, producing a phantom duplicate field (red '!') and a broken
filter. _fix_quantitative_date_instances() repairs it (:ok] -> :qk]) while keeping
the field's <column> definition intact so the reference still binds.
"""
from lxml import etree

import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, FilterSpec, VizIntent


METADATA = DataSourceMetadata(
    datasource_name="order",
    datasource_caption="order",
    fields=[
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
    ],
    luid="ds-luid-1",
)


def _viz_year_filter() -> VizIntent:
    return VizIntent(
        viz_type="bar_chart",
        title="Sales by Category in 2024",
        x_field="Category",
        y_field="Sales",
        filters=[FilterSpec(field="Order Date", op="year", value=2024)],
        aggregation="SUM",
        action="new",
    )


def test_date_range_filter_instance_is_consistent():
    """No quantitative column-instance keeps an ordinal (:ok]) key suffix."""
    _fn, out_path = twb_generator.generate_twb(
        _viz_year_filter(), METADATA,
        server_ds_content_url="order", server_ds_name="order",
    )
    try:
        root = etree.parse(str(out_path)).getroot()

        # The contradictory state (:ok] + quantitative) must not exist anywhere
        bad = [
            ci for ci in root.iter("column-instance")
            if ci.get("type") == "quantitative" and (ci.get("name") or "").endswith(":ok]")
        ]
        assert not bad, f"found contradictory quantitative :ok] instances: {[c.get('name') for c in bad]}"

        # The Order Date instance is now a quantitative key (:qk])
        date_instances = [
            ci.get("name") for ci in root.iter("column-instance")
            if ci.get("column") == "[Order Date]"
        ]
        assert "[none:Order Date:qk]" in date_instances

        # The field <column> DEFINITION is preserved so the reference resolves
        plain_date_cols = [
            c for c in root.iter("column")
            if c.get("name") == "[Order Date]" and c.find("calculation") is None
        ]
        assert plain_date_cols, "date field <column> definition must remain"

        # The filter references the consistent :qk] instance
        date_filters = [
            f for f in root.iter("filter")
            if "Order Date" in (f.get("column") or "")
        ]
        assert date_filters and all(
            f.get("column", "").endswith(":qk]") for f in date_filters
        )
    finally:
        out_path.unlink(missing_ok=True)


def test_fix_only_touches_contradictory_instances(tmp_path):
    """A legitimate ordinal instance (type='ordinal', :ok]) is left untouched."""
    twb_path = tmp_path / "mixed.twb"
    twb_path.write_text(
        "<workbook><worksheet><table><view><datasource-dependencies>"
        # contradictory: quantitative type but ordinal suffix -> must be renamed
        '<column-instance column="[Order Date]" derivation="None"'
        ' name="[none:Order Date:ok]" pivot="key" type="quantitative"/>'
        # legitimate ordinal instance -> must stay :ok]
        '<column-instance column="[Ship Date]" derivation="None"'
        ' name="[none:Ship Date:ok]" pivot="key" type="ordinal"/>'
        "</datasource-dependencies></view></table></worksheet></workbook>",
        encoding="utf-8",
    )
    twb_generator._fix_quantitative_date_instances(str(twb_path))
    out = twb_path.read_text(encoding="utf-8")
    assert "[none:Order Date:qk]" in out      # contradictory one renamed
    assert "[none:Order Date:ok]" not in out
    assert "[none:Ship Date:ok]" in out        # legitimate ordinal preserved


def test_fix_is_noop_without_date_filters(tmp_path):
    """No quantitative :ok] instances -> file is unchanged."""
    twb_path = tmp_path / "plain.twb"
    twb_path.write_text(
        "<workbook><worksheet><column-instance column=\"[Category]\""
        " name=\"[none:Category:nk]\" type=\"nominal\"/></worksheet></workbook>",
        encoding="utf-8",
    )
    before = twb_path.read_text(encoding="utf-8")
    twb_generator._fix_quantitative_date_instances(str(twb_path))
    assert twb_path.read_text(encoding="utf-8") == before
