"""Phase-2 Class #2 regression tests — field-role / value-slot correctness.

These are deterministic, model-independent guards for the post-LLM correction
(`_validate_and_correct_intent`) and the resulting generated XML. The end-to-end
chart-type-accuracy measurement (LLM in the loop) lives in eval_intent.py; these
pin the deterministic correction that turns a wrong-role LLM intent into a valid
chart — most importantly preventing SUM(<dimension>) (FIX-053), the root of the
reported "list sales and profit by category and region" table that rendered
SUM(Region) as a red/invalid pill.
"""
import os
import lxml.etree as ET

from schemas import VizIntent, DataSourceMetadata, FieldInfo, FieldType
from main import _validate_and_correct_intent

DS = DataSourceMetadata(
    datasource_name="ventes", luid="ventes-luid",
    fields=[
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sub Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Segment", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Customer", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
    ],
)


def _vi(**kw):
    base = dict(title="t", action="new", datasource_luid="ventes-luid")
    base.update(kw)
    return VizIntent(**base)


def _correct(viz, q="question"):
    return _validate_and_correct_intent(viz, [DS], q)


# --------------------------------------------------------------------------
# FIX-053 — value slot must hold a measure (the reported SUM(Region) defect)
# --------------------------------------------------------------------------

def test_text_table_dimension_in_value_slot_swapped_with_measure():
    """text table with y=dimension and color=measure → swap so the measure is the
    value (the 'list sales and profit by category and region' → SUM(Region) fix)."""
    v = _vi(viz_type="text", x_field="Category", y_field="Region", color_field="Profit")
    c = _correct(v, "list sales and profit by category and region")
    assert c.y_field == "Profit"      # measure now in the value slot
    assert c.color_field == "Region"  # 2nd dimension moved to color (columns)


def test_treemap_dimension_in_value_slot_swapped():
    """treemap 'breakdown of profit by category and sub-category' must not put a
    dimension (Sub Category) in the size/value slot."""
    v = _vi(viz_type="treemap", x_field="Category", y_field="Sub Category", color_field="Profit")
    c = _correct(v, "breakdown of profit by category and sub-category")
    assert c.y_field == "Profit"
    assert c.color_field == "Sub Category"


def test_no_measure_available_falls_back_to_countd():
    """Two dimensions and no measure anywhere → count the dimension (COUNTD),
    never SUM(<dimension>)."""
    v = _vi(viz_type="bar_chart", x_field="Category", y_field="Region", aggregation="SUM")
    c = _correct(v, "orders by category and region")
    assert c.aggregation == "COUNTD"


def test_heatmap_keeps_dimension_on_y():
    """heatmap intentionally has y=dimension + color=measure — must NOT be swapped."""
    v = _vi(viz_type="heatmap", x_field="Region", y_field="Category", color_field="Sales")
    c = _correct(v, "heatmap of sales by region and category")
    assert c.y_field == "Category"      # 2nd dimension stays on y
    assert c.color_field == "Sales"     # measure stays on color


def test_countd_of_dimension_left_unchanged():
    """COUNTD(Customer) is a valid value-slot — must not be downgraded/swapped."""
    v = _vi(viz_type="bar_chart", x_field="Region", y_field="Customer", aggregation="COUNTD")
    c = _correct(v, "number of distinct customers by region")
    assert c.y_field == "Customer"
    assert c.aggregation == "COUNTD"


def test_normal_bar_measure_on_y_untouched():
    """A correct intent (dimension x, measure y) is left alone."""
    v = _vi(viz_type="bar_chart", x_field="Region", y_field="Sales")
    c = _correct(v, "sales by region")
    assert c.x_field == "Region" and c.y_field == "Sales"


def test_corrected_text_table_generates_without_sum_of_dimension():
    """End result: the corrected table intent generates a .twb with no SUM(<dimension>)."""
    from twb_generator import generate_twb
    v = _vi(viz_type="text", x_field="Category", y_field="Region", color_field="Profit")
    c = _correct(v, "list sales and profit by category and region")
    _fn, path = generate_twb(c, DS, server_ds_content_url="ventes", server_ds_name="ventes")
    try:
        txt = open(str(path), encoding="utf-8").read().lower()
        for dim in ("region", "category", "sub_category", "segment", "customer"):
            assert f"sum:{dim}" not in txt, f"generated SUM of dimension {dim}"
    finally:
        try:
            os.remove(str(path))
        except OSError:
            pass
