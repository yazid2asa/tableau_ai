"""
test_filter_cards.py — Filter card display type resolution tests.

Covers:
  - Date field -> relative_date card
  - Numeric measure -> slider_range card
  - top_n -> no filter card
  - Large dimension -> dropdown_search
  - Small dimension -> single_value_list
"""
from schemas import FilterSpec
from twb_generator import resolve_filter_display_type


def _meta(fields):
    return {"datasource_name": "superstore", "fields": fields}


def test_date_field_gets_relative_date_card():
    """Date field with op=year resolves to relative_date display type."""
    fs = FilterSpec(field="Order_Date", op="year", value=2024)
    metadata = _meta([
        {"name": "Order_Date", "type": "date", "role": "dimension"},
    ])
    resolve_filter_display_type(fs, metadata)
    assert fs.display_type == "relative_date"
    assert fs.show_filter_card is True


def test_numeric_measure_gets_slider():
    """Numeric measure with op=gt resolves to slider_range display type."""
    fs = FilterSpec(field="Sales", op="gt", value=1000)
    metadata = _meta([
        {"name": "Sales", "type": "float", "role": "measure"},
    ])
    resolve_filter_display_type(fs, metadata)
    assert fs.display_type == "slider_range"


def test_top_n_no_card():
    """top_n filter should hide the filter card entirely."""
    fs = FilterSpec(field="Category", op="top_n", value=10, by="Sales")
    resolve_filter_display_type(fs)
    assert fs.show_filter_card is False


def test_large_dimension_gets_dropdown_search():
    """High-cardinality dimension resolves to dropdown_search."""
    fs = FilterSpec(field="Customer", op="eq", value="John")
    metadata = _meta([
        {"name": "Customer", "type": "string", "role": "dimension", "distinct_count": 500},
    ])
    resolve_filter_display_type(fs, metadata)
    assert fs.display_type == "dropdown_search"


def test_small_dimension_gets_list():
    """Low-cardinality dimension resolves to single_value_list."""
    fs = FilterSpec(field="Category", op="in", values=["Furniture", "Technology"])
    metadata = _meta([
        {"name": "Category", "type": "string", "role": "dimension", "distinct_count": 3},
    ])
    resolve_filter_display_type(fs, metadata)
    assert fs.display_type == "single_value_list"
