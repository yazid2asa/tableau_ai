"""Tests for M11 — Agent Self-Correction (post-LLM validation)."""
import pytest
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent
from main import _validate_and_correct_intent


def _ds(name, fields_spec, luid="luid-1"):
    """Helper: create DataSourceMetadata. fields_spec = [(name, type, role), ...]"""
    fields = [FieldInfo(name=n, type=t, role=r) for n, t, r in fields_spec]
    return DataSourceMetadata(datasource_name=name, luid=luid, fields=fields)


def _intent(**kwargs):
    """Helper: create VizIntent with defaults."""
    defaults = {
        "viz_type": "bar_chart", "title": "Test", "x_field": "Category",
        "y_field": "Sales", "aggregation": "SUM",
    }
    defaults.update(kwargs)
    return VizIntent(**defaults)


# --- Field role enforcement ---

def test_swaps_measure_x_and_dimension_y():
    """If x=measure and y=dimension, they get swapped."""
    ds = _ds("DB", [
        ("Sales", FieldType.FLOAT, "measure"),
        ("Category", FieldType.STRING, "dimension"),
    ])
    intent = _intent(x_field="Sales", y_field="Category", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "sales by category")
    assert result.x_field == "Category"
    assert result.y_field == "Sales"


def test_no_swap_when_both_measures_scatter():
    """Two measures (scatter plot) should NOT be swapped."""
    ds = _ds("DB", [
        ("Sales", FieldType.FLOAT, "measure"),
        ("Profit", FieldType.FLOAT, "measure"),
    ])
    intent = _intent(viz_type="scatter", x_field="Sales", y_field="Profit", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "sales vs profit")
    assert result.x_field == "Sales"
    assert result.y_field == "Profit"


def test_no_swap_when_roles_correct():
    """Correctly assigned roles should not be changed."""
    ds = _ds("DB", [
        ("Category", FieldType.STRING, "dimension"),
        ("Sales", FieldType.FLOAT, "measure"),
    ])
    intent = _intent(x_field="Category", y_field="Sales", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "sales by category")
    assert result.x_field == "Category"
    assert result.y_field == "Sales"


# --- Chart type inference ---

def test_bar_to_line_for_time_series():
    """bar_chart with date x_field + time words -> line_chart."""
    ds = _ds("DB", [
        ("Order Date", FieldType.DATE, "dimension"),
        ("Sales", FieldType.FLOAT, "measure"),
    ])
    intent = _intent(viz_type="bar_chart", x_field="Order Date", y_field="Sales", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "évolution des ventes par mois")
    assert result.viz_type == "line_chart"


def test_no_change_when_already_line():
    """Already line_chart should not be changed."""
    ds = _ds("DB", [
        ("Date", FieldType.DATE, "dimension"),
        ("Revenue", FieldType.FLOAT, "measure"),
    ])
    intent = _intent(viz_type="line_chart", x_field="Date", y_field="Revenue", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "trend over time")
    assert result.viz_type == "line_chart"


def test_bar_stays_bar_without_time_words():
    """bar_chart with date field but NO time words should stay bar."""
    ds = _ds("DB", [
        ("Date", FieldType.DATE, "dimension"),
        ("Sales", FieldType.FLOAT, "measure"),
    ])
    intent = _intent(viz_type="bar_chart", x_field="Date", y_field="Sales", datasource_luid="luid-1")
    result = _validate_and_correct_intent(intent, [ds], "sales by date")
    assert result.viz_type == "bar_chart"


# --- Datasource LUID validation ---

def test_invalid_luid_corrected():
    """Invalid datasource_luid gets corrected to best matching DS."""
    ds = _ds("Sales DB", [
        ("Sales", FieldType.FLOAT, "measure"),
        ("Region", FieldType.STRING, "dimension"),
    ], luid="real-luid")
    intent = _intent(datasource_luid="fake-luid", x_field="Region", y_field="Sales")
    result = _validate_and_correct_intent(intent, [ds], "sales by region")
    assert result.datasource_luid == "real-luid"


def test_valid_luid_unchanged():
    """Valid datasource_luid should not be changed."""
    ds = _ds("DB", [("Sales", FieldType.FLOAT, "measure")], luid="valid-luid")
    intent = _intent(datasource_luid="valid-luid")
    result = _validate_and_correct_intent(intent, [ds], "show sales")
    assert result.datasource_luid == "valid-luid"


# --- KPI fix ---

def test_kpi_moves_y_to_x():
    """KPI type: y_field should be moved to x_field, y_field set to empty."""
    ds = _ds("DB", [("Revenue", FieldType.FLOAT, "measure")])
    intent = _intent(viz_type="kpi", x_field="", y_field="Revenue")
    result = _validate_and_correct_intent(intent, [ds], "total revenue")
    assert result.x_field == "Revenue"
    assert result.y_field == ""


def test_kpi_already_correct():
    """KPI with correct field assignment should not change."""
    ds = _ds("DB", [("Revenue", FieldType.FLOAT, "measure")])
    intent = _intent(viz_type="kpi", x_field="Revenue", y_field="")
    result = _validate_and_correct_intent(intent, [ds], "total revenue")
    assert result.x_field == "Revenue"
    assert result.y_field == ""


# --- No datasources (graceful) ---

def test_no_datasources_no_crash():
    """With empty datasource list, function should not crash."""
    intent = _intent()
    result = _validate_and_correct_intent(intent, [], "show sales")
    assert result.viz_type == "bar_chart"  # unchanged
