"""Tests for M8 — Datasource Guardrails (sample exclusion + field-match ranking)."""
import pytest

from schemas import DataSourceMetadata, FieldInfo, FieldType
from main import _rank_datasources_by_relevance


def _ds(name: str, fields: list[str], luid: str = "luid-1") -> DataSourceMetadata:
    """Helper to create a DataSourceMetadata with string field names."""
    return DataSourceMetadata(
        datasource_name=name,
        luid=luid,
        fields=[FieldInfo(name=f, type=FieldType.STRING) for f in fields],
    )


# --- Sample datasource exclusion ---

def test_excludes_superstore():
    """Superstore (Tableau's sample) is excluded when other DS exist."""
    superstore = _ds("Sample - Superstore", ["Sales", "Profit", "Category"], "ss-luid")
    real_ds = _ds("Company Revenue", ["Revenue", "Region", "Quarter"], "real-luid")
    result = _rank_datasources_by_relevance("Show sales by region", [superstore, real_ds])
    assert len(result) == 1
    assert result[0].datasource_name == "Company Revenue"


def test_excludes_world_indicators():
    """World Indicators sample DS is excluded."""
    world = _ds("World Indicators", ["Country", "GDP", "Population"], "wi-luid")
    real_ds = _ds("Sales Data", ["Country", "Revenue"], "real-luid")
    result = _rank_datasources_by_relevance("Revenue by country", [world, real_ds])
    assert all(ds.datasource_name != "World Indicators" for ds in result)


def test_excludes_coffee_chain():
    """Coffee Chain sample DS is excluded."""
    coffee = _ds("Coffee Chain", ["Product", "Sales"], "cc-luid")
    real_ds = _ds("Product Analytics", ["Product", "Revenue"], "real-luid")
    result = _rank_datasources_by_relevance("Product sales", [coffee, real_ds])
    assert all(ds.datasource_name != "Coffee Chain" for ds in result)


def test_fallback_when_all_samples():
    """If ALL datasources are samples, return the original list (graceful fallback)."""
    ss = _ds("Sample - Superstore", ["Sales", "Profit"], "ss-luid")
    wi = _ds("World Indicators", ["GDP"], "wi-luid")
    result = _rank_datasources_by_relevance("Show sales", [ss, wi])
    assert len(result) == 2  # both returned as fallback


# --- Field-match ranking ---

def test_ranks_by_field_relevance():
    """DS with more matching fields ranks higher."""
    ds_low = _ds("General", ["ID", "Status", "Notes"], "low-luid")
    ds_high = _ds("Sales DB", ["Sales", "Region", "Category", "Profit"], "high-luid")
    result = _rank_datasources_by_relevance("Show sales by region and category", [ds_low, ds_high])
    assert result[0].datasource_name == "Sales DB"


def test_all_non_sample_datasources_returned():
    """EVERY non-sample datasource is returned (FIX-011 — the old top-3 cap hid
    datasources ranked 4+ from the LLM, so valid questions got 'I don't see
    field X' even though X existed in a lower-ranked datasource)."""
    datasources = [
        _ds(f"DS_{i}", [f"Field_{i}"], f"luid-{i}")
        for i in range(6)
    ]
    result = _rank_datasources_by_relevance("anything", datasources)
    assert len(result) == 6


def test_empty_list_returns_empty():
    """Empty input returns empty output."""
    result = _rank_datasources_by_relevance("Show sales", [])
    assert result == []


def test_single_datasource_passes_through():
    """A single non-sample DS passes through regardless of field match."""
    ds = _ds("My Data", ["Unrelated"], "my-luid")
    result = _rank_datasources_by_relevance("Show revenue", [ds])
    assert len(result) == 1
    assert result[0].datasource_name == "My Data"


def test_case_insensitive_pattern_match():
    """Sample pattern matching is case-insensitive."""
    ds = _ds("SUPERSTORE Sales", ["Sales"], "ss-luid")
    real = _ds("Actual Data", ["Sales"], "real-luid")
    result = _rank_datasources_by_relevance("sales", [ds, real])
    assert len(result) == 1
    assert result[0].datasource_name == "Actual Data"


def test_case_insensitive_field_match():
    """Field matching against question is case-insensitive."""
    ds = _ds("Revenue DB", ["Total_Revenue", "Region"], "rev-luid")
    result = _rank_datasources_by_relevance("show total_revenue by region", [ds])
    assert len(result) == 1
