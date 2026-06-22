"""
test_calculated_fields.py — M4 calculated field generation tests.

Covers:
  - Profit margin calculated field
  - Calculated field used as y_field
  - Clarification needed (ambiguous formula)
  - Missing datasource fields
  - Multiple calculated fields
"""
import json
import uuid
from unittest.mock import AsyncMock, patch

from conftest import make_tool_response

import pytest

SAMPLE_METADATA = {
    "datasource_name": "superstore",
    "datasource_caption": "Sample - Superstore",
    "fields": [
        {"name": "Category", "type": "string", "role": "dimension"},
        {"name": "Sales", "type": "float", "role": "measure"},
        {"name": "Profit", "type": "float", "role": "measure"},
        {"name": "Order_ID", "type": "string", "role": "dimension"},
        {"name": "Customer", "type": "string", "role": "dimension"},
    ],
}


@pytest.mark.asyncio
async def test_profit_margin_calculated_field(client):
    """A calculated field for profit margin is created and used."""
    intent = {
        "viz_type": "bar_chart",
        "title": "Taux de Marge par Catégorie",
        "x_field": "Category",
        "y_field": "Taux de Marge",
        "color_field": None,
        "filters": [],
        "calculated_fields": [
            {
                "name": "Taux de Marge",
                "formula": "SUM([Profit])/SUM([Sales])",
                "datatype": "real",
                "role": "measure",
            }
        ],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "montre moi le taux de marge par catégorie",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["viz_intent"]["calculated_fields"]) == 1
    assert data["viz_intent"]["calculated_fields"][0]["name"] == "Taux de Marge"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_calculated_field_used_as_y_field(client):
    """Calculated field name can be used as y_field and passes field validation."""
    intent = {
        "viz_type": "bar_chart",
        "title": "Panier Moyen par Catégorie",
        "x_field": "Category",
        "y_field": "Panier Moyen",
        "color_field": None,
        "filters": [],
        "calculated_fields": [
            {
                "name": "Panier Moyen",
                "formula": "SUM([Sales])/COUNTD([Order_ID])",
                "datatype": "real",
                "role": "measure",
            }
        ],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "panier moyen par catégorie",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    # Should NOT have field validation error because calc field name is accepted
    assert data["mode"] != "clarification"
    assert data["viz_intent"]["y_field"] == "Panier Moyen"


@pytest.mark.asyncio
async def test_clarification_needed_ambiguous(client):
    """When clarification_needed is set, no .twb is generated."""
    intent = {
        "viz_type": "bar_chart",
        "title": "",
        "x_field": "Category",
        "y_field": "Sales",
        "color_field": None,
        "filters": [],
        "calculated_fields": [],
        "clarification_needed": "Que voulez-vous dire par performance: profit total, taux de marge, ou classement?",
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "clarify",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "montre moi la performance par catégorie",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification"
    assert data["clarification_needed"] is not None
    assert "performance" in data["clarification_needed"]
    assert data["twb_filename"] == ""


@pytest.mark.asyncio
async def test_missing_field_validation(client):
    """When a field doesn't exist in datasource, field validation returns error."""
    intent = {
        "viz_type": "bar_chart",
        "title": "Revenue by Category",
        "x_field": "Category",
        "y_field": "Revenue",  # NOT in SAMPLE_METADATA
        "color_field": None,
        "filters": [],
        "calculated_fields": [],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "revenue by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification"
    assert "Revenue" in data["message"]


@pytest.mark.asyncio
async def test_multiple_calculated_fields(client):
    """Multiple calculated fields can be created in a single request."""
    intent = {
        "viz_type": "text",
        "title": "Métriques par Catégorie",
        "x_field": "Category",
        "y_field": "Taux de Marge",
        "color_field": None,
        "filters": [],
        "calculated_fields": [
            {
                "name": "Taux de Marge",
                "formula": "SUM([Profit])/SUM([Sales])",
                "datatype": "real",
                "role": "measure",
            },
            {
                "name": "Profit par Client",
                "formula": "SUM([Profit])/COUNTD([Customer])",
                "datatype": "real",
                "role": "measure",
            },
        ],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "taux de marge et profit par client par catégorie",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["viz_intent"]["calculated_fields"]) == 2


def test_calc_datatype_normalization():
    """The LLM sometimes emits 'STRING' / 'float' / 'int' for calc-field datatype.
    Tableau's loader rejects anything outside lowercase {string|integer|real|boolean|date|datetime}
    with `<column> tag contained an invalid data type`. Coerce them before writing XML."""
    from twb_generator import _normalize_calc_datatype as norm
    assert norm("STRING") == "string"
    assert norm("String") == "string"
    assert norm("float") == "real"
    assert norm("FLOAT") == "real"
    assert norm("int") == "integer"
    assert norm("INTEGER") == "integer"
    assert norm("bool") == "boolean"
    assert norm("timestamp") == "datetime"
    assert norm("real") == "real"
    assert norm("") == "real"
    assert norm(None) == "real"
    assert norm("gibberish") == "real"  # safe fallback


@pytest.mark.asyncio
async def test_cross_datasource_calc_pipeline_clarification(client, mock_tableau_server):
    """FIX-044 integration: pipeline returns mode='clarification' (not a TWB) when the
    LLM generates a calc field referencing a secondary-datasource-only field.

    Reproduces: question "taux de remplissage moyen" → LLM emits
    formula=[Cargo Weight Kg]/[capacity_kg] where capacity_kg is in vehicles (secondary).
    Expected: ChatResponse.mode='clarification' with an actionable message.
    """
    from schemas import DataSourceMetadata, FieldInfo, FieldType

    trips_ds = DataSourceMetadata(
        datasource_name="trips",
        luid="trips-luid",
        fields=[
            FieldInfo(name="Cargo Weight Kg", type=FieldType.FLOAT, role="measure"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        ],
    )
    vehicles_ds = DataSourceMetadata(
        datasource_name="vehicles",
        luid="vehicles-luid",
        fields=[
            FieldInfo(name="Capacity Kg", type=FieldType.FLOAT, role="measure",
                      local_name="capacity_kg"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        ],
    )
    mock_tableau_server["get_all_datasource_schemas"].return_value = [trips_ds, vehicles_ds]

    intent = {
        "viz_type": "kpi",
        "title": "Taux de remplissage moyen",
        "x_field": "Fill Rate",
        "y_field": "",
        "color_field": None,
        "filters": [],
        "calculated_fields": [
            {
                "name": "Fill Rate",
                "formula": "[Cargo Weight Kg] / [capacity_kg]",
                "datatype": "real",
                "role": "measure",
            }
        ],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "AVG",
        "color_scheme": "tableau10",
        "action": "new",
        "datasource_luid": "trips-luid",
        "secondary_datasource_luid": "vehicles-luid",
    }

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "taux de remplissage moyen",
            "session_id": str(uuid.uuid4()),
            "metadata": None,
            "conversation_history": [],
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification", f"Expected clarification, got: {data}"
    assert data["clarification_needed"] is not None
    assert "Fill Rate" in data["clarification_needed"]
    assert data["twb_filename"] == ""


def test_cross_datasource_calc_field_stripped():
    """FIX-044: calc fields referencing secondary-only fields are dropped.

    Reproduces the "taux de remplissage moyen" failure: the LLM generated
    [Cargo Weight Kg] / [capacity_kg] where capacity_kg belongs exclusively to
    the vehicles secondary datasource. Tableau rejects such a formula with
    "The calculation contains errors" because a primary-datasource calc field
    can only reference fields from that same datasource.
    """
    from twb_generator import _filter_cross_datasource_calc_fields
    from schemas import VizIntent, CalculatedField, DataSourceMetadata, FieldInfo, FieldType

    primary_meta = DataSourceMetadata(
        datasource_name="trips",
        fields=[
            FieldInfo(name="Cargo Weight Kg", type=FieldType.FLOAT, role="measure"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        ],
    )
    secondary_meta = DataSourceMetadata(
        datasource_name="vehicles",
        fields=[
            FieldInfo(name="Capacity Kg", type=FieldType.FLOAT, role="measure", local_name="capacity_kg"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),  # linking field
        ],
    )
    cross_ds_cf = CalculatedField(
        name="Fill Rate",
        formula="[Cargo Weight Kg] / [capacity_kg]",  # capacity_kg is secondary-only
        datatype="real",
        role="measure",
    )
    valid_cf = CalculatedField(
        name="Weight per Vehicle",
        formula="SUM([Cargo Weight Kg]) / COUNTD([Vehicle Id])",  # both fields in primary
        datatype="real",
        role="measure",
    )
    viz = VizIntent(
        viz_type="kpi",
        title="Fill Rate",
        x_field="Fill Rate",
        y_field="",
        calculated_fields=[cross_ds_cf, valid_cf],
        action="new",
    )

    result = _filter_cross_datasource_calc_fields(viz, primary_meta, secondary_meta)

    assert len(result.calculated_fields) == 1, "cross-DS calc field must be stripped"
    assert result.calculated_fields[0].name == "Weight per Vehicle"


def test_cross_datasource_calc_field_no_secondary():
    """Without a secondary datasource, _filter_cross_datasource_calc_fields is a no-op."""
    from twb_generator import _filter_cross_datasource_calc_fields
    from schemas import VizIntent, CalculatedField, DataSourceMetadata, FieldInfo, FieldType

    primary_meta = DataSourceMetadata(
        datasource_name="trips",
        fields=[FieldInfo(name="Cargo Weight Kg", type=FieldType.FLOAT, role="measure")],
    )
    calc_field = CalculatedField(
        name="Some Calc",
        formula="[Cargo Weight Kg] / [Unknown Field]",
        datatype="real",
        role="measure",
    )
    viz = VizIntent(
        viz_type="kpi",
        title="Test",
        x_field="Some Calc",
        y_field="",
        calculated_fields=[calc_field],
        action="new",
    )
    result = _filter_cross_datasource_calc_fields(viz, primary_meta, None)
    assert len(result.calculated_fields) == 1, "no secondary → no filtering"


def test_cross_datasource_calc_field_linking_field_not_stripped():
    """A calc field referencing only the linking field (shared by both DS) is kept.

    The linking field (e.g., Vehicle Id) exists in BOTH primary and secondary,
    so it is NOT secondary-only and must not be dropped.
    """
    from twb_generator import _filter_cross_datasource_calc_fields
    from schemas import VizIntent, CalculatedField, DataSourceMetadata, FieldInfo, FieldType

    primary_meta = DataSourceMetadata(
        datasource_name="trips",
        fields=[
            FieldInfo(name="Cargo Weight Kg", type=FieldType.FLOAT, role="measure"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        ],
    )
    secondary_meta = DataSourceMetadata(
        datasource_name="vehicles",
        fields=[
            FieldInfo(name="Capacity Kg", type=FieldType.FLOAT, role="measure", local_name="capacity_kg"),
            FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        ],
    )
    # References only Vehicle Id — exists in both → NOT secondary-only → keep
    cf = CalculatedField(
        name="Trip Count",
        formula="COUNTD([Vehicle Id])",
        datatype="integer",
        role="measure",
    )
    viz = VizIntent(
        viz_type="kpi",
        title="Trip Count",
        x_field="Trip Count",
        y_field="",
        calculated_fields=[cf],
        action="new",
    )
    result = _filter_cross_datasource_calc_fields(viz, primary_meta, secondary_meta)
    assert len(result.calculated_fields) == 1, "linking field is shared — calc must survive"
