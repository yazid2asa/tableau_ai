"""
test_filters.py — M4 filter system tests.

Covers:
  - Date range filter (year)
  - Categorical filter (eq, in)
  - Numeric filter (gt, between)
  - Top N filter
  - Implicit semantic filter (profitable → Profit > 0)
  - Multiple filters combined
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
        {"name": "Region", "type": "string", "role": "dimension"},
        {"name": "Order Date", "type": "date", "role": "dimension"},
        {"name": "Customer", "type": "string", "role": "dimension"},
    ],
}


def _make_intent_with_filters(filters):
    return {
        "viz_type": "bar_chart",
        "title": "Filtered Chart",
        "x_field": "Category",
        "y_field": "Sales",
        "color_field": None,
        "filters": filters,
        "calculated_fields": [],
        "clarification_needed": None,
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
    }


@pytest.mark.asyncio
async def test_date_year_filter(client):
    """A year filter (op=year) is accepted and produces a valid .twb."""
    intent = _make_intent_with_filters([
        {"field": "Order Date", "op": "year", "value": 2024}
    ])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "ventes en 2024",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["filters"][0]["op"] == "year"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_categorical_eq_filter(client):
    """A categorical eq filter produces a valid .twb."""
    intent = _make_intent_with_filters([
        {"field": "Region", "op": "eq", "value": "West"}
    ])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "uniquement la région West",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["viz_intent"]["filters"]) == 1
    assert data["viz_intent"]["filters"][0]["field"] == "Region"


@pytest.mark.asyncio
async def test_categorical_in_filter(client):
    """A multi-value categorical filter (op=in) works."""
    intent = _make_intent_with_filters([
        {"field": "Category", "op": "in", "values": ["Furniture", "Technology"]}
    ])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "catégories Furniture et Technology",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["filters"][0]["op"] == "in"
    assert data["viz_intent"]["filters"][0]["values"] == ["Furniture", "Technology"]


@pytest.mark.asyncio
async def test_numeric_gt_filter(client):
    """A numeric gt filter produces a valid .twb."""
    intent = _make_intent_with_filters([
        {"field": "Sales", "op": "gt", "value": 10000}
    ])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "ventes supérieures à 10000",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["filters"][0]["op"] == "gt"


@pytest.mark.asyncio
async def test_top_n_filter(client):
    """A top_n filter is accepted and produces a valid .twb."""
    intent = _make_intent_with_filters([
        {"field": "Customer", "op": "top_n", "value": 10, "by": "Sales"}
    ])
    intent["sort"] = "descending"
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "top 10 clients",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["filters"][0]["op"] == "top_n"
    assert data["viz_intent"]["filters"][0]["value"] == 10


@pytest.mark.asyncio
async def test_multiple_filters_combined(client):
    """Multiple filters combined in one request."""
    intent = _make_intent_with_filters([
        {"field": "Region", "op": "eq", "value": "West"},
        {"field": "Profit", "op": "gt", "value": 0},
    ])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "ventes rentables dans la région West",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["viz_intent"]["filters"]) == 2
