"""
test_conversation.py — M4 conversational continuity tests.

Covers:
  - Modify filter on existing chart
  - Change chart type (action=modify)
  - New chart detection (action=new)
  - Clarification flow (action=clarify)
  - Session isolation (two parallel sessions)
  - Session reset clears session state
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
    ],
}

MOCK_BAR_INTENT = {
    "viz_type": "bar_chart",
    "title": "Sales by Category",
    "x_field": "Category",
    "y_field": "Sales",
    "color_field": None,
    "filters": [],
    "calculated_fields": [],
    "clarification_needed": None,
    "sort": "descending",
    "aggregation": "SUM",
    "color_scheme": "tableau10",
    "action": "new",
}


@pytest.mark.asyncio
async def test_modify_adds_filter(client):
    """Create a chart, then say 'add filter 2024' → same viz_type, filter appended."""
    session_id = str(uuid.uuid4())

    # First turn: create bar chart
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        resp1 = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp1.status_code == 200
    assert resp1.json()["viz_intent"]["viz_type"] == "bar_chart"

    # Second turn: modify — add filter
    modified_intent = {
        **MOCK_BAR_INTENT,
        "action": "modify",
        "filters": [{"field": "Order Date", "op": "year", "value": 2024}],
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(modified_intent)
        resp2 = await client.post("/chat", json={
            "question": "add a filter for 2024",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["viz_intent"]["viz_type"] == "bar_chart"  # same type
    assert len(data["viz_intent"]["filters"]) == 1
    assert data["viz_intent"]["filters"][0]["op"] == "year"


@pytest.mark.asyncio
async def test_change_chart_type(client):
    """Create bar chart, then 'change to line chart' → viz_type updated, fields preserved."""
    session_id = str(uuid.uuid4())

    # First turn
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    # Second turn: change type
    line_intent = {
        **MOCK_BAR_INTENT,
        "viz_type": "line_chart",
        "x_field": "Order Date",
        "action": "modify",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(line_intent)
        resp = await client.post("/chat", json={
            "question": "change to a line chart over time",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["viz_type"] == "line_chart"
    assert data["viz_intent"]["y_field"] == "Sales"  # preserved


@pytest.mark.asyncio
async def test_new_chart_detection(client):
    """A completely unrelated question after an existing chart → action=new."""
    session_id = str(uuid.uuid4())

    # First turn
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    # Second turn: completely new question
    new_intent = {
        "viz_type": "pie",
        "title": "Profit Distribution by Region",
        "x_field": "Region",
        "y_field": "Profit",
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
        mock_llm.return_value = make_tool_response(new_intent)
        resp = await client.post("/chat", json={
            "question": "Show profit distribution by region as a pie chart",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["viz_intent"]["viz_type"] == "pie"
    assert data["viz_intent"]["action"] == "new"


@pytest.mark.asyncio
async def test_clarification_flow(client):
    """Ambiguous follow-up returns clarification_needed, no .twb generated."""
    session_id = str(uuid.uuid4())

    # First turn
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    # Second turn: ambiguous
    clarify_intent = {
        **MOCK_BAR_INTENT,
        "action": "clarify",
        "clarification_needed": "Voulez-vous filtrer le graphique existant sur West, ou créer un nouveau graphique pour West uniquement?",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(clarify_intent)
        resp = await client.post("/chat", json={
            "question": "show West",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification"
    assert data["clarification_needed"] is not None
    assert data["twb_filename"] == ""


@pytest.mark.asyncio
async def test_session_isolation(client):
    """Two parallel sessions do not bleed state."""
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    # Session A: bar chart
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        resp_a = await client.post("/chat", json={
            "question": "Sales by category",
            "session_id": session_a,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp_a.status_code == 200

    # Session B: pie chart (new session, action=new, no previous intent)
    pie_intent = {
        "viz_type": "pie",
        "title": "Profit by Region",
        "x_field": "Region",
        "y_field": "Profit",
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
        mock_llm.return_value = make_tool_response(pie_intent)
        resp_b = await client.post("/chat", json={
            "question": "Profit by region pie chart",
            "session_id": session_b,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp_b.status_code == 200

    # Verify carts are separate
    cart_a = (await client.get(f"/session/{session_a}/charts")).json()
    cart_b = (await client.get(f"/session/{session_b}/charts")).json()
    assert len(cart_a["charts"]) == 1
    assert cart_a["charts"][0]["viz_type"] == "bar_chart"
    assert len(cart_b["charts"]) == 1
    assert cart_b["charts"][0]["viz_type"] == "pie"


@pytest.mark.asyncio
async def test_session_reset_clears_state(client):
    """Resetting a session clears the session state (turns)."""
    session_id = str(uuid.uuid4())

    # Create a chart
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    # Reset
    await client.post("/session/reset", json={"session_id": session_id})

    # After reset, a modify action should NOT find previous intent
    # (the session state is cleared, so it's treated as a new chart)
    new_intent = {
        **MOCK_BAR_INTENT,
        "action": "new",  # LLM should return "new" since no previous context
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(new_intent)
        resp = await client.post("/chat", json={
            "question": "add a filter for 2024",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    # Cart should have only 1 chart (the new one, not accumulated from before reset)
    cart = (await client.get(f"/session/{session_id}/charts")).json()
    assert len(cart["charts"]) == 1
