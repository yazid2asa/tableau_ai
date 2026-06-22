"""Tests for the unified, SQLite-persisted conversational memory.

Guarantees:
  - Chart turns record a READABLE memory line (not raw JSON).
  - General questions record an 'answer' turn (the agent remembers what it explained).
  - Memory + cart survive the in-memory cache being wiped (simulating uvicorn --reload).
  - Reset clears both the in-memory cache and the persisted row.
"""
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_tool_response, make_text_response
import main

SAMPLE_METADATA = {
    "datasource_name": "superstore",
    "datasource_caption": "Sample - Superstore",
    "fields": [
        {"name": "Category", "type": "string", "role": "dimension"},
        {"name": "Sales", "type": "float", "role": "measure"},
    ],
}

BAR = {
    "viz_type": "bar_chart", "title": "Sales by Category",
    "x_field": "Category", "y_field": "Sales", "color_field": None,
    "filters": [], "calculated_fields": [], "clarification_needed": None,
    "sort": None, "aggregation": "SUM", "color_scheme": "tableau10", "action": "new",
}


async def _chat(client, sid, question="sales by category", resp=None):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = resp if resp is not None else make_tool_response(BAR)
        return await client.post("/chat", json={
            "question": question, "session_id": sid,
            "metadata": SAMPLE_METADATA, "conversation_history": [],
        })


@pytest.mark.asyncio
async def test_chart_turn_records_readable_memory(client):
    sid = str(uuid.uuid4())
    await _chat(client, sid)
    turn = main._session_states[sid].turns[-1]
    assert turn.kind == "create"
    assert turn.assistant_text and "{" not in turn.assistant_text  # readable, not JSON
    assert "Sales by Category" in turn.assistant_text


@pytest.mark.asyncio
async def test_text_question_records_answer_turn(client):
    sid = str(uuid.uuid4())
    r = await _chat(client, sid, "what is a KPI?",
                    resp=make_text_response("A KPI is a key performance indicator."))
    assert r.json()["mode"] == "conversation"
    turn = main._session_states[sid].turns[-1]
    assert turn.kind == "answer"
    assert "KPI" in turn.assistant_text
    assert turn.resolved_intent is None


@pytest.mark.asyncio
async def test_memory_survives_cache_wipe(client):
    """Clearing the in-memory cache (≈ uvicorn --reload) must not lose the session."""
    sid = str(uuid.uuid4())
    await _chat(client, sid)

    main._session_states.clear()  # simulate a backend reload mid-conversation

    cart = (await client.get(f"/session/{sid}/charts")).json()
    assert len(cart["charts"]) == 1  # cart rehydrated from SQLite

    state = await main._hydrate_session_state(sid)
    assert len(state.turns) == 1
    assert state.turns[0].assistant_text  # readable memory preserved across the wipe


@pytest.mark.asyncio
async def test_reset_clears_persisted_memory(client):
    sid = str(uuid.uuid4())
    await _chat(client, sid)
    await client.post("/session/reset", json={"session_id": sid})

    main._session_states.clear()  # ensure we read from SQLite, not the cache
    cart = (await client.get(f"/session/{sid}/charts")).json()
    assert cart["charts"] == []  # row was deleted from SQLite too
