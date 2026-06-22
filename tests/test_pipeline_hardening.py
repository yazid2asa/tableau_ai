"""Pipeline-hardening tests:

  - P2: per-session asyncio.Lock serializes concurrent same-session requests
        while letting different sessions run concurrently.
  - P4: real self-correction — _revise_intent_with_feedback feeds judge feedback
        back to the LLM; the pipeline adopts the revision only if it scores at
        least as well (regression guard).
"""
import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_tool_response
from llm import LLMResponse
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent
from main import _get_session_lock, _revise_intent_with_feedback


SAMPLE_METADATA = {
    "datasource_name": "superstore",
    "datasource_caption": "Sample - Superstore",
    "fields": [
        {"name": "Category", "type": "string", "role": "dimension"},
        {"name": "Sales", "type": "float", "role": "measure"},
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


def _chat_body(session_id):
    return {
        "question": "Show me sales by category",
        "session_id": session_id,
        "metadata": SAMPLE_METADATA,
        "conversation_history": [],
    }


# ---------------------------------------------------------------------------
# P2 — per-session lock
# ---------------------------------------------------------------------------

def test_session_lock_is_per_session_and_stable():
    """Same id → same lock object; different id → different lock."""
    a1 = _get_session_lock("sess-A")
    a2 = _get_session_lock("sess-A")
    b = _get_session_lock("sess-B")
    assert a1 is a2
    assert a1 is not b


@pytest.mark.asyncio
async def test_same_session_requests_serialize(client, mock_judge_viz):
    """Two concurrent /chat calls for the SAME session never overlap inside the
    pipeline — the per-session lock serializes them (max concurrency == 1)."""
    mock_judge_viz.return_value = (0.92, "ok")
    session_id = str(uuid.uuid4())
    state = {"current": 0, "max": 0}

    async def fake_llm(messages, tools=None, **kwargs):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.05)
        state["current"] -= 1
        return make_tool_response(MOCK_BAR_INTENT)

    with patch("main.call_llm", new=fake_llm):
        results = await asyncio.gather(
            client.post("/chat", json=_chat_body(session_id)),
            client.post("/chat", json=_chat_body(session_id)),
        )

    assert all(r.status_code == 200 for r in results)
    assert state["max"] == 1


@pytest.mark.asyncio
async def test_different_sessions_run_concurrently(client, mock_judge_viz):
    """Two concurrent /chat calls for DIFFERENT sessions overlap — proving the
    lock is per-session, not global. A barrier(2) only releases if both pipelines
    reach the LLM call at the same time; a global lock would deadlock and time out."""
    mock_judge_viz.return_value = (0.92, "ok")
    barrier = asyncio.Barrier(2)

    async def fake_llm(messages, tools=None, **kwargs):
        await asyncio.wait_for(barrier.wait(), timeout=5.0)
        return make_tool_response(MOCK_BAR_INTENT)

    with patch("main.call_llm", new=fake_llm):
        results = await asyncio.gather(
            client.post("/chat", json=_chat_body(str(uuid.uuid4()))),
            client.post("/chat", json=_chat_body(str(uuid.uuid4()))),
        )

    assert all(r.status_code == 200 for r in results)


# ---------------------------------------------------------------------------
# P4 — self-correction (_revise_intent_with_feedback)
# ---------------------------------------------------------------------------

def _meta():
    return DataSourceMetadata(
        datasource_name="DB", luid="luid-1",
        fields=[
            FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension"),
            FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        ],
    )


@pytest.mark.asyncio
async def test_revise_returns_corrected_intent():
    original = VizIntent(viz_type="bar_chart", title="t", x_field="Order Date",
                         y_field="Sales", datasource_luid="luid-1")
    corrected = {**MOCK_BAR_INTENT, "viz_type": "line_chart",
                 "x_field": "Order Date", "y_field": "Sales", "datasource_luid": "luid-1"}
    with patch("main.call_llm", new_callable=AsyncMock) as m:
        m.return_value = make_tool_response(corrected)
        result = await _revise_intent_with_feedback(
            original, "use a line chart for a date axis", "sales trend", _meta(),
        )
    assert result is not None
    assert result.viz_type == "line_chart"
    assert result.datasource_luid == "luid-1"


@pytest.mark.asyncio
async def test_revise_preserves_datasource_when_dropped():
    """If the revision omits datasource_luid, the original choice is preserved."""
    original = VizIntent(viz_type="bar_chart", title="t", x_field="Order Date",
                         y_field="Sales", datasource_luid="luid-1")
    revised_args = {**MOCK_BAR_INTENT, "x_field": "Order Date", "y_field": "Sales"}
    revised_args.pop("datasource_luid", None)  # ensure key absent
    with patch("main.call_llm", new_callable=AsyncMock) as m:
        m.return_value = make_tool_response(revised_args)
        result = await _revise_intent_with_feedback(original, "fix it", "q", _meta())
    assert result is not None
    assert result.datasource_luid == "luid-1"


@pytest.mark.asyncio
async def test_revise_returns_none_when_no_tool_call():
    original = VizIntent(viz_type="bar_chart", title="t", x_field="Order Date",
                         y_field="Sales", datasource_luid="luid-1")
    with patch("main.call_llm", new_callable=AsyncMock) as m:
        m.return_value = LLMResponse(content="I'm not sure how to fix this.")
        result = await _revise_intent_with_feedback(original, "fix it", "q", _meta())
    assert result is None


@pytest.mark.asyncio
async def test_revise_returns_none_on_llm_error():
    original = VizIntent(viz_type="bar_chart", title="t", x_field="Order Date",
                         y_field="Sales", datasource_luid="luid-1")
    with patch("main.call_llm", new_callable=AsyncMock) as m:
        m.side_effect = ValueError("provider exploded")
        result = await _revise_intent_with_feedback(original, "fix it", "q", _meta())
    assert result is None


@pytest.mark.asyncio
async def test_self_correction_keeps_original_when_revision_worse(client, mock_judge_viz):
    """Regression guard: a worse revision is discarded — the original score stands."""
    # Initial judge 0.50 (< 0.75 threshold → triggers self-correction); the
    # re-judge of the revision scores 0.40 (worse) → original is kept.
    mock_judge_viz.side_effect = [
        (0.50, "Initial is weak."),
        (0.40, "Revision is worse."),
    ]
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.quick_validate", return_value=(0.5, "force judge")):
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        resp = await client.post("/chat", json=_chat_body(str(uuid.uuid4())))

    assert resp.status_code == 200
    data = resp.json()
    assert data["judge_score"] == pytest.approx(0.50, abs=1e-3)
    assert data["warning"] is not None  # still below threshold
