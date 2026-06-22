"""
test_quality.py — M3+M4 quality & observability tests.

Covers:
  - LLM-as-a-Judge integration in /chat and /chat/stream
  - judge_score / judge_feedback in ChatResponse
  - Qualité partielle warning when judge score < threshold
  - Monitoring dashboard endpoints (/monitoring, /monitoring/metrics)
  - M4: Field validation against datasource metadata
  - M4: Reweighted judge criteria
  - M4: Clarification flow returns no .twb
  - M4: validate_intent_fields with close match suggestion
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


# ---------------------------------------------------------------------------
# Judge integration in /chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_returns_judge_score(client, mock_judge_viz):
    """ChatResponse includes judge_score and judge_feedback when generation succeeds."""
    mock_judge_viz.return_value = (0.88, "Good visualization — fields match expected types.")

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.quick_validate", return_value=(0.5, "force judge")):
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["judge_score"] == pytest.approx(0.88, abs=1e-3)
    assert data["judge_feedback"] == "Good visualization — fields match expected types."
    assert data["warning"] is None  # score >= 0.75, no warning


@pytest.mark.asyncio
async def test_chat_partial_quality_warning(client, mock_judge_viz):
    """When judge score < 0.75 after retries, response includes warning."""
    mock_judge_viz.return_value = (0.60, "Viz type may not be ideal for this question.")

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.quick_validate", return_value=(0.5, "force judge")):
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["judge_score"] == pytest.approx(0.60, abs=1e-3)
    assert data["warning"] is not None
    assert "0.60" in data["warning"]


@pytest.mark.asyncio
async def test_chat_judge_not_called_for_sheet_added(client, mock_judge_viz):
    """Judge is not invoked when mode == sheet_added."""
    from config import settings

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        # Create initial workbook
        first_resp = await client.post("/chat", json={
            "question": "Show me sales",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert first_resp.status_code == 200
    original_filename = first_resp.json()["twb_filename"]

    # Rename file to match workbook_name
    workbook_name = "test_sheet_added_judge"
    original_path = settings.output_dir / original_filename
    target_path = settings.output_dir / f"{workbook_name}.twb"
    original_path.rename(target_path)

    initial_call_count = mock_judge_viz.call_count

    try:
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
            resp = await client.post("/chat", json={
                "question": "Add a line chart",
                "session_id": str(uuid.uuid4()),
                "metadata": SAMPLE_METADATA,
                "conversation_history": [],
                "workbook_name": workbook_name,
            })
        assert resp.status_code == 200
        assert resp.json()["mode"] == "sheet_added"
        # Judge should NOT have been called for sheet_added mode
        assert mock_judge_viz.call_count == initial_call_count
    finally:
        target_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Judge retry logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_retry_succeeds_on_second_attempt(client, mock_judge_viz):
    """If first judge attempt fails threshold, retry succeeds — no warning emitted."""
    # First call: below threshold. Second call: above threshold.
    mock_judge_viz.side_effect = [
        (0.60, "Initial attempt was weak."),
        (0.85, "Improved after retry."),
    ]

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.quick_validate", return_value=(0.5, "force judge")):
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["warning"] is None  # second attempt passed
    assert data["judge_score"] == pytest.approx(0.85, abs=1e-3)


# ---------------------------------------------------------------------------
# Monitoring dashboard endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_monitoring_metrics_endpoint(client):
    """GET /monitoring/metrics returns a JSON dict with expected keys."""
    response = await client.get("/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()

    expected_keys = {
        "total_generations",
        "success_rate",
        "avg_judge_score",
        "avg_latency_ms",
        "p95_latency_ms",
        "viz_type_distribution",
        "feedback_positive",
        "feedback_negative",
        "recent_generations",
        "recent_feedback",
        "judge_score_buckets",
        "hourly_generations",
    }
    assert expected_keys.issubset(data.keys())


@pytest.mark.asyncio
async def test_monitoring_dashboard_html(client):
    """GET /monitoring returns HTML with the dashboard."""
    response = await client.get("/monitoring")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    content = response.text
    assert "monitoring" in content.lower() or "generation" in content.lower()


@pytest.mark.asyncio
async def test_monitoring_metrics_after_generation(client, mock_judge_viz):
    """After a successful generation, total_generations increments."""
    mock_judge_viz.return_value = (0.90, "Excellent visualization.")

    # Check initial count
    before = (await client.get("/monitoring/metrics")).json()
    initial_count = before["total_generations"]

    # Generate a chart
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        chat_resp = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert chat_resp.status_code == 200

    # Check count incremented
    after = (await client.get("/monitoring/metrics")).json()
    assert after["total_generations"] == initial_count + 1


# ---------------------------------------------------------------------------
# M4: Field validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_field_validation_missing_field(client):
    """Field validation detects missing field and returns clarification."""
    intent = {
        **MOCK_BAR_INTENT,
        "y_field": "NonExistentField",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "Show nonexistent data",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification"
    assert "NonExistentField" in data["message"]


@pytest.mark.asyncio
async def test_field_auto_correction(client):
    """Close-match typo is auto-corrected silently (e.g. 'Revenu' → 'Revenue')."""
    meta_with_revenue = {
        **SAMPLE_METADATA,
        "fields": [
            {"name": "Category", "type": "string", "role": "dimension"},
            {"name": "Sales", "type": "float", "role": "measure"},
            {"name": "Revenue", "type": "float", "role": "measure"},
        ],
    }
    intent = {
        **MOCK_BAR_INTENT,
        "y_field": "Revenu",  # typo — close to "Revenue"
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "Show revenu by category",
            "session_id": str(uuid.uuid4()),
            "metadata": meta_with_revenue,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    # Auto-corrected: "Revenu" → "Revenue", chart generated successfully
    assert data["mode"] != "clarification"
    assert data["viz_intent"]["y_field"] == "Revenue"


@pytest.mark.asyncio
async def test_judge_reweighted_criteria():
    """Judge weights are updated to M4 values."""
    from judge import _WEIGHTS
    assert _WEIGHTS["viz_relevance"] == pytest.approx(0.45)
    assert _WEIGHTS["field_cohesion"] == pytest.approx(0.30)
    assert _WEIGHTS["completeness"] == pytest.approx(0.15)
    assert _WEIGHTS["xml_validity"] == pytest.approx(0.10)
    assert sum(_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_clarification_returns_no_twb(client):
    """When LLM returns clarification_needed, response has mode=clarification and no twb."""
    intent = {
        **MOCK_BAR_INTENT,
        "clarification_needed": "Quel champ souhaitez-vous analyser?",
        "action": "clarify",
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        resp = await client.post("/chat", json={
            "question": "show me the data",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarification"
    assert data["twb_filename"] == ""
    assert data["clarification_needed"] is not None
