import json
import uuid
from unittest.mock import AsyncMock, patch

from conftest import make_tool_response, make_text_response

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
    "sort": "descending",
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}


def _parse_sse_text(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = "message"
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
        if data_str:
            try:
                events.append({"event": event_type, "data": json.loads(data_str)})
            except json.JSONDecodeError:
                pass
    return events


@pytest.mark.asyncio
async def test_chat_stream_success(client):
    """POST /chat/stream emits status, intent, result, and done events."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)

        response = await client.post(
            "/chat/stream",
            json={
                "question": "Show me sales by category",
                "session_id": str(uuid.uuid4()),
                "metadata": SAMPLE_METADATA,
                "conversation_history": [],
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_text(response.text)
    event_types = [e["event"] for e in events]

    assert "status" in event_types, "Expected at least one status event"
    assert "result" in event_types, "Expected a result event"
    assert "done" in event_types, "Expected a done event"

    result_events = [e for e in events if e["event"] == "result"]
    assert len(result_events) == 1
    result_data = result_events[0]["data"]
    assert result_data["viz_intent"]["viz_type"] == "bar_chart"
    assert result_data["twb_filename"].endswith(".twb")
    assert result_data["twb_download_url"].startswith("/download/")
    assert "session_id" in result_data
    assert "trace_id" in result_data


@pytest.mark.asyncio
async def test_chat_stream_conversation_mode(client):
    """POST /chat/stream emits a conversation result when LLM returns text only."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_text_response("Oui, je peux créer des graphiques !")

        response = await client.post(
            "/chat/stream",
            json={
                "question": "tu peut faire des graphiques ?",
                "session_id": str(uuid.uuid4()),
                "metadata": None,
                "conversation_history": [],
            },
        )

    assert response.status_code == 200

    events = _parse_sse_text(response.text)
    event_types = [e["event"] for e in events]

    assert "result" in event_types, "Expected a result event"
    assert "done" in event_types, "Expected a done event"

    result_events = [e for e in events if e["event"] == "result"]
    assert len(result_events) == 1
    result_data = result_events[0]["data"]
    assert result_data["mode"] == "conversation"
    assert "graphiques" in result_data["message"]
    assert result_data["twb_filename"] == ""
