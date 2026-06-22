import json
import uuid
from unittest.mock import AsyncMock, patch

from conftest import make_tool_response

import pytest

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

MOCK_LINE_INTENT = {
    "viz_type": "line_chart",
    "title": "Revenue Over Time",
    "x_field": "Order Date",
    "y_field": "Revenue",
    "color_field": None,
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

SAMPLE_METADATA = {
    "datasource_name": "superstore",
    "datasource_caption": "Sample - Superstore",
    "fields": [
        {"name": "Category", "type": "string", "role": "dimension"},
        {"name": "Sales", "type": "float", "role": "measure"},
        {"name": "Order Date", "type": "date", "role": "dimension"},
        {"name": "Revenue", "type": "float", "role": "measure"},
    ],
}


@pytest.mark.asyncio
async def test_get_session_charts_empty(client):
    """Empty session returns empty cart."""
    resp = await client.get(f"/session/{uuid.uuid4()}/charts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["charts"] == []


@pytest.mark.asyncio
async def test_get_session_charts_accumulates(client):
    """After a /chat call, the cart has one entry."""
    session_id = str(uuid.uuid4())
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    resp = await client.get(f"/session/{session_id}/charts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["charts"]) == 1
    assert data["charts"][0]["viz_type"] == "bar_chart"


@pytest.mark.asyncio
async def test_cart_accumulates_multiple_charts(client):
    """Two /chat calls in the same session produce a cart with two entries."""
    session_id = str(uuid.uuid4())
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
        mock_llm.return_value = make_tool_response(MOCK_LINE_INTENT)
        await client.post("/chat", json={
            "question": "Show revenue trend",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    resp = await client.get(f"/session/{session_id}/charts")
    assert resp.status_code == 200
    assert len(resp.json()["charts"]) == 2


@pytest.mark.asyncio
async def test_download_session_workbook(client):
    """POST /download/{session_id} returns a valid multi-sheet .twb."""
    session_id = str(uuid.uuid4())
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    resp = await client.post(f"/download/{session_id}", json={"metadata": SAMPLE_METADATA})
    assert resp.status_code == 200
    content = resp.content.decode("utf-8")
    assert "<workbook" in content
    assert MOCK_BAR_INTENT["title"] in content


@pytest.mark.asyncio
async def test_download_session_workbook_client_charts(client):
    """POST /download/{session_id} with charts in body uses those instead of server cart."""
    session_id = str(uuid.uuid4())
    resp = await client.post(f"/download/{session_id}", json={
        "metadata": SAMPLE_METADATA,
        "charts": [MOCK_BAR_INTENT, MOCK_LINE_INTENT],
    })
    assert resp.status_code == 200
    content = resp.content.decode("utf-8")
    assert MOCK_BAR_INTENT["title"] in content
    assert MOCK_LINE_INTENT["title"] in content


@pytest.mark.asyncio
async def test_download_empty_cart_returns_404(client):
    """Downloading with no charts returns 404."""
    resp = await client.post(f"/download/{uuid.uuid4()}", json={"metadata": SAMPLE_METADATA})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_reset_clears_cart(client):
    """Resetting the session also clears the cart."""
    session_id = str(uuid.uuid4())
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        await client.post("/chat", json={
            "question": "Show sales by category",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    await client.post("/session/reset", json={"session_id": session_id})

    resp = await client.get(f"/session/{session_id}/charts")
    assert resp.status_code == 200
    assert resp.json()["charts"] == []
