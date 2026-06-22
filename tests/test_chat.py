import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config import settings
from conftest import make_tool_response, make_text_response

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
async def test_bar_chart_generation(client):
    # Patch call_llm in main's namespace (where it was imported)
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)

        response = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "bar_chart"
    assert data["viz_intent"]["title"] == "Sales by Category"
    assert data["twb_filename"].endswith(".twb")
    assert data["twb_download_url"].startswith("/download/")
    assert "session_id" in data
    assert "trace_id" in data


@pytest.mark.asyncio
async def test_line_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_LINE_INTENT)

        response = await client.post("/chat", json={
            "question": "Show revenue trend over time",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "line_chart"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_chat_text_response_conversation_mode(client):
    """When LLM returns text (no tool call), response is conversation mode."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_text_response("Oui, je peux faire du blending !")

        response = await client.post("/chat", json={
            "question": "tu peut faire le blending ?",
            "session_id": str(uuid.uuid4()),
            "metadata": None,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "conversation"
    assert "blending" in data["message"]
    assert data["twb_filename"] == ""
    assert data["viz_intent"] is None


@pytest.mark.asyncio
async def test_download_generated_twb(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)

        chat_resp = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert chat_resp.status_code == 200
    download_url = chat_resp.json()["twb_download_url"]

    dl_resp = await client.get(download_url)
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] == "application/octet-stream"
    content = dl_resp.content.decode("utf-8")
    assert "<?xml" in content
    assert "<workbook" in content


@pytest.mark.asyncio
async def test_download_path_traversal_blocked(client):
    response = await client.get("/download/../config.py")
    assert response.status_code in (400, 404)


@pytest.mark.asyncio
async def test_session_reset(client):
    session_id = str(uuid.uuid4())
    response = await client.post("/session/reset", json={"session_id": session_id})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_add_sheet_to_existing(client):
    """When workbook_name matches an existing .twb, mode == sheet_added."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)

        # First, generate a workbook to pre-populate output/
        first_resp = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert first_resp.status_code == 200
    original_filename = first_resp.json()["twb_filename"]  # e.g. "abc12345_bar_chart.twb"

    # Rename the file so workbook_name matches predictably
    workbook_name = "test_existing_wb"
    original_path = settings.output_dir / original_filename
    target_path = settings.output_dir / f"{workbook_name}.twb"
    original_path.rename(target_path)

    try:
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_tool_response(MOCK_LINE_INTENT)

            resp = await client.post("/chat", json={
                "question": "Show revenue trend over time",
                "session_id": str(uuid.uuid4()),
                "metadata": SAMPLE_METADATA,
                "conversation_history": [],
                "workbook_name": workbook_name,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "sheet_added"
        assert data["twb_filename"] == f"{workbook_name}.twb"

        # Verify the workbook now contains both worksheets
        content = target_path.read_text(encoding="utf-8")
        assert MOCK_BAR_INTENT["title"] in content
        assert MOCK_LINE_INTENT["title"] in content
    finally:
        target_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_fallback_when_workbook_not_found(client):
    """When workbook_name doesn't match any file, falls back to new_workbook."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)

        resp = await client.post("/chat", json={
            "question": "Show me sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
            "workbook_name": "nonexistent_workbook_xyz",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "new_workbook"
    assert data["twb_filename"].endswith(".twb")
