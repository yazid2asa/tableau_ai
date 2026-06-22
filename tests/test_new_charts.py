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
        {"name": "Sub-Category", "type": "string", "role": "dimension"},
        {"name": "Customer", "type": "string", "role": "dimension"},
        {"name": "Sales", "type": "float", "role": "measure"},
        {"name": "Profit", "type": "float", "role": "measure"},
        {"name": "Order Date", "type": "date", "role": "dimension"},
        {"name": "Quantity", "type": "integer", "role": "measure"},
    ],
}

GANTT_METADATA = {
    "datasource_name": "projects",
    "datasource_caption": "Project Tracker",
    "fields": [
        {"name": "Task", "type": "string", "role": "dimension"},
        {"name": "Start Date", "type": "date", "role": "dimension"},
        {"name": "Duration", "type": "float", "role": "measure"},
    ],
}

MOCK_PIE_INTENT = {
    "viz_type": "pie",
    "title": "Sales Share by Category",
    "x_field": "Category",
    "y_field": "Sales",
    "color_field": None,
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_SCATTER_INTENT = {
    "viz_type": "scatter",
    "title": "Sales vs Profit",
    "x_field": "Sales",
    "y_field": "Profit",
    "color_field": "Category",
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_AREA_INTENT = {
    "viz_type": "area",
    "title": "Revenue Over Time",
    "x_field": "Order Date",
    "y_field": "Sales",
    "color_field": None,
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_HEATMAP_INTENT = {
    "viz_type": "heatmap",
    "title": "Sales Heatmap by Category and Sub-Category",
    "x_field": "Category",
    "y_field": "Sub-Category",
    "color_field": "Sales",
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_TEXT_INTENT = {
    "viz_type": "text",
    "title": "Sales by Category and Sub-Category",
    "x_field": "Sub-Category",
    "y_field": "Sales",
    "color_field": "Category",
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_GANTT_INTENT = {
    "viz_type": "gantt",
    "title": "Project Timeline",
    "x_field": "Task",
    "y_field": "Duration",
    "color_field": "Start Date",
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_KPI_INTENT = {
    "viz_type": "kpi",
    "title": "Total Clients",
    "x_field": "Customer",
    "y_field": "",
    "color_field": None,
    "filters": [],
    "sort": None,
    "aggregation": "COUNTD",
    "color_scheme": "tableau10",
}

MOCK_COMBO_INTENT = {
    "viz_type": "combo",
    "title": "Sales and Profit by Category",
    "x_field": "Category",
    "y_field": "Sales",
    "color_field": "Profit",
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}

MOCK_TREEMAP_INTENT = {
    "viz_type": "treemap",
    "title": "Revenue by Category",
    "x_field": "Category",
    "y_field": "Sales",
    "color_field": None,
    "filters": [],
    "sort": None,
    "aggregation": "SUM",
    "color_scheme": "tableau10",
}


@pytest.mark.asyncio
async def test_pie_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_PIE_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me the share of sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "pie"
    assert data["twb_filename"].endswith(".twb")
    assert data["twb_download_url"].startswith("/download/")


@pytest.mark.asyncio
async def test_scatter_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_SCATTER_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me the correlation between sales and profit",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "scatter"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_area_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_AREA_INTENT)
        response = await client.post("/chat", json={
            "question": "Show cumulative revenue over time",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "area"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_heatmap_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_HEATMAP_INTENT)
        response = await client.post("/chat", json={
            "question": "Show a heatmap of sales by category and sub-category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "heatmap"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_treemap_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_TREEMAP_INTENT)
        response = await client.post("/chat", json={
            "question": "Show a treemap of revenue by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "treemap"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_kpi_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_KPI_INTENT)
        response = await client.post("/chat", json={
            "question": "Montre moi le total de clients",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "kpi"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_combo_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_COMBO_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me sales as bars and profit as a line by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "combo"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_text_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_TEXT_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me a table of sales by category and sub-category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "text"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_gantt_chart_generation(client):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_GANTT_INTENT)
        response = await client.post("/chat", json={
            "question": "Show me a Gantt chart of project tasks",
            "session_id": str(uuid.uuid4()),
            "metadata": GANTT_METADATA,
            "conversation_history": [],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["viz_intent"]["viz_type"] == "gantt"
    assert data["twb_filename"].endswith(".twb")


@pytest.mark.asyncio
async def test_new_chart_twb_is_valid_xml(client):
    """Download the generated .twb for a pie chart and verify it is valid XML."""
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(MOCK_PIE_INTENT)
        chat_resp = await client.post("/chat", json={
            "question": "Show me the share of sales by category",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })
    assert chat_resp.status_code == 200
    download_url = chat_resp.json()["twb_download_url"]
    dl_resp = await client.get(download_url)
    assert dl_resp.status_code == 200
    content = dl_resp.content.decode("utf-8")
    assert "<?xml" in content
    assert "<workbook" in content
