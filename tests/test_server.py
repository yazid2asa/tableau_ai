"""Tests for Tableau Server integration — all Server calls mocked."""
import json
import uuid

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from conftest import make_tool_response
from schemas import DataSourceMetadata, FieldInfo, FieldType


MOCK_BAR_INTENT = {
    "viz_type": "bar_chart",
    "title": "Sales by Region",
    "x_field": "Region",
    "y_field": "Revenue",
    "color_field": None,
    "filters": [],
    "sort": "descending",
    "aggregation": "SUM",
    "color_scheme": "tableau10",
    "action": "new",
    "datasource_luid": "ds-luid-1",
}

SAMPLE_METADATA = {
    "datasource_name": "superstore",
    "datasource_caption": "Sample - Superstore",
    "fields": [
        {"name": "Region", "type": "string", "role": "dimension"},
        {"name": "Revenue", "type": "float", "role": "measure"},
    ],
}


# 1. test_signin_returns_token
@pytest.mark.asyncio
async def test_signin_returns_token():
    """signin() signs in via TSC and returns (token, site_id)."""
    mock_server = MagicMock()
    mock_server.auth_token = "test-token-123"
    mock_server.site_id = "site-luid-456"
    mock_server.auth.sign_in = MagicMock()

    with patch("tableau_server._get_server", return_value=mock_server):
        import tableau_server
        tableau_server._signed_in = False
        tableau_server._server = mock_server

        with patch("tableau_server.settings") as mock_settings:
            mock_settings.tableau_pat_name = "test-pat"
            mock_settings.tableau_pat_secret = "test-secret"
            mock_settings.tableau_site_id = "test-site"
            mock_settings.tableau_server_url = "https://tableau.test.com"

            token, site_id = await tableau_server.signin()
            assert token == "test-token-123"
            assert site_id == "site-luid-456"
            tableau_server._signed_in = False


# 2. test_publish_workbook_returns_luid
@pytest.mark.asyncio
async def test_publish_workbook_returns_luid(tmp_path):
    """publish_workbook() uses TSC to publish and returns workbook LUID."""
    twb_file = tmp_path / "test.twb"
    twb_file.write_text("<workbook/>")

    mock_published = MagicMock()
    mock_published.id = "wb-luid-789"
    mock_published.name = "test"

    mock_server = MagicMock()
    mock_server.auth_token = "token"
    mock_server.site_id = "site-id"
    mock_server.auth.sign_in = MagicMock()
    mock_server.workbooks.publish.return_value = mock_published
    mock_server.projects.get.return_value = (
        [MagicMock(id="project-123", name="default")], MagicMock()
    )

    with patch("tableau_server._get_server", return_value=mock_server):
        import tableau_server
        tableau_server._signed_in = False
        tableau_server._server = mock_server
        tableau_server._cached_project_luid = None

        with patch("tableau_server.settings") as mock_settings:
            mock_settings.tableau_pat_name = "test-pat"
            mock_settings.tableau_pat_secret = "test-secret"
            mock_settings.tableau_site_id = "test-site"
            mock_settings.tableau_server_url = "https://tableau.test.com"

            luid = await tableau_server.publish_workbook(str(twb_file), "project-123")
            assert luid == "wb-luid-789"
            mock_server.workbooks.publish.assert_called_once()

        tableau_server._signed_in = False
        tableau_server._cached_project_luid = None


# 3. test_get_all_datasource_schemas_returns_metadata_list
@pytest.mark.asyncio
async def test_get_all_datasource_schemas_returns_metadata_list():
    """get_all_datasource_schemas() parses GraphQL response into DataSourceMetadata list."""
    graphql_response = {
        "data": {
            "publishedDatasourcesConnection": {
                "nodes": [
                    {
                        "luid": "ds-luid-1",
                        "name": "Sales Data",
                        "fields": [
                            {"name": "Region", "dataType": "STRING", "role": "DIMENSION"},
                            {"name": "Revenue", "dataType": "REAL", "role": "MEASURE"},
                        ],
                    },
                    {
                        "luid": "ds-luid-2",
                        "name": "Products",
                        "fields": [
                            {"name": "Product Name", "dataType": "STRING", "role": "DIMENSION"},
                        ],
                    },
                ]
            }
        }
    }

    mock_server = MagicMock()
    mock_server.auth_token = "token"
    mock_server.site_id = "site-id"
    mock_server.auth.sign_in = MagicMock()
    mock_server.metadata.query.return_value = graphql_response

    with patch("tableau_server._get_server", return_value=mock_server):
        import tableau_server
        tableau_server._signed_in = False
        tableau_server._server = mock_server

        with patch("tableau_server.settings") as mock_settings:
            mock_settings.tableau_pat_name = "test-pat"
            mock_settings.tableau_pat_secret = "test-secret"
            mock_settings.tableau_site_id = "test-site"
            mock_settings.tableau_server_url = "https://tableau.test.com"
            mock_settings.tableau_datasource_filter = ""

            schemas = await tableau_server.get_all_datasource_schemas()

            assert len(schemas) == 2
            assert schemas[0].datasource_name == "Sales Data"
            assert schemas[0].luid == "ds-luid-1"
            assert len(schemas[0].fields) == 2
            assert schemas[0].fields[0].name == "Region"
            assert schemas[0].fields[0].type == FieldType.STRING
            assert schemas[0].fields[0].role == "dimension"
            assert schemas[0].fields[1].name == "Revenue"
            assert schemas[0].fields[1].type == FieldType.FLOAT
            assert schemas[0].fields[1].role == "measure"
            assert schemas[1].datasource_name == "Products"

        tableau_server._signed_in = False


# 4. test_llm_receives_all_schemas_in_prompt
def test_llm_receives_all_schemas_in_prompt():
    """build_intent_prompt() includes all datasource schemas when available_datasources is provided."""
    from prompts import build_intent_prompt

    ds1 = DataSourceMetadata(
        datasource_name="Sales",
        luid="luid-1",
        fields=[
            FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
            FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure"),
        ],
    )
    ds2 = DataSourceMetadata(
        datasource_name="Products",
        luid="luid-2",
        fields=[
            FieldInfo(name="Product Name", type=FieldType.STRING, role="dimension"),
            FieldInfo(name="Price", type=FieldType.FLOAT, role="measure"),
        ],
    )

    messages = build_intent_prompt(
        question="Show revenue by product",
        metadata=None,
        history=[],
        available_datasources=[ds1, ds2],
    )

    user_msg = messages[-1]["content"]
    assert "Sales" in user_msg
    assert "Products" in user_msg
    assert "Region" in user_msg
    assert "Revenue" in user_msg
    assert "Product Name" in user_msg
    assert "Price" in user_msg
    assert "luid-1" in user_msg
    assert "luid-2" in user_msg
    assert "datasource_luid" in user_msg


# 5. test_chat_returns_view_url_when_server_configured
@pytest.mark.asyncio
async def test_chat_returns_view_url_when_server_configured(client):
    """When server mode is active, /chat publishes the workbook and returns view_url."""
    mock_schemas = [
        DataSourceMetadata(
            datasource_name="Sales Data",
            luid="ds-luid-1",
            fields=[
                FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
                FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure"),
            ],
        ),
    ]

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.publish_workbook", new_callable=AsyncMock) as mock_publish, \
         patch("main.get_view_url", new_callable=AsyncMock) as mock_view_url, \
         patch("main.get_all_datasource_schemas", new_callable=AsyncMock) as mock_get_schemas, \
         patch("main.get_datasource_content_url", new_callable=AsyncMock) as mock_content_url:

        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        mock_publish.return_value = "wb-luid-001"
        mock_view_url.return_value = "https://tableau.test.com/#/site/test/views/SalesByRegion/Sheet1"
        mock_get_schemas.return_value = mock_schemas
        mock_content_url.return_value = "SalesData"

        response = await client.post("/chat", json={
            "question": "Show revenue by region",
            "session_id": str(uuid.uuid4()),
            "metadata": SAMPLE_METADATA,
            "conversation_history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["view_url"] == "https://tableau.test.com/#/site/test/views/SalesByRegion/Sheet1"
    assert data["viz_intent"]["viz_type"] == "bar_chart"
    assert data["twb_filename"].endswith(".twb")


# 6. test_same_workbook_accumulation_on_server
@pytest.mark.asyncio
async def test_same_workbook_accumulation_on_server(client):
    """Second question in same session adds sheet to same workbook, re-publishes."""
    mock_schemas = [
        DataSourceMetadata(
            datasource_name="Sales Data",
            luid="ds-luid-1",
            fields=[
                FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
                FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure"),
            ],
        ),
    ]

    session_id = str(uuid.uuid4())

    second_intent = {
        "viz_type": "line_chart",
        "title": "Revenue Trend",
        "x_field": "Region",
        "y_field": "Revenue",
        "color_field": None,
        "filters": [],
        "sort": None,
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
        "datasource_luid": "ds-luid-1",
    }

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.publish_workbook", new_callable=AsyncMock) as mock_publish, \
         patch("main.get_view_url", new_callable=AsyncMock) as mock_view_url, \
         patch("main.get_all_datasource_schemas", new_callable=AsyncMock) as mock_get_schemas, \
         patch("main.get_datasource_content_url", new_callable=AsyncMock) as mock_content_url:

        mock_llm.return_value = make_tool_response(MOCK_BAR_INTENT)
        mock_publish.return_value = "wb-luid-001"
        mock_view_url.return_value = "https://tableau.test.com/#/views/Analyse/Sheet1"
        mock_get_schemas.return_value = mock_schemas
        mock_content_url.return_value = "SalesData"

        # First request — creates new workbook
        resp1 = await client.post("/chat", json={
            "question": "Show revenue by region",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
        })
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["view_url"] is not None

        # Second request — should add sheet to same workbook
        mock_llm.return_value = make_tool_response(second_intent)
        mock_publish.return_value = "wb-luid-001"  # same workbook LUID (overwrite)

        resp2 = await client.post("/chat", json={
            "question": "Show revenue trend",
            "session_id": session_id,
            "metadata": SAMPLE_METADATA,
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["view_url"] is not None
        # Second request should re-use the session workbook (sheet_added mode)
        assert data2["mode"] == "sheet_added"

    # publish_workbook called twice (once per question)
    assert mock_publish.call_count == 2


# 7. test_server_metadata_used_for_field_validation
@pytest.mark.asyncio
async def test_server_metadata_used_for_field_validation(client):
    """In Server mode, datasource schema from GraphQL is used for field validation."""
    mock_schemas = [
        DataSourceMetadata(
            datasource_name="Sales Data",
            luid="ds-luid-1",
            fields=[
                FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
                FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure"),
            ],
        ),
    ]

    # LLM returns intent with wrong field name
    bad_intent = {
        "viz_type": "bar_chart",
        "title": "Sales by Region",
        "x_field": "Regio",  # close match to "Region"
        "y_field": "Revenue",
        "filters": [],
        "aggregation": "SUM",
        "color_scheme": "tableau10",
        "action": "new",
        "datasource_luid": "ds-luid-1",
    }

    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.publish_workbook", new_callable=AsyncMock) as mock_publish, \
         patch("main.get_view_url", new_callable=AsyncMock) as mock_view_url, \
         patch("main.get_all_datasource_schemas", new_callable=AsyncMock) as mock_get_schemas, \
         patch("main.get_datasource_content_url", new_callable=AsyncMock) as mock_content_url:

        mock_llm.return_value = make_tool_response(bad_intent)
        mock_publish.return_value = "wb-luid-001"
        mock_view_url.return_value = "https://tableau.test.com/#/views/Test/Sheet1"
        mock_get_schemas.return_value = mock_schemas
        mock_content_url.return_value = "SalesData"

        response = await client.post("/chat", json={
            "question": "Show sales by region",
            "session_id": str(uuid.uuid4()),
        })

    assert response.status_code == 200
    data = response.json()
    # "Regio" should be auto-corrected to "Region" since it's a close match
    assert data["viz_intent"]["x_field"] == "Region"
