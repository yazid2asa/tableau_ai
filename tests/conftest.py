import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from database import init_db
from llm import LLMResponse, ToolCall
from main import app


def make_tool_response(intent_dict: dict) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(name="generate_chart", arguments=intent_dict)]
    )


def make_text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text)


@pytest_asyncio.fixture(autouse=True, scope="session")
async def setup_db():
    """Ensure all DB tables exist before any test runs."""
    await init_db()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def mock_judge_viz():
    """Auto-mock judge_viz in all tests to avoid real LLM API calls.

    quick_validate is NOT mocked — tests exercise the real hybrid-judge gate
    (QUICK_VALIDATE_SKIP_THRESHOLD). The test datasources keep field roles clean,
    so a well-formed intent scores below the skip gate only when it should.
    """
    with patch("main.judge_viz", new_callable=AsyncMock) as mock:
        mock.return_value = (0.92, "Mock judge: visualization looks correct.")
        yield mock


@pytest.fixture(autouse=True)
def mock_tableau_server():
    """Mock Tableau Server API calls in all tests to avoid real API calls.

    Server-specific tests (test_server.py) override with their own mocks.
    """
    with patch("main.get_all_datasource_schemas", new_callable=AsyncMock, return_value=[]) as mock_schemas, \
         patch("main.publish_workbook", new_callable=AsyncMock, return_value="mock-luid") as mock_publish, \
         patch("main.get_view_url", new_callable=AsyncMock, return_value="https://tableau.test/view") as mock_view, \
         patch("main.get_datasource_content_url", new_callable=AsyncMock, return_value="ds-content-url") as mock_ds_url:
        yield {
            "get_all_datasource_schemas": mock_schemas,
            "publish_workbook": mock_publish,
            "get_view_url": mock_view,
            "get_datasource_content_url": mock_ds_url,
        }
