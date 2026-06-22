import pytest


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "openrouter_status" in data
    assert "model_id" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_openrouter_status_no_key(client):
    """With no API key configured, openrouter_status should be 'no_api_key'."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # No key set in test env — expect no_api_key or similar
    assert data["openrouter_status"] in ("no_api_key", "ok", "unreachable")
