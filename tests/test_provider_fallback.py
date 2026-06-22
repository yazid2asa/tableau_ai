"""Provider fallback tests (P7) — Groq/Google fall back to OpenRouter on a
transient failure (5xx, timeout, transport error, malformed/empty response) when
an OpenRouter key is present; hard errors (bad key, no credits) propagate.

These exercise llm.call_llm directly with the network layer
(_call_openai_compatible / _call_google) mocked, so no real API calls are made.
"""
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from config import settings
from llm import call_llm, LLMResponse, _is_transient_error


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def groq_with_openrouter(monkeypatch):
    """Active provider = groq, with an OpenRouter fallback key configured."""
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "groq-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "or-key")
    monkeypatch.setattr(settings, "groq_model_id", "groq-model")
    monkeypatch.setattr(settings, "model_id", "or-model")


# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    httpx.TimeoutException("timed out"),
    httpx.ConnectError("refused"),
    ValueError("[groq] HTTP 500: Provider had an internal server error."),
    ValueError("[groq] HTTP 503: Provider is temporarily unavailable."),
    ValueError("[groq] HTTP 429: Rate limit reached."),
    ValueError("[groq] Malformed response: bad json"),
    ValueError("[groq] LLM returned neither content nor tool calls."),
])
def test_transient_errors_classified(exc):
    assert _is_transient_error(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("[groq] HTTP 401: Invalid API key."),
    ValueError("[groq] HTTP 402: Account has insufficient credits."),
])
def test_hard_errors_not_transient(exc):
    assert _is_transient_error(exc) is False


# ---------------------------------------------------------------------------
# Groq → OpenRouter fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_falls_back_to_openrouter_on_500(groq_with_openrouter):
    success = LLMResponse(content="from openrouter")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            ValueError("[groq] HTTP 500: Provider had an internal server error."),
            success,
        ]
        result = await call_llm(MESSAGES)
    assert result.content == "from openrouter"
    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_groq_falls_back_on_timeout(groq_with_openrouter):
    success = LLMResponse(content="recovered")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [httpx.TimeoutException("timed out"), success]
        result = await call_llm(MESSAGES)
    assert result.content == "recovered"
    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_groq_falls_back_on_empty_response(groq_with_openrouter):
    success = LLMResponse(content="recovered")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            ValueError("[groq] LLM returned neither content nor tool calls."),
            success,
        ]
        result = await call_llm(MESSAGES)
    assert result.content == "recovered"
    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_hard_error_does_not_fall_back(groq_with_openrouter):
    """A bad API key (401) is NOT transient — it must propagate, not fall back."""
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ValueError("[groq] HTTP 401: Invalid API key.")
        with pytest.raises(ValueError, match="401"):
            await call_llm(MESSAGES)
    # Only the groq attempt — no OpenRouter retry.
    assert mock_call.call_count == 1


@pytest.mark.asyncio
async def test_no_fallback_when_no_openrouter_key(monkeypatch):
    """Without an OpenRouter key, a transient groq error propagates (nowhere to fall back)."""
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "groq-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ValueError("[groq] HTTP 500: Provider had an internal server error.")
        with pytest.raises(ValueError, match="500"):
            await call_llm(MESSAGES)
    assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# Google → OpenRouter fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_google_falls_back_to_openrouter_on_500(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "google")
    monkeypatch.setattr(settings, "google_api_key", "g-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "or-key")
    monkeypatch.setattr(settings, "model_id", "or-model")
    success = LLMResponse(content="from openrouter")
    with patch("llm._call_google", new_callable=AsyncMock) as mock_google, \
         patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_or:
        mock_google.side_effect = ValueError("[google] HTTP 500: Provider had an internal server error.")
        mock_or.return_value = success
        result = await call_llm(MESSAGES)
    assert result.content == "from openrouter"
    assert mock_google.call_count == 1
    assert mock_or.call_count == 1
