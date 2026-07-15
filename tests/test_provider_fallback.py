"""Provider fallback tests (P7) — providers fall back along their chain
(mistral → google → openrouter ; google/groq → openrouter) on a transient
failure (5xx, timeout, transport error, malformed/empty response) when the
next provider has a key; hard errors (bad key, no credits) propagate.

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
    """Without a Mistral key, Google still falls straight to OpenRouter."""
    monkeypatch.setattr(settings, "llm_provider", "google")
    monkeypatch.setattr(settings, "google_api_key", "g-key")
    monkeypatch.setattr(settings, "mistral_api_key", "")
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


@pytest.mark.asyncio
async def test_google_falls_back_to_mistral_first(monkeypatch):
    """With a Mistral key configured, a Gemini 429 lands on Mistral (the paid,
    reliable fallback) BEFORE the free OpenRouter model (2026-07-09 decision:
    gemini 20/20 primary, mistral-large 16/20 fallback #1)."""
    monkeypatch.setattr(settings, "llm_provider", "google")
    monkeypatch.setattr(settings, "google_api_key", "g-key")
    monkeypatch.setattr(settings, "mistral_api_key", "m-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "or-key")
    monkeypatch.setattr(settings, "mistral_base_url", "https://api.mistral.ai/v1")
    success = LLMResponse(content="from mistral")
    with patch("llm._call_google", new_callable=AsyncMock) as mock_google, \
         patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_oc:
        mock_google.side_effect = ValueError("[google] HTTP 429: Rate limit reached.")
        mock_oc.return_value = success
        result = await call_llm(MESSAGES)
    assert result.content == "from mistral"
    assert mock_google.call_count == 1
    assert mock_oc.call_count == 1
    assert mock_oc.call_args[0][0] == "https://api.mistral.ai/v1"


# ---------------------------------------------------------------------------
# Mistral → Google → OpenRouter chain
# ---------------------------------------------------------------------------

@pytest.fixture
def mistral_full_chain(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mistral")
    monkeypatch.setattr(settings, "mistral_api_key", "m-key")
    monkeypatch.setattr(settings, "google_api_key", "g-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "or-key")
    monkeypatch.setattr(settings, "mistral_model_id", "mistral-medium-latest")


@pytest.mark.asyncio
async def test_mistral_falls_back_to_google_on_429(mistral_full_chain):
    success = LLMResponse(content="from google")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_oc, \
         patch("llm._call_google", new_callable=AsyncMock) as mock_google:
        mock_oc.side_effect = ValueError("[mistral] HTTP 429: Rate limit reached.")
        mock_google.return_value = success
        result = await call_llm(MESSAGES)
    assert result.content == "from google"
    assert mock_oc.call_count == 1        # mistral only — chain stopped at google
    assert mock_google.call_count == 1


@pytest.mark.asyncio
async def test_mistral_then_google_then_openrouter(mistral_full_chain):
    """Both mistral AND google transiently down → the terminal OpenRouter answers."""
    success = LLMResponse(content="from openrouter")
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_oc, \
         patch("llm._call_google", new_callable=AsyncMock) as mock_google:
        mock_oc.side_effect = [
            ValueError("[mistral] HTTP 503: Provider is temporarily unavailable."),
            success,
        ]
        mock_google.side_effect = ValueError("[google] HTTP 500: Provider had an internal server error.")
        result = await call_llm(MESSAGES)
    assert result.content == "from openrouter"
    assert mock_oc.call_count == 2        # mistral, then openrouter
    assert mock_google.call_count == 1


@pytest.mark.asyncio
async def test_mistral_hard_error_propagates(mistral_full_chain):
    with patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_oc, \
         patch("llm._call_google", new_callable=AsyncMock) as mock_google:
        mock_oc.side_effect = ValueError("[mistral] HTTP 401: Invalid API key.")
        with pytest.raises(ValueError, match="401"):
            await call_llm(MESSAGES)
    assert mock_oc.call_count == 1
    assert mock_google.call_count == 0


class _FakeNullToolCallsResponse:
    """A Mistral-shaped plain-text answer: explicit "tool_calls": null."""
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "salut", "tool_calls": None}}]}


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return _FakeNullToolCallsResponse()


@pytest.mark.asyncio
async def test_explicit_null_tool_calls_is_parsed_as_no_tool_calls(monkeypatch):
    """Mistral emits "tool_calls": null on plain-text answers (OpenRouter/Groq
    omit the key). The parser must treat it as an empty list — the reported
    live crash was TypeError('NoneType' object is not iterable) surfacing as
    an ASGI 500 because TypeError isn't transient-classified."""
    from llm import _call_openai_compatible
    monkeypatch.setattr("llm.httpx.AsyncClient", _FakeClient)
    result = await _call_openai_compatible(
        "https://api.mistral.ai/v1", "key", "mistral-medium-latest", MESSAGES,
        provider_label="mistral")
    assert result.content == "salut"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_mistral_without_key_uses_google(monkeypatch):
    """LLM_PROVIDER=mistral with an empty key → Google serves transparently
    (the state of a fresh .env before the user pastes MISTRAL_API_KEY)."""
    monkeypatch.setattr(settings, "llm_provider", "mistral")
    monkeypatch.setattr(settings, "mistral_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "g-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "or-key")
    success = LLMResponse(content="from google")
    with patch("llm._call_google", new_callable=AsyncMock) as mock_google, \
         patch("llm._call_openai_compatible", new_callable=AsyncMock) as mock_oc:
        mock_google.return_value = success
        result = await call_llm(MESSAGES)
    assert result.content == "from google"
    assert mock_google.call_count == 1
    assert mock_oc.call_count == 0
