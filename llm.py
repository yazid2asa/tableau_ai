import copy
import json
import logging
from dataclasses import dataclass, field

import httpx
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def first_tool_args(self) -> dict | None:
        return self.tool_calls[0].arguments if self.tool_calls else None

_HTTP_ERROR_MESSAGES = {
    429: "Rate limit reached. Wait a moment and try again.",
    401: "Invalid API key.",
    402: "Account has insufficient credits.",
    500: "Provider had an internal server error.",
    502: "Bad gateway from provider.",
    503: "Provider is temporarily unavailable. Try again shortly.",
    504: "Provider gateway timeout.",
}

# Substrings that mark an error as worth retrying on a fallback provider. The
# numeric codes are always present because raised messages are prefixed with
# "HTTP {code}:" (see _call_openai_compatible).
_TRANSIENT_MARKERS = (
    "429", "rate limit",
    "500", "502", "503", "504",
    "temporarily unavailable", "timed out", "timeout",
    "malformed", "unexpected response structure",
    "neither content nor tool calls",
)


def _is_transient_error(exc: Exception) -> bool:
    """True for errors worth retrying on a fallback provider: rate limits, 5xx,
    timeouts, transport failures, and malformed/empty responses."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


async def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    timeout: float = 60.0,
    provider_label: str = "provider",
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Shared caller for OpenAI-compatible APIs (OpenRouter, Groq)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter" in base_url:
        headers["HTTP-Referer"] = "http://localhost:8000"
        headers["X-Title"] = "Text-to-Viz Agent"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            msg = _HTTP_ERROR_MESSAGES.get(code, str(exc))
            raise ValueError(f"[{provider_label}] HTTP {code}: {msg}") from exc

        try:
            message = r.json()["choices"][0]["message"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"[{provider_label}] Malformed response: {exc}"
            ) from exc
        content = message.get("content")
        tool_calls_raw = message.get("tool_calls", [])

        parsed_tool_calls = []
        for tc in tool_calls_raw:
            func = tc["function"]
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            parsed_tool_calls.append(ToolCall(name=func["name"], arguments=args))

        if not content and not parsed_tool_calls:
            raise ValueError(
                f"[{provider_label}] LLM returned neither content nor tool calls."
            )
        return LLMResponse(content=content, tool_calls=parsed_tool_calls)


def _convert_messages_to_google(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Google Generative AI format.

    Returns (system_instruction, contents) where system_instruction is extracted
    from the system message and contents is the list of user/model parts.
    """
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_instruction = text
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return system_instruction, contents


def _sanitize_schema_for_google(schema: dict) -> dict:
    """Convert OpenAI-style JSON Schema to Google-compatible format.

    Google rejects: union types ["string", "null"], None in enums, None defaults,
    missing type fields, arrays without items.
    """
    if not schema:
        return {"type": "string"}

    out = {}
    for key, value in schema.items():
        if key == "default":
            continue
        if key == "type" and isinstance(value, list):
            real_types = [t for t in value if t is not None and t != "null"]
            out["type"] = real_types[0] if real_types else "string"
            continue
        if key == "enum" and isinstance(value, list):
            cleaned = [v for v in value if v is not None]
            if cleaned:
                out["enum"] = cleaned
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {
                k: _sanitize_schema_for_google(v) for k, v in value.items()
            }
            continue
        if key == "items" and isinstance(value, dict):
            out["items"] = _sanitize_schema_for_google(value)
            continue
        out[key] = value

    if "type" not in out:
        out["type"] = "string"
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}

    return out


def _convert_tools_for_google(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tools to Google functionDeclarations format."""
    declarations = []
    for tool in tools:
        func = tool.get("function", {})
        decl = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        if "parameters" in func:
            decl["parameters"] = _sanitize_schema_for_google(func["parameters"])
        declarations.append(decl)
    return declarations


async def _call_google(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call Google Generative AI via the google-genai SDK."""
    from google import genai

    api_key = settings.google_api_key
    if not api_key:
        raise ValueError("[google] No GOOGLE_API_KEY configured.")

    system_instruction, contents = _convert_messages_to_google(messages)

    config: dict = {"temperature": 0.1}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if tools:
        func_declarations = _convert_tools_for_google(tools)
        if func_declarations:
            config["tools"] = [{"function_declarations": func_declarations}]

    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        # Extract HTTP status code from the exception (sdk raises ClientError/ServerError
        # with a status_code attribute, or embeds the code in the message).
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if code:
            msg = _HTTP_ERROR_MESSAGES.get(int(code), str(exc))
            raise ValueError(f"[google] HTTP {code}: {msg}") from exc
        raise ValueError(f"[google] Error: {exc}") from exc

    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"[google] Unexpected response structure: {response}") from exc

    content = None
    parsed_tool_calls: list[ToolCall] = []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            parsed_tool_calls.append(ToolCall(
                name=fc.name,
                arguments=dict(fc.args) if fc.args else {},
            ))
        elif getattr(part, "text", None):
            content = part.text

    if not content and not parsed_tool_calls:
        raise ValueError("[google] LLM returned neither content nor tool calls.")
    return LLMResponse(content=content, tool_calls=parsed_tool_calls)


async def call_llm(
    messages: list[dict],
    model_override: str | None = None,
    provider_override: str | None = None,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the LLM via the configured provider.

    Groq and Google fall back to OpenRouter on a transient failure (rate limit,
    5xx, timeout, transport error, malformed/empty response) when an OpenRouter
    key is available. Hard errors (bad key, no credits) propagate immediately.
    """
    provider = provider_override or settings.llm_provider

    if provider == "groq":
        if not settings.groq_api_key:
            logger.info("llm: no GROQ_API_KEY configured, falling back to OpenRouter")
            provider = "openrouter"
        else:
            model = model_override or settings.groq_model_id
            try:
                result = await _call_openai_compatible(
                    settings.groq_base_url, settings.groq_api_key, model, messages,
                    timeout=30.0, provider_label="groq", tools=tools,
                )
                logger.info("llm: provider=groq model=%s", model)
                return result
            except Exception as exc:
                if _is_transient_error(exc) and settings.openrouter_api_key:
                    logger.warning("llm: Groq transient error (%s), falling back to OpenRouter", exc)
                    provider = "openrouter"
                else:
                    raise

    if provider == "google":
        if not settings.google_api_key:
            logger.info("llm: no GOOGLE_API_KEY configured, falling back to OpenRouter")
            provider = "openrouter"
        else:
            model = model_override or settings.google_model_id
            try:
                result = await _call_google(model, messages, tools=tools)
                logger.info("llm: provider=google model=%s", model)
                return result
            except Exception as exc:
                if _is_transient_error(exc) and settings.openrouter_api_key:
                    logger.warning("llm: Google transient error (%s), falling back to OpenRouter", exc)
                    provider = "openrouter"
                else:
                    raise

    if provider == "openrouter":
        model = model_override or settings.model_id
        result = await _call_openai_compatible(
            settings.openrouter_base_url, settings.openrouter_api_key, model, messages,
            timeout=60.0, provider_label="openrouter", tools=tools,
        )
        logger.info("llm: provider=openrouter model=%s", model)
        return result

    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'groq', 'openrouter', or 'google'.")


def get_active_provider() -> str:
    """The provider call_llm will actually use, accounting for no-key fallback."""
    provider = settings.llm_provider
    if provider == "groq" and not settings.groq_api_key:
        return "openrouter"
    if provider == "google" and not settings.google_api_key:
        return "openrouter"
    return provider


def get_active_model() -> str:
    """The model id of the active provider (see get_active_provider)."""
    provider = get_active_provider()
    if provider == "groq":
        return settings.groq_model_id
    if provider == "google":
        return settings.google_model_id
    return settings.model_id


async def check_provider_status() -> str:
    """Probe the ACTIVE provider. Returns 'ok', 'no_api_key', 'unreachable', or 'error_{code}'."""
    provider = get_active_provider()

    if provider == "google":
        if not settings.google_api_key:
            return "no_api_key"
        try:
            from google import genai
            client = genai.Client(api_key=settings.google_api_key)
            await client.aio.models.get(model=settings.google_model_id)
            return "ok"
        except Exception:
            return "unreachable"

    # OpenAI-compatible providers (groq / openrouter)
    if provider == "groq":
        base_url, api_key = settings.groq_base_url, settings.groq_api_key
    else:
        base_url, api_key = settings.openrouter_base_url, settings.openrouter_api_key

    if not api_key:
        return "no_api_key"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            return "ok" if r.status_code == 200 else f"error_{r.status_code}"
    except Exception:
        return "unreachable"
