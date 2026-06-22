"""
observability.py — LangFuse v4 tracing layer for Text-to-Viz Agent.

LangFuse SDK v4 API (breaking change from v2/v3):
  - langfuse.start_observation(trace_context=..., name=..., as_type="span"|"generation", ...)
  - span.start_observation(...)  ← creates a child observation
  - span.score_trace(name=..., value=..., comment=...)
  - usage_details={"input": N, "output": N}  (not promptTokens/completionTokens)
  - langfuse.flush() / langfuse.shutdown()

All public functions degrade silently to no-ops when:
  - the `langfuse` package is not installed, or
  - LangFuse is disabled via config (langfuse_enabled=False), or
  - credentials (public_key / secret_key) are absent, or
  - the LangFuse host is unreachable at runtime.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LangFuse import — graceful degradation if package is absent
# ---------------------------------------------------------------------------

try:
    from langfuse import Langfuse  # type: ignore
    from langfuse.types import TraceContext  # type: ignore

    _LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning(
        "langfuse package not installed — LangFuse tracing disabled. "
        "Install with: pip install langfuse"
    )
    _LANGFUSE_AVAILABLE = False

# Module-level singleton — set by init_langfuse()
_client: "Langfuse | None" = None


# ---------------------------------------------------------------------------
# No-op sentinel — mirrors the v4 span API
# ---------------------------------------------------------------------------


class _NoOpTrace:
    """Returned whenever LangFuse is disabled or unavailable."""

    def start_observation(self, **kwargs: Any) -> "_NoOpTrace":
        return self

    def end(self, **kwargs: Any) -> "_NoOpTrace":
        return self

    def score_trace(self, **kwargs: Any) -> None:
        pass

    def score(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> "_NoOpTrace":
        return self


_NOOP_TRACE = _NoOpTrace()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_langfuse() -> None:
    """
    Initialize the LangFuse client singleton.
    Called once at application startup (inside FastAPI lifespan).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _client

    if _client is not None:
        return

    if not _LANGFUSE_AVAILABLE:
        return

    try:
        from config import settings  # type: ignore
    except Exception as exc:
        logger.warning("observability: could not import settings — LangFuse disabled: %s", exc)
        return

    enabled: bool = getattr(settings, "langfuse_enabled", True)
    public_key: str = getattr(settings, "langfuse_public_key", "")
    secret_key: str = getattr(settings, "langfuse_secret_key", "")
    host: str = getattr(settings, "langfuse_host", "https://cloud.langfuse.com")

    if not enabled:
        logger.info("observability: LangFuse disabled via config")
        return

    if not public_key or not secret_key:
        logger.info("observability: LangFuse credentials not set — tracing disabled")
        return

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("observability: LangFuse v4 client initialised (host=%s)", host)
    except Exception as exc:
        logger.warning("observability: failed to initialise LangFuse client: %s", exc)
        _client = None


def create_trace(
    trace_id: str,
    session_id: str,
    question: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Create a root LangFuse trace span for a single chat request.
    Returns a span object (or _NoOpTrace if LangFuse is unavailable).

    The returned object is passed to add_generation_span() and score_trace().
    """
    if _client is None:
        return _NOOP_TRACE

    try:
        trace_context = TraceContext(trace_id=trace_id, session_id=session_id)
        root = _client.start_observation(
            trace_context=trace_context,
            name="chat",
            as_type="span",
            input={"question": question},
            metadata=metadata or {},
        )
        return root
    except Exception as exc:
        logger.warning("observability: create_trace failed — %s", exc)
        return _NOOP_TRACE


def add_generation_span(
    trace: Any,
    name: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    model: str | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """
    Add a child generation span to an existing trace.

    Parameters
    ----------
    trace:       The span returned by create_trace().
    name:        Span name — e.g. "llm_call", "judge_validation".
    input_data:  Input dict (e.g. {"messages": [...]}).
    output_data: Output dict (e.g. {"response": "..."}).
    model:       Optional model identifier.
    usage:       Optional {"prompt_tokens": N, "completion_tokens": N}.
                 Mapped to v4 usage_details={"input": N, "output": N}.
    """
    if isinstance(trace, _NoOpTrace):
        return _NOOP_TRACE

    try:
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": "generation",
            "input": input_data,
            "output": output_data,
        }
        if model is not None:
            kwargs["model"] = model
        if usage:
            kwargs["usage_details"] = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }

        gen = trace.start_observation(**kwargs)
        gen.end()
        return gen
    except Exception as exc:
        logger.warning("observability: add_generation_span failed — %s", exc)
        return _NOOP_TRACE


def score_trace(
    trace: Any,
    judge_score: float,
    judge_feedback: str,
) -> None:
    """
    Attach an LLM-as-a-Judge quality score to the trace.

    Parameters
    ----------
    trace:          The span returned by create_trace().
    judge_score:    Float in [0, 1].
    judge_feedback: Human-readable explanation from the judge.
    """
    if isinstance(trace, _NoOpTrace):
        return

    try:
        trace.score_trace(
            name="judge_score",
            value=judge_score,
            comment=judge_feedback,
        )
    except Exception as exc:
        logger.warning("observability: score_trace failed — %s", exc)


def end_trace(trace: Any, output: dict[str, Any] | None = None) -> None:
    """End the root trace span."""
    if isinstance(trace, _NoOpTrace):
        return
    try:
        trace.end()
    except Exception as exc:
        logger.warning("observability: end_trace failed — %s", exc)


def flush() -> None:
    """Flush the LangFuse client buffer on application shutdown."""
    if _client is None:
        return
    try:
        _client.flush()
        logger.info("observability: LangFuse client flushed")
    except Exception as exc:
        logger.warning("observability: flush failed — %s", exc)
