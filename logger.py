"""Structured JSONL logging for LLM traces.

Writes one JSON object per line to logs/llm_traces.jsonl with auto-rotation
(max 100 MB per file, 30 backup files kept).
"""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_LOGS_DIR = Path("logs")
_LOG_FILE = _LOGS_DIR / "llm_traces.jsonl"

# Ensure the directory exists at import time.
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated logger — does NOT propagate to the root logger so it won't mix
# with uvicorn / FastAPI log output.
_logger = logging.getLogger("llm_traces")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

if not _logger.handlers:
    _handler = RotatingFileHandler(
        filename=str(_LOG_FILE),
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=30,
        encoding="utf-8",
    )
    # Each record's message is already a JSON string; just emit the message
    # followed by a newline (no extra formatting).
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_trace(
    trace_id: str,
    session_id: str,
    question: str,
    viz_type: str,
    title: str,
    model_id: str,
    latency_ms: float,
    token_usage: dict,
    judge_score: float | None,
    judge_feedback: str | None,
    status: str,
    error_message: str | None = None,
) -> None:
    """Append one structured trace entry to the JSONL log file.

    Parameters
    ----------
    trace_id:       Unique identifier for this generation trace.
    session_id:     Conversation session identifier.
    question:       Original natural-language question from the user.
    viz_type:       Resolved visualization type (e.g. "bar_chart").
    title:          Title of the generated workbook/sheet.
    model_id:       LLM model used (e.g. "minimax/minimax-m2.5:free").
    latency_ms:     End-to-end generation latency in milliseconds.
    token_usage:    Dict with keys "prompt_tokens" and "completion_tokens".
    judge_score:    Quality score from LLM-as-a-Judge (0.0–1.0) or None.
    judge_feedback: Human-readable feedback from the judge or None.
    status:         One of "success", "error", or "partial".
    error_message:  Error details when status is "error", otherwise None.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "session_id": session_id,
        "question": question,
        "viz_type": viz_type,
        "title": title,
        "model_id": model_id,
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "judge_score": judge_score,
        "judge_feedback": judge_feedback,
        "status": status,
        "error_message": error_message,
    }
    _logger.info(json.dumps(entry, ensure_ascii=False))


def get_recent_traces(n: int = 50) -> list[dict]:
    """Read the last *n* traces from the JSONL log file.

    Returns a list of dicts (most-recent last, matching file order).
    Returns an empty list if the file does not exist or is empty.
    """
    if not _LOG_FILE.exists():
        return []

    lines: list[str] = []
    try:
        with open(_LOG_FILE, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    # Take only the last n non-empty lines.
    recent_lines = [line for line in lines if line.strip()][-n:]

    traces: list[dict] = []
    for line in recent_lines:
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines rather than crashing.
            continue

    return traces
