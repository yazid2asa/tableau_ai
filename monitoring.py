"""
monitoring.py — Metrics aggregation for the Text-to-Viz monitoring dashboard.

Provides:
  - get_monitoring_metrics(db) → dict   (all KPIs for the dashboard template)
  - log_generation(db, ...)             (insert a GenerationLog row)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Feedback, GenerationLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy mapped object to a plain dict."""
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def log_generation(
    db: AsyncSession,
    trace_id: str,
    session_id: str,
    question: str,
    viz_type: str,
    title: str,
    model_id: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    judge_score: Optional[float],
    judge_feedback: Optional[str],
    status: str,
) -> None:
    """Insert a GenerationLog row into the database.

    Args:
        db:               Active async SQLAlchemy session.
        trace_id:         UUID string that ties this log to LangFuse and feedback.
        session_id:       Session UUID from the chat request.
        question:         Original natural language question from the user.
        viz_type:         Resolved viz type (e.g. "bar_chart", "line_chart").
        title:            Chart title produced by the LLM.
        model_id:         OpenRouter model identifier used for this generation.
        latency_ms:       Total end-to-end latency in milliseconds.
        prompt_tokens:    Number of prompt tokens consumed (0 when unavailable).
        completion_tokens: Number of completion tokens consumed (0 when unavailable).
        judge_score:      LLM-as-a-Judge overall score in [0, 1] or None.
        judge_feedback:   Judge textual feedback or None.
        status:           One of "success", "error", "partial".
    """
    entry = GenerationLog(
        trace_id=trace_id,
        session_id=session_id,
        question=question,
        viz_type=viz_type,
        title=title,
        model_id=model_id,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        judge_score=judge_score,
        judge_feedback=judge_feedback,
        status=status,
    )
    db.add(entry)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.error("log_generation: DB commit failed — %s", exc)


async def get_monitoring_metrics(db: AsyncSession) -> dict:
    """Aggregate all KPIs needed for the monitoring dashboard.

    Returns a flat dict with the following keys:

    total_generations   int
    success_rate        float   — percentage (0–100) of rows with status=="success"
    avg_judge_score     float   — mean of non-null judge_scores; 0.0 when none exist
    avg_latency_ms      float   — mean latency; 0.0 when no rows
    p95_latency_ms      float   — 95th-percentile latency; 0.0 when < 20 rows
    viz_type_distribution  dict — {"bar_chart": N, ...}
    feedback_positive   int     — feedback rows where score >= 0.5
    feedback_negative   int     — feedback rows where score < 0.5
    recent_generations  list    — last 10 GenerationLog rows as dicts
    recent_feedback     list    — last 10 Feedback rows as dicts
    judge_score_buckets dict    — {"0.0-0.5": N, "0.5-0.75": N, "0.75-1.0": N}
    hourly_generations  list    — last 24 h: [{"hour": "14:00", "count": 5}, ...]
    """

    # ------------------------------------------------------------------
    # 1. Total generations
    # ------------------------------------------------------------------
    total_result = await db.execute(select(func.count()).select_from(GenerationLog))
    total_generations: int = total_result.scalar_one() or 0

    # ------------------------------------------------------------------
    # 2. Success rate
    # ------------------------------------------------------------------
    if total_generations > 0:
        success_result = await db.execute(
            select(func.count()).select_from(GenerationLog).where(
                GenerationLog.status == "success"
            )
        )
        success_count: int = success_result.scalar_one() or 0
        success_rate = round(success_count / total_generations * 100, 1)
    else:
        success_rate = 0.0

    # ------------------------------------------------------------------
    # 3. Average judge score (non-null rows only)
    # ------------------------------------------------------------------
    avg_score_result = await db.execute(
        select(func.avg(GenerationLog.judge_score)).where(
            GenerationLog.judge_score.is_not(None)
        )
    )
    avg_judge_score_raw = avg_score_result.scalar_one()
    avg_judge_score = round(float(avg_judge_score_raw), 3) if avg_judge_score_raw is not None else 0.0

    # ------------------------------------------------------------------
    # 4. Average latency
    # ------------------------------------------------------------------
    avg_latency_result = await db.execute(
        select(func.avg(GenerationLog.latency_ms))
    )
    avg_latency_raw = avg_latency_result.scalar_one()
    avg_latency_ms = round(float(avg_latency_raw), 1) if avg_latency_raw is not None else 0.0

    # ------------------------------------------------------------------
    # 5. P95 latency — fetch all latencies and compute in Python
    #    (SQLite has no built-in PERCENTILE_CONT; this keeps the query
    #     simple and correct without raw SQL)
    # ------------------------------------------------------------------
    p95_latency_ms = 0.0
    if total_generations >= 1:
        latencies_result = await db.execute(
            select(GenerationLog.latency_ms).order_by(GenerationLog.latency_ms)
        )
        latencies = [row[0] for row in latencies_result.all() if row[0] is not None]
        if latencies:
            idx = max(0, int(len(latencies) * 0.95) - 1)
            p95_latency_ms = round(latencies[idx], 1)

    # ------------------------------------------------------------------
    # 6. Viz-type distribution
    # ------------------------------------------------------------------
    vt_result = await db.execute(
        select(GenerationLog.viz_type, func.count().label("cnt"))
        .group_by(GenerationLog.viz_type)
        .order_by(func.count().desc())
    )
    viz_type_distribution: dict[str, int] = {
        row.viz_type: row.cnt for row in vt_result.all() if row.viz_type
    }

    # ------------------------------------------------------------------
    # 7. Feedback counts
    # ------------------------------------------------------------------
    pos_result = await db.execute(
        select(func.count()).select_from(Feedback).where(Feedback.score >= 0.5)
    )
    feedback_positive: int = pos_result.scalar_one() or 0

    neg_result = await db.execute(
        select(func.count()).select_from(Feedback).where(Feedback.score < 0.5)
    )
    feedback_negative: int = neg_result.scalar_one() or 0

    # ------------------------------------------------------------------
    # 8. Recent generations (last 10)
    # ------------------------------------------------------------------
    recent_gen_result = await db.execute(
        select(GenerationLog)
        .order_by(GenerationLog.created_at.desc())
        .limit(10)
    )
    recent_generations = [_row_to_dict(row) for row in recent_gen_result.scalars().all()]

    # ------------------------------------------------------------------
    # 9. Recent feedback (last 10)
    # ------------------------------------------------------------------
    recent_fb_result = await db.execute(
        select(Feedback).order_by(Feedback.timestamp.desc()).limit(10)
    )
    recent_feedback = [_row_to_dict(row) for row in recent_fb_result.scalars().all()]

    # ------------------------------------------------------------------
    # 10. Judge score buckets
    # ------------------------------------------------------------------
    judge_score_buckets: dict[str, int] = {
        "0.0-0.5": 0,
        "0.5-0.75": 0,
        "0.75-1.0": 0,
    }
    if total_generations > 0:
        # Bucket 0.0 – 0.5  (exclusive upper bound)
        b1 = await db.execute(
            select(func.count()).select_from(GenerationLog).where(
                GenerationLog.judge_score.is_not(None),
                GenerationLog.judge_score < 0.5,
            )
        )
        judge_score_buckets["0.0-0.5"] = b1.scalar_one() or 0

        # Bucket 0.5 – 0.75  (exclusive upper bound)
        b2 = await db.execute(
            select(func.count()).select_from(GenerationLog).where(
                GenerationLog.judge_score.is_not(None),
                GenerationLog.judge_score >= 0.5,
                GenerationLog.judge_score < 0.75,
            )
        )
        judge_score_buckets["0.5-0.75"] = b2.scalar_one() or 0

        # Bucket 0.75 – 1.0  (inclusive upper bound)
        b3 = await db.execute(
            select(func.count()).select_from(GenerationLog).where(
                GenerationLog.judge_score.is_not(None),
                GenerationLog.judge_score >= 0.75,
            )
        )
        judge_score_buckets["0.75-1.0"] = b3.scalar_one() or 0

    # ------------------------------------------------------------------
    # 11. Hourly generations — last 24 hours
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC to match DB
    cutoff = now - timedelta(hours=24)

    hourly_result = await db.execute(
        select(GenerationLog.created_at).where(
            GenerationLog.created_at >= cutoff
        )
    )
    hourly_rows = hourly_result.scalars().all()

    # Build a bucket per full hour for the last 24 hours
    hour_counts: dict[str, int] = {}
    for h in range(24):
        bucket_dt = cutoff + timedelta(hours=h)
        label = bucket_dt.strftime("%H:00")
        hour_counts[label] = 0

    for created_at in hourly_rows:
        if created_at is None:
            continue
        # Truncate to hour
        label = created_at.strftime("%H:00")
        if label in hour_counts:
            hour_counts[label] += 1

    hourly_generations = [{"hour": h, "count": c} for h, c in hour_counts.items()]

    return {
        "total_generations": total_generations,
        "success_rate": success_rate,
        "avg_judge_score": avg_judge_score,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "viz_type_distribution": viz_type_distribution,
        "feedback_positive": feedback_positive,
        "feedback_negative": feedback_negative,
        "recent_generations": recent_generations,
        "recent_feedback": recent_feedback,
        "judge_score_buckets": judge_score_buckets,
        "hourly_generations": hourly_generations,
    }
