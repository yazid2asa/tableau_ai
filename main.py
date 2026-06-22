import difflib
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio
import re as _re

from config import settings, SAMPLE_DS_PATTERNS
from database import (
    AsyncSessionLocal, Feedback, GenerationLog, get_db, init_db,
    load_session_memory, save_session_memory, delete_session_memory,
)
from judge import judge_viz, quick_validate, QUICK_VALIDATE_SKIP_THRESHOLD
from logger import log_trace
from monitoring import get_monitoring_metrics, log_generation
from observability import add_generation_span, create_trace, end_trace, flush, init_langfuse, score_trace
from llm import call_llm, check_provider_status, get_active_provider, get_active_model, LLMResponse
from prompts import build_intent_prompt, TOOLS
from schemas import (
    ChatRequest,
    ChatResponse,
    ConversationTurn,
    DataSourceMetadata,
    DownloadRequest,
    FeedbackRequest,
    HealthResponse,
    SessionState,
    VizIntent,
)
from rag_knowledge import (
    get_golden_examples_collection,
    init_rag,
    rebuild_bm25,
    retrieve_examples,
    store_successful_generation,
)
from rag_ingestion import run_ingestion
from twb_generator import (
    add_sheet_to_existing,
    modify_sheet_in_existing,
    generate_multi_sheet_twb,
    generate_twb,
    patch_twb,
    resolve_filter_display_type,
    _normalize,
)
from tableau_server import (
    get_all_datasource_schemas,
    get_datasource_content_url,
    publish_workbook,
    get_view_url,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agentic RAG — skip retrieval for modification turns
# ---------------------------------------------------------------------------

_MODIFY_KEYWORDS = {
    "ajoute", "filtre", "change", "modifie", "trie", "renomme",
    "add", "filter", "change", "sort", "remove", "also", "now",
    "maintenant", "même", "switch", "make it", "include", "exclude",
}


def _is_likely_modify(question: str, has_previous: bool) -> bool:
    """Heuristic: skip RAG if this looks like a chart modification."""
    if not has_previous:
        return False
    q_lower = question.lower()
    return any(kw in q_lower for kw in _MODIFY_KEYWORDS)


def _parse_reasoning_response(raw: str) -> tuple[str, dict]:
    """Parse LLM response with <reasoning> and <output> blocks.

    Returns (reasoning_text, intent_dict).
    Falls back to raw JSON parsing if tags not found.
    """
    if not raw or not raw.strip():
        raise json.JSONDecodeError("Empty LLM response", raw or "", 0)

    reasoning_match = _re.search(r"<reasoning>(.*?)</reasoning>", raw, _re.DOTALL)
    output_match = _re.search(r"<output>(.*?)</output>", raw, _re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if output_match:
        json_str = output_match.group(1).strip()
    else:
        # Fallback: try to find JSON in the raw response
        json_str = raw.strip()
        # Strip markdown fences if present
        if json_str.startswith("```"):
            lines = json_str.splitlines()
            json_str = "\n".join(l for l in lines[1:] if not l.strip().startswith("```"))
        # Try to extract JSON object from surrounding text
        if not json_str.startswith("{"):
            brace_match = _re.search(r"\{.*\}", json_str, _re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)

    intent_data = json.loads(json_str)
    return reasoning, intent_data


async def _fetch_rag_async(question: str, datasource_name: str = "") -> list[dict]:
    """Wrap sync RAG retrieval for asyncio.gather."""
    return await asyncio.to_thread(retrieve_examples, question, datasource_name=datasource_name)


async def _init_rag_background():
    """Heavy RAG init in background — server starts accepting requests immediately."""
    try:
        await asyncio.to_thread(init_rag)
        await run_ingestion(
            None,
            get_golden_examples_collection(),
        )
        await asyncio.to_thread(rebuild_bm25)
        logger.info("RAG initialization complete (background)")
    except Exception as exc:
        logger.warning("RAG background init failed — %s", exc)


def _cleanup_orphan_twb_files(max_age_hours: float = 24.0) -> None:
    """Delete stale uuid-named .twb files left behind by generate_twb.

    The session workbook (`Analyse_*.twb`) is always kept — it is the live,
    accumulating artifact for a session. Only the throwaway uuid-named files
    (the first-generation output we copy into the session workbook) are removed,
    and only once older than max_age_hours so an in-flight request is never hit.
    """
    out = settings.output_dir
    if not out.exists():
        return
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    for p in out.glob("*.twb"):
        if p.name.startswith("Analyse_"):
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info("Startup cleanup: removed %d orphan .twb file(s)", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    settings.output_dir.mkdir(exist_ok=True)
    _cleanup_orphan_twb_files()
    await init_db()
    init_langfuse()
    asyncio.create_task(_init_rag_background())
    yield
    flush()


app = FastAPI(title="Text-to-Viz API", version="1.0.0", lifespan=lifespan)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session-state cache: session_id → SessionState. This is the SINGLE
# source of conversational memory (turns + readable history + cart + summary). It is
# backed by SQLite (table session_memory) so it survives backend reloads — see
# _hydrate_session_state / _persist_session_state below.
_session_states: dict[str, SessionState] = {}

# Per-session locks serialize concurrent requests for the same session_id so two
# in-flight turns can't interleave their reads/writes of the shared SessionState
# (and the accumulated workbook). Different sessions still run fully in parallel.
_session_locks: dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Return the asyncio.Lock for a session, creating it on first use."""
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock

# How many recent turns are sent to the LLM verbatim; older turns are folded into
# SessionState.summary once the session grows past _SUMMARIZE_AFTER turns.
_HISTORY_WINDOW_TURNS = 8
_SUMMARIZE_AFTER = 16


async def _hydrate_session_state(session_id: str) -> SessionState:
    """Return the SessionState for a session, loading it from SQLite if it isn't in
    the in-memory cache (so memory survives an uvicorn --reload mid-conversation)."""
    state = _session_states.get(session_id)
    if state is not None:
        return state
    raw = await load_session_memory(session_id)
    if raw:
        try:
            state = SessionState(**json.loads(raw))
        except Exception as exc:  # corrupt/old blob — start fresh rather than crash
            logger.warning("Could not load session memory for %s: %s", session_id, exc)
            state = SessionState(session_id=session_id)
    else:
        state = SessionState(session_id=session_id)
    _session_states[session_id] = state
    return state


async def _persist_session_state(state: SessionState) -> None:
    """Persist a SessionState to SQLite (excluding the re-fetchable datasource list)."""
    try:
        blob = json.dumps(state.model_dump(exclude={"available_datasources"}), default=str)
        await save_session_memory(state.session_id, blob)
    except Exception as exc:  # never let persistence break a chat turn
        logger.warning("Could not persist session memory for %s: %s", state.session_id, exc)


def _build_history_messages(state: SessionState, window: int = _HISTORY_WINDOW_TURNS) -> list[dict]:
    """Build readable {role, content} messages from the last `window` turns.

    Assistant turns carry the human-readable outcome (assistant_text), never the raw
    JSON intent — so the model's memory of its own replies stays clean and compact.
    """
    msgs: list[dict] = []
    for turn in state.turns[-window:]:
        msgs.append({"role": "user", "content": turn.user_message})
        if turn.assistant_text:
            msgs.append({"role": "assistant", "content": turn.assistant_text})
    return msgs


def _summarize_chart_turn(viz: VizIntent, action: str) -> str:
    """One-line readable record of a chart turn, used as the assistant's memory."""
    verb = "Modified" if action == "modify" else "Created"
    text = f"{verb} a {viz.viz_type.replace('_', ' ')} '{viz.title}'"
    if viz.x_field:
        text += f" ({viz.x_field}" + (f" by {viz.y_field}" if viz.y_field else "") + ")"
    if viz.filters:
        text += " filtered on " + ", ".join(f.field for f in viz.filters if f.field)
    return text


async def _maybe_compact_session(state: SessionState) -> None:
    """When the session grows long, fold the oldest turns into a rolling summary via
    one cheap LLM call, so context stays bounded without losing earlier intent."""
    overflow = len(state.turns) - _HISTORY_WINDOW_TURNS
    if overflow < (_SUMMARIZE_AFTER - _HISTORY_WINDOW_TURNS):
        return
    old_turns = state.turns[:overflow]
    lines = []
    for t in old_turns:
        lines.append(f"User: {t.user_message}")
        if t.assistant_text:
            lines.append(f"Assistant: {t.assistant_text}")
    transcript = "\n".join(lines)
    prompt = [
        {"role": "system", "content": "Summarize this BI chat in 4-6 concise bullet points: what the user asked, concepts explained, and charts created. Keep field/chart names."},
        {"role": "user", "content": (state.summary + "\n" if state.summary else "") + transcript},
    ]
    try:
        resp = await call_llm(prompt)
        if resp.content:
            state.summary = resp.content.strip()
            # drop the summarized turns, keep the recent window
            state.turns = state.turns[overflow:]
    except Exception as exc:
        logger.warning("Session compaction failed (keeping full history): %s", exc)


def _question_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric/accented tokens of a string (for overlap scoring)."""
    return set(_re.findall(r"[a-zA-Z0-9À-ɏ]+", text.lower()))


def _session_prior_luids(session_state) -> set[str]:
    """LUIDs of datasources already used in earlier turns of this session."""
    luids: set[str] = set()
    for turn in getattr(session_state, "turns", []):
        intent = getattr(turn, "resolved_intent", None)
        if intent is not None and intent.datasource_luid:
            luids.add(intent.datasource_luid)
    return luids


def _rank_datasources_by_relevance(
    question: str,
    datasources: list[DataSourceMetadata],
    prior_luids: set[str] | None = None,
) -> list[DataSourceMetadata]:
    """Filter out sample datasources and rank remaining by relevance.

    1. Tokenize the question (lowercase, split on whitespace + punctuation).
    2. For each datasource, count how many field names appear in question tokens
       (case-insensitive partial match) — the primary score.
    3. Exclude datasources whose name matches any SAMPLE_DS_PATTERNS.
    4. Sort by (field match, used-earlier-in-session, name/question overlap)
       descending, so ties break toward a datasource already in play and then
       toward one whose NAME echoes the question; return top 3.
    5. Graceful fallback: if all were filtered out, return original list.
    """
    if not datasources:
        return datasources

    # Tokenize question
    tokens = _re.findall(r"[a-zA-Z0-9\u00C0-\u024F]+", question.lower())

    # Identify sample datasources. Each kept entry is scored on three axes that
    # break ties in priority order: field-name overlap, then whether the
    # datasource was already used earlier in the session, then name/question
    # overlap.
    token_set = set(tokens)
    non_sample: list[tuple[DataSourceMetadata, int, int, int]] = []
    excluded_names: list[str] = []

    for ds in datasources:
        ds_name_lower = ds.datasource_name.lower()
        is_sample = any(pat in ds_name_lower for pat in SAMPLE_DS_PATTERNS)
        if is_sample:
            excluded_names.append(ds.datasource_name)
            continue

        # Count field-name matches against question tokens (primary score)
        field_score = 0
        for field in ds.fields:
            field_lower = field.name.lower()
            for tok in tokens:
                if tok in field_lower or field_lower in tok:
                    field_score += 1
                    break  # count each field at most once

        prior_flag = 1 if (prior_luids and ds.luid and ds.luid in prior_luids) else 0
        name_score = len(token_set & _question_tokens(ds.datasource_name))
        non_sample.append((ds, field_score, prior_flag, name_score))

    if excluded_names:
        logger.info(
            "Datasource guardrails: excluded %d sample datasource(s): %s",
            len(excluded_names), ", ".join(excluded_names),
        )

    # Graceful fallback: if all datasources were samples, return original list
    if not non_sample:
        logger.warning(
            "All %d datasources matched sample patterns — returning unfiltered list",
            len(datasources),
        )
        return datasources

    # Sort by (field match, used-earlier-in-session, name overlap) descending.
    # Python's sort is stable, so equal-on-all-axes datasources keep list order.
    non_sample.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)

    # Return EVERY non-sample datasource (ordered by relevance). The earlier top-3
    # cap meant the LLM never saw datasources 4..N — so a perfectly valid question
    # like "total revenue" against an `orders` datasource would be answered with
    # "I don't see a Revenue field" because `orders` had been ranked lower than
    # trips/target/vehicles and silently dropped. The hard cap of 20 at the
    # `get_all_datasource_schemas` fetch (see tableau_server.py) is the real ceiling.
    return [t[0] for t in non_sample]


def _auto_correct_field(field: str, available_names: list[str]) -> str | None:
    """Return the correct field name if a close match exists, else None."""
    matches = difflib.get_close_matches(field, available_names, n=1, cutoff=0.6)
    return matches[0] if matches else None


def auto_correct_intent_fields(
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
) -> VizIntent:
    """Auto-correct field names in VizIntent to match datasource fields EXACTLY.

    Fixes common LLM mistakes like "Sub Category" → "Sub-Category".
    Uses normalized comparison (ignoring spaces/hyphens/underscores) to find
    the correct exact field name from the datasource.
    Returns a corrected copy of the VizIntent.
    """
    if not metadata or not metadata.fields:
        return viz

    available_names = [f.name for f in metadata.fields]
    # Also include calculated field names
    for cf in (viz.calculated_fields or []):
        available_names.append(cf.name)

    # Build normalized → exact name lookup
    norm_to_exact: dict[str, str] = {}
    for name in available_names:
        norm_to_exact[_normalize(name)] = name

    updates: dict = {}

    for attr in ("x_field", "y_field", "color_field"):
        val = getattr(viz, attr)
        if not val:
            continue
        # Check if exact name already matches
        if val in available_names:
            continue
        # Check if normalized form matches — use the exact datasource name
        norm_val = _normalize(val)
        if norm_val in norm_to_exact:
            exact = norm_to_exact[norm_val]
            if exact != val:
                updates[attr] = exact
        else:
            # No normalized match — try fuzzy matching
            corrected = _auto_correct_field(val, available_names)
            if corrected:
                updates[attr] = corrected

    # Also correct filter field names
    if viz.filters:
        corrected_filters = []
        filters_changed = False
        for fspec in viz.filters:
            if fspec.field and fspec.field not in available_names:
                norm_f = _normalize(fspec.field)
                if norm_f in norm_to_exact:
                    exact_f = norm_to_exact[norm_f]
                    if exact_f != fspec.field:
                        fspec = fspec.model_copy(update={"field": exact_f})
                        filters_changed = True
                else:
                    corrected_f = _auto_correct_field(fspec.field, available_names)
                    if corrected_f:
                        fspec = fspec.model_copy(update={"field": corrected_f})
                        filters_changed = True
            corrected_filters.append(fspec)
        if filters_changed:
            updates["filters"] = corrected_filters

    if updates:
        return viz.model_copy(update=updates)
    return viz


def validate_intent_fields(
    viz: VizIntent,
    metadata: DataSourceMetadata | None,
) -> str | None:
    """
    Returns an error message if fields in viz intent
    do not match available datasource fields.
    Returns None if validation passes.
    Uses normalized comparison to tolerate space/underscore/hyphen differences.
    """
    if not metadata or not metadata.fields:
        return None  # no metadata to validate against

    available = {_normalize(f.name) for f in metadata.fields}
    # Also include calculated field names as available
    for cf in (viz.calculated_fields or []):
        available.add(_normalize(cf.name))

    to_check = [viz.x_field, viz.y_field]
    # Don't validate color_field for combo charts (it's a measure name, not a datasource field)
    if viz.viz_type not in ("combo", "gantt") and viz.color_field:
        to_check.append(viz.color_field)

    for field in to_check:
        if field and _normalize(field) not in available:
            field_names = sorted({f.name for f in metadata.fields})
            close = difflib.get_close_matches(field, field_names, n=1, cutoff=0.5)
            if close:
                return (
                    f"Le champ '{field}' n'existe pas dans votre datasource. "
                    f"Voulez-vous dire '{close[0]}'? "
                    f"Champs disponibles: {field_names}"
                )
            return (
                f"Le champ '{field}' n'existe pas dans votre datasource. "
                f"Champs disponibles: {field_names}"
            )
    return None


_TIME_WORDS = _re.compile(
    r"\b(trend|évolution|evolution|mois|année|annee|over\s+time|monthly|yearly|"
    r"par\s+mois|par\s+année|par\s+annee|weekly|daily|quarterly|trimestre|"
    r"au\s+fil\s+du\s+temps|tendance|historique)\b",
    _re.IGNORECASE,
)


def _find_field_info(field_name: str, datasources: list[DataSourceMetadata], datasource_luid: str | None):
    """Find FieldInfo for a field name in the chosen datasource."""
    if not field_name:
        return None
    # First: search in the chosen datasource
    if datasource_luid:
        for ds in datasources:
            if ds.luid == datasource_luid:
                for f in ds.fields:
                    if f.name.lower() == field_name.lower():
                        return f
    # Fallback: search all datasources
    for ds in datasources:
        for f in ds.fields:
            if f.name.lower() == field_name.lower():
                return f
    return None


def _validate_and_correct_intent(
    viz_intent: VizIntent,
    available_datasources: list[DataSourceMetadata],
    question: str,
) -> VizIntent:
    """Validate and correct LLM-generated VizIntent before TWB generation.

    Applies corrections in order:
    1. Datasource LUID validation
    2. Field name fuzzy matching (delegates to auto_correct_intent_fields)
    3. Field role enforcement (swap x/y if roles are wrong)
    4. Chart type inference check (bar_chart → line_chart for time series)
    5. KPI fix (y_field → x_field)
    """
    if not available_datasources:
        return viz_intent

    updates: dict = {}

    # --- 1. Datasource LUID validation ---
    if viz_intent.datasource_luid:
        luid_found = any(ds.luid == viz_intent.datasource_luid for ds in available_datasources)
        if not luid_found:
            old_luid = viz_intent.datasource_luid
            # Find best match by field overlap with all mentioned fields
            intent_fields = {f.lower() for f in [viz_intent.x_field, viz_intent.y_field, viz_intent.color_field or ""] if f}
            best_ds = None
            best_score = -1
            for ds in available_datasources:
                ds_field_names = {f.name.lower() for f in ds.fields}
                overlap = len(intent_fields & ds_field_names)
                if overlap > best_score:
                    best_score = overlap
                    best_ds = ds
            if best_ds and best_score > 0:
                updates["datasource_luid"] = best_ds.luid
                logger.info("Self-correction: datasource_luid '%s' not found, corrected to '%s'", old_luid, best_ds.luid)
            else:
                updates["datasource_luid"] = None
                logger.info("Self-correction: datasource_luid '%s' not found, cleared (no match)", old_luid)

    if updates:
        viz_intent = viz_intent.model_copy(update=updates)
        updates = {}

    # --- 2. Field name fuzzy matching (delegate to existing function) ---
    # Find the effective datasource metadata for auto_correct_intent_fields.
    # When blending, MERGE primary + secondary fields — otherwise a
    # secondary-only field (e.g. vehicle_type in vehicles while primary is
    # trips) gets fuzzy-corrected to the closest primary field (vehicle_id)
    # and the rest of the pipeline never sees it again. FIX-043.
    effective_ds = None
    if viz_intent.datasource_luid:
        for ds in available_datasources:
            if ds.luid == viz_intent.datasource_luid:
                effective_ds = ds
                break
    if not effective_ds and available_datasources:
        effective_ds = available_datasources[0]

    if effective_ds and viz_intent.secondary_datasource_luid:
        secondary_ds = next(
            (ds for ds in available_datasources if ds.luid == viz_intent.secondary_datasource_luid),
            None,
        )
        if secondary_ds:
            primary_names_lower = {f.name.lower() for f in effective_ds.fields}
            merged_fields = list(effective_ds.fields)
            for f in secondary_ds.fields:
                if f.name.lower() not in primary_names_lower:
                    merged_fields.append(f)
            effective_ds = effective_ds.model_copy(update={"fields": merged_fields})

    if effective_ds:
        viz_intent = auto_correct_intent_fields(viz_intent, effective_ds)

    # --- 3. Field role enforcement ---
    if viz_intent.x_field and viz_intent.y_field:
        x_info = _find_field_info(viz_intent.x_field, available_datasources, viz_intent.datasource_luid)
        y_info = _find_field_info(viz_intent.y_field, available_datasources, viz_intent.datasource_luid)

        if x_info and y_info:
            x_is_measure = x_info.role == "measure"
            y_is_dimension = y_info.role == "dimension"
            # Both measures is valid (scatter plot) — don't swap
            # Swap only when x is measure AND y is dimension
            if x_is_measure and y_is_dimension:
                logger.info("Self-correction: swapped x_field/y_field (role mismatch)")
                updates["x_field"] = viz_intent.y_field
                updates["y_field"] = viz_intent.x_field

    if updates:
        viz_intent = viz_intent.model_copy(update=updates)
        updates = {}

    # --- 4. Chart type inference check ---
    if viz_intent.viz_type == "bar_chart" and viz_intent.x_field:
        x_info = _find_field_info(viz_intent.x_field, available_datasources, viz_intent.datasource_luid)
        is_date = x_info and x_info.type.value in ("date", "datetime") if x_info else False
        has_time_words = bool(_TIME_WORDS.search(question))
        if is_date and has_time_words:
            logger.info("Self-correction: viz_type bar_chart → line_chart (time series detected)")
            updates["viz_type"] = "line_chart"

    if updates:
        viz_intent = viz_intent.model_copy(update=updates)
        updates = {}

    # --- 5. KPI fix ---
    if viz_intent.viz_type == "kpi" and viz_intent.y_field:
        logger.info("Self-correction: kpi — moved y_field '%s' to x_field", viz_intent.y_field)
        updates["x_field"] = viz_intent.y_field
        updates["y_field"] = ""

    if updates:
        viz_intent = viz_intent.model_copy(update=updates)

    return viz_intent


def _normalize_filter_set(filters) -> tuple:
    """Order-independent, hashable representation of a filter list — UI-only
    fields (display_type, show_filter_card) are excluded so they never trigger
    a new sheet."""
    if not filters:
        return ()
    items = sorted(
        json.dumps(
            f.model_dump(exclude_none=True, exclude={"display_type", "show_filter_card"}),
            sort_keys=True, default=str,
        )
        for f in filters
    )
    return tuple(items)


def _is_structural_change(new: VizIntent, prev: VizIntent) -> bool:
    """True when a follow-up should become a NEW sheet rather than an in-place tweak.

    Triggers a new sheet:
      - x/y/color changes (added/swapped dimension or measure)
      - viz_type changes (different chart type = different analytical perspective)
      - filters added/removed/changed (different data slice = different analytical view)

    Stays in-place (sort, aggregation, title-only tweaks) leave both the data
    slice and the chart type identical, so they update the current sheet.
    Comparison is normalized to tolerate caption/physical spelling differences.
    """
    if (
        _normalize(new.x_field or "") != _normalize(prev.x_field or "")
        or _normalize(new.y_field or "") != _normalize(prev.y_field or "")
        or _normalize(new.color_field or "") != _normalize(prev.color_field or "")
    ):
        return True
    if (new.viz_type or "") != (prev.viz_type or ""):
        return True
    if _normalize_filter_set(new.filters) != _normalize_filter_set(prev.filters):
        return True
    return False


def _auto_select_best_datasource(
    viz_intent: VizIntent,
    datasources: list[DataSourceMetadata],
    prior_luids: set[str] | None = None,
    question: str = "",
) -> DataSourceMetadata:
    """Auto-select the best datasource by matching viz intent fields to datasource fields.

    Primary score is field overlap between the intent and the datasource. Ties
    (including the all-zero-overlap case) break toward a datasource already used
    earlier in the session, then toward one whose name echoes the question, then
    by list order (stable).
    """
    intent_fields = set()
    for f in [viz_intent.x_field, viz_intent.y_field, viz_intent.color_field]:
        if f:
            intent_fields.add(_normalize(f))
    for fspec in (viz_intent.filters or []):
        if fspec.field:
            intent_fields.add(_normalize(fspec.field))

    q_tokens = _question_tokens(question) if question else set()

    best_ds = datasources[0]
    best_key = (-1, -1, -1)
    for ds in datasources:
        ds_fields_norm = {_normalize(f.name) for f in ds.fields}
        overlap = len(intent_fields & ds_fields_norm)
        prior_flag = 1 if (prior_luids and ds.luid and ds.luid in prior_luids) else 0
        name_score = len(q_tokens & _question_tokens(ds.datasource_name)) if q_tokens else 0
        key = (overlap, prior_flag, name_score)
        if key > best_key:
            best_key = key
            best_ds = ds
    return best_ds


def _rescue_intent_across_datasources(
    viz_intent: VizIntent,
    all_datasources: list[DataSourceMetadata] | None,
) -> VizIntent:
    """Last-resort fix when the LLM picked a datasource that doesn't contain all of
    the intent's x/y/color fields. We scan every available datasource:

      1. If a single datasource holds all of them → switch ``datasource_luid``.
      2. Otherwise, if two datasources together cover them and share at least one
         common field → set ``datasource_luid`` + ``secondary_datasource_luid`` so
         the generator blends them on the common key.

    If neither pass succeeds, the intent is returned unchanged and downstream
    validation will surface a clear "field doesn't exist" message.
    """
    if not all_datasources:
        return viz_intent

    intent_fields = {
        _normalize(f) for f in [viz_intent.x_field, viz_intent.y_field, viz_intent.color_field] if f
    }
    if not intent_fields:
        return viz_intent

    # Build a (normalized field set) per datasource once
    ds_fields: list[tuple[DataSourceMetadata, set[str]]] = [
        (ds, {_normalize(f.name) for f in ds.fields}) for ds in all_datasources
    ]

    # Is the current choice already complete?
    current = next(
        (fs for ds, fs in ds_fields if ds.luid == viz_intent.datasource_luid),
        None,
    )
    if current is not None and intent_fields.issubset(current):
        return viz_intent

    # Pass 1 — single-datasource solution
    for ds, fs in ds_fields:
        if intent_fields.issubset(fs):
            logger.info(
                "Cross-source rescue: switched datasource_luid '%s' → '%s' (%s) — has all intent fields",
                viz_intent.datasource_luid, ds.luid, ds.datasource_name,
            )
            return viz_intent.model_copy(update={"datasource_luid": ds.luid})

    # Pass 2 — two-datasource blend
    for primary, p_fields in ds_fields:
        covered = intent_fields & p_fields
        missing = intent_fields - p_fields
        if not covered or not missing:
            continue
        for secondary, s_fields in ds_fields:
            if secondary.luid == primary.luid:
                continue
            if not missing.issubset(s_fields):
                continue
            # Need at least one common field to act as the blend link
            if p_fields & s_fields:
                logger.info(
                    "Cross-source rescue: blending '%s' + '%s' (shared key present)",
                    primary.datasource_name, secondary.datasource_name,
                )
                return viz_intent.model_copy(update={
                    "datasource_luid": primary.luid,
                    "secondary_datasource_luid": secondary.luid,
                })

    return viz_intent  # No rescue possible — caller will return the friendly error


def _make_suggestion(question: str, error_type: str, metadata=None) -> str:
    """Generate a reformulation suggestion based on the error type."""
    if error_type == "json_decode":
        return "Try asking more specifically, e.g. 'Show a bar chart of Sales by Category'"
    if error_type == "viz_type":
        return "Supported types: bar chart, line chart, pie, scatter, area, heatmap, treemap"
    if error_type == "field_mismatch" and metadata and metadata.fields:
        field_names = [f.name for f in metadata.fields]
        # Extract potential field references from the question (words > 3 chars)
        words = [w.strip("?.,!") for w in question.split() if len(w) > 3]
        for word in words:
            matches = difflib.get_close_matches(word, field_names, n=1, cutoff=0.6)
            if matches:
                return f"Did you mean the field '{matches[0]}'? Available fields: {', '.join(field_names[:5])}"
        return f"Available fields: {', '.join(field_names[:8])}"
    return "Try rephrasing your question with explicit field names and chart type"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        provider=get_active_provider(),
        openrouter_status=await check_provider_status(),
        model_id=get_active_model(),
        server_mode=True,
    )


# ---------------------------------------------------------------------------
# Server datasources — list available Tableau Server datasources
# ---------------------------------------------------------------------------

@app.get("/server/datasources")
async def server_datasources():
    """Return list of available datasources from Tableau Server (name + luid)."""
    schemas = await get_all_datasource_schemas()
    return {
        "datasources": [{"name": ds.datasource_name, "luid": ds.luid} for ds in schemas],
        "server_mode": True,
    }


@app.get("/server/test-publish")
async def test_publish():
    """Diagnostic: verify project, then publish a real .twb via generate_twb().

    Tests auth + project + publish independently of the chat flow.
    Uses TSC library for all Tableau API calls.
    """
    from tableau_server import signin

    results = {"auth": None, "project": None, "publish": None,
               "project_id": settings.tableau_default_project_id}

    # Step 1: Test auth
    try:
        token, site_id = await signin()
        results["auth"] = {"status": "ok", "site_id": site_id, "token_prefix": token[:12] + "..."}
    except Exception as e:
        results["auth"] = {"status": "error", "detail": str(e)}
        return results

    # Step 2: Verify project (via TSC resolve)
    try:
        import asyncio
        from tableau_server import _resolve_project_luid_sync, _ensure_signed_in
        resolved = await asyncio.to_thread(_resolve_project_luid_sync, settings.tableau_default_project_id)
        results["project"] = {"status": "ok", "id": resolved}
    except Exception as e:
        results["project"] = {"status": "error", "detail": str(e)}

    # Step 3: Fetch datasources + pick the first one for test publish
    schemas = []
    try:
        schemas = await get_all_datasource_schemas()
        results["datasources"] = [{"name": ds.datasource_name, "luid": ds.luid} for ds in schemas[:5]]
    except Exception as e:
        results["datasources"] = {"status": "error", "detail": str(e)}

    if not schemas:
        results["publish"] = {"status": "error", "detail": "No published datasources found — cannot test publish"}
        return results

    # Step 4: Generate a .twb wired to a REAL published datasource via sqlproxy
    test_ds = schemas[0]  # use first available datasource
    results["test_datasource"] = {"name": test_ds.datasource_name, "luid": test_ds.luid}

    # Get contentUrl for sqlproxy wiring
    try:
        ds_content_url = await get_datasource_content_url(test_ds.luid)
        results["test_datasource"]["content_url"] = ds_content_url
    except Exception as e:
        results["publish"] = {"status": "error", "detail": f"Could not get datasource contentUrl: {e}"}
        return results

    # Use real fields from the published datasource
    from schemas import VizIntent

    # Pick first dimension and first measure from the datasource
    dims = [f for f in test_ds.fields if f.role == "dimension"]
    measures = [f for f in test_ds.fields if f.role == "measure"]
    x_field = dims[0].name if dims else (test_ds.fields[0].name if test_ds.fields else "Field1")
    y_field = measures[0].name if measures else (test_ds.fields[-1].name if len(test_ds.fields) > 1 else "Field2")

    test_viz = VizIntent(
        viz_type="bar_chart",
        title="Test Publish",
        x_field=x_field,
        y_field=y_field,
        action="new",
        datasource_luid=test_ds.luid,
    )

    twb_path = None
    try:
        # Generate .twb wired to the real published datasource
        filename, twb_path = generate_twb(
            test_viz, test_ds,
            server_ds_content_url=ds_content_url,
            server_ds_name=test_ds.datasource_name,
        )
        results["twb"] = {"status": "ok", "filename": filename, "size": twb_path.stat().st_size}

        # Publish without overwrite for first-time test
        luid = await publish_workbook(str(twb_path), settings.tableau_default_project_id, overwrite=False)
        results["publish"] = {"status": "ok", "workbook_luid": luid}

        try:
            view_url = await get_view_url(luid)
            results["publish"]["view_url"] = view_url
        except Exception as e:
            results["publish"]["view_url_error"] = str(e)
    except Exception as e:
        results["publish"] = {"status": "error", "detail": str(e)}
    finally:
        if twb_path:
            for p in [str(twb_path), str(twb_path).replace(".twb", ".twbx")]:
                try:
                    os.remove(p)
                except OSError:
                    pass

    return results


# ---------------------------------------------------------------------------
# Chat — main generation endpoint
# ---------------------------------------------------------------------------

async def _revise_intent_with_feedback(
    viz_intent: VizIntent,
    judge_feedback: str,
    question: str,
    metadata: DataSourceMetadata | None,
) -> VizIntent | None:
    """Ask the LLM to revise a low-scoring intent using the judge's feedback.

    Returns the revised VizIntent, or None if the model didn't return a usable
    tool call. The caller re-judges and adopts the revision only when it scores
    at least as well as the original (regression guard), so a worse revision is
    discarded rather than shipped.
    """
    field_list = (
        ", ".join(f.name for f in metadata.fields)
        if metadata and metadata.fields else "unknown"
    )
    messages = [
        {"role": "system", "content": (
            "You are refining a Tableau visualization intent that a quality judge "
            "rated as inadequate. Call the generate_chart tool to return a corrected "
            "intent. Keep the same datasource; change only what the feedback calls out. "
            f"Available fields: {field_list}."
        )},
        {"role": "user", "content": (
            f"Original question: {question}\n\n"
            f"Rejected intent: {json.dumps(viz_intent.model_dump())}\n\n"
            f"Judge feedback: {judge_feedback}\n\n"
            "Return an improved generate_chart tool call."
        )},
    ]
    try:
        resp = await call_llm(messages, tools=TOOLS)
    except Exception as exc:
        logger.warning("Self-correction LLM call failed: %s", exc)
        return None
    if not resp.has_tool_call:
        return None
    try:
        revised = VizIntent(**resp.first_tool_args)
    except Exception as exc:
        logger.warning("Self-correction returned an invalid intent: %s", exc)
        return None
    # Preserve the datasource choice if the revision dropped it.
    if not revised.datasource_luid and viz_intent.datasource_luid:
        revised = revised.model_copy(update={"datasource_luid": viz_intent.datasource_luid})
    return auto_correct_intent_fields(revised, metadata)


async def _run_chat_pipeline(req: ChatRequest):
    """Single source of truth for a chat turn — shared by /chat and /chat/stream.

    Yields event dicts so each caller can render however it likes:
      {"type": "status", "step", "message"}                 progress (ignored by /chat)
      {"type": "intent", "viz_intent"}                      resolved intent (ignored by /chat)
      {"type": "error",  "code", "message", "suggestion"}   terminal failure
      {"type": "result", "response": ChatResponse}          terminal success/clarification

    All side effects (session memory, cart, logging, RAG store, publish) run
    BEFORE the terminal result/error event, because consumers stop iterating as
    soon as they see it. Concurrent turns for the same session are serialized by
    a per-session lock so they can't interleave reads/writes of SessionState.
    """
    async with _get_session_lock(req.session_id):
        start_ms = time.monotonic() * 1000
        trace_id = uuid.uuid4().hex  # 32 hex chars, no hyphens — required by LangFuse v4
        # Conversational memory — hydrated from SQLite so it survives backend reloads
        session_state = await _hydrate_session_state(req.session_id)
        previous_intent = session_state.last_intent
        history = _build_history_messages(session_state)

        lf_trace = create_trace(
            trace_id=trace_id,
            session_id=req.session_id,
            question=req.question,
            metadata={"has_metadata": req.metadata is not None},
        )

        yield {"type": "status", "step": "analyzing", "message": "Analyse de votre question..."}

        # Agentic RAG: skip retrieval for modification turns
        skip_rag = _is_likely_modify(req.question, previous_intent is not None)
        ds_name = req.metadata.datasource_name if req.metadata else ""

        # Parallel fetch: datasource schemas + RAG retrieval
        if not session_state.available_datasources:
            # Tableau Cloud's Metadata API occasionally returns 5xx ("down for
            # maintenance"), and TSC can also throw on auth / transport blips.
            # Don't let those crash the SSE stream — yield a friendly error and
            # leave session_state.available_datasources empty so the next turn
            # retries automatically.
            try:
                if not skip_rag and settings.rag_enabled:
                    schemas, rag_examples = await asyncio.gather(
                        get_all_datasource_schemas(),
                        _fetch_rag_async(req.question, ds_name),
                    )
                else:
                    schemas = await get_all_datasource_schemas()
                    rag_examples = []
                    if skip_rag:
                        logger.info("Agentic RAG: skipping retrieval (modification turn detected)")
            except Exception as exc:
                logger.exception("Tableau Metadata API fetch failed")
                yield {"type": "error", "code": 503,
                       "message": "Tableau Server est temporairement indisponible (Metadata API). Réessayez dans quelques instants.",
                       "suggestion": f"Detail: {type(exc).__name__}: {str(exc)[:200]}"}
                return
            session_state.available_datasources = schemas
        else:
            if not skip_rag and settings.rag_enabled:
                try:
                    rag_examples = await _fetch_rag_async(req.question, ds_name)
                except Exception as exc:
                    logger.warning("RAG retrieval failed (continuing without examples): %s", exc)
                    rag_examples = []
            else:
                rag_examples = []
                if skip_rag:
                    logger.info("Agentic RAG: skipping retrieval (modification turn detected)")

        available_datasources = session_state.available_datasources

        # Filter and rank datasources before passing to LLM. Ties break toward a
        # datasource already used earlier in this session, then name/question overlap.
        prior_luids = _session_prior_luids(session_state)
        if available_datasources:
            available_datasources = _rank_datasources_by_relevance(
                req.question, available_datasources, prior_luids=prior_luids,
            )

        yield {"type": "status", "step": "context",
               "message": f"Contexte trouvé : {len(rag_examples)} exemples similaires"}

        # LLM Call: conversational agent with generate_chart tool
        messages = build_intent_prompt(
            req.question, req.metadata, history, previous_intent,
            rag_knowledge=None, rag_examples=rag_examples,
            available_datasources=available_datasources,
            session_summary=session_state.summary,
            charts_so_far=[f"{v.title} ({v.viz_type})" for v in session_state.cart],
        )

        yield {"type": "status", "step": "reasoning", "message": "Raisonnement en cours..."}

        try:
            llm_response = await call_llm(messages, tools=TOOLS)
        except ValueError as exc:
            yield {"type": "error", "code": 502, "message": str(exc),
                   "suggestion": "Check that your API key is valid and the model is available."}
            return

        # Branch: conversation (text-only) vs chart generation (tool call)
        if not llm_response.has_tool_call:
            text = llm_response.content or ""
            session_state.turns.append(ConversationTurn(
                user_message=req.question, kind="answer", assistant_text=text,
            ))
            await _persist_session_state(session_state)
            end_trace(lf_trace)
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=text,
                mode="conversation",
            )}
            return

        intent_data = llm_response.first_tool_args
        reasoning_text = llm_response.content or ""

        add_generation_span(
            lf_trace, "llm_call", {"messages": messages},
            {"response": json.dumps(intent_data), "reasoning": reasoning_text},
            model=settings.model_id,
        )

        if reasoning_text:
            preview = reasoning_text.split("\n")[0][:100]
            yield {"type": "status", "step": "intent", "message": f"Intention détectée : {preview}"}

        try:
            viz_intent = VizIntent(**intent_data)
        except Exception as exc:
            yield {"type": "error", "code": 422,
                   "message": f"Invalid viz intent structure: {exc}",
                   "suggestion": _make_suggestion(req.question, "viz_type", req.metadata)}
            return

        # Auto-correct field names (e.g. "Sub Category" → "Sub_Category").
        # Skip when the LLM set a secondary datasource — req.metadata reflects
        # only the Extension's currently-open worksheet, so it doesn't know
        # about the secondary's fields and would fuzzy-correct vehicle_type
        # back to vehicle_id. The Server-aware pass inside
        # _validate_and_correct_intent (which uses available_datasources +
        # merges the secondary) handles the correction correctly. FIX-043.
        if not viz_intent.secondary_datasource_luid:
            viz_intent = auto_correct_intent_fields(viz_intent, req.metadata)

        # Agent self-correction: validate and fix LLM output
        viz_intent = _validate_and_correct_intent(viz_intent, available_datasources, req.question)

        # Cross-source rescue: if the LLM chose a datasource that doesn't contain
        # every x/y/color field, switch to one that does — or set up a blend across
        # two complementary datasources. This is the single biggest source of
        # "field doesn't exist in your datasource" errors on multi-source workbooks.
        all_session_ds = session_state.available_datasources or available_datasources or []
        viz_intent = _rescue_intent_across_datasources(viz_intent, all_session_ds)

        # Resolve filter display types for filter cards
        if viz_intent.filters and req.metadata:
            for fspec in viz_intent.filters:
                resolve_filter_display_type(fspec, req.metadata.model_dump() if hasattr(req.metadata, 'model_dump') else req.metadata)

        # Handle clarification flow
        if viz_intent.clarification_needed:
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=viz_intent.clarification_needed,
                mode="clarification",
                clarification_needed=viz_intent.clarification_needed,
            )}
            return

        # Field validation against datasource metadata. In Server mode (where we
        # have Server schemas) we skip the Extension-metadata check — req.metadata
        # reflects the currently-open worksheet, not the chosen Server datasource,
        # so it would reject perfectly valid cross-source intents. The authoritative
        # check runs below against effective_metadata (the chosen Server schema).
        field_error = None if all_session_ds else validate_intent_fields(viz_intent, req.metadata)
        if field_error:
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=field_error,
                warning=field_error,
                mode="clarification",
                clarification_needed=field_error,
            )}
            return

        yield {"type": "intent", "viz_intent": viz_intent.model_dump()}
        yield {"type": "status", "step": "generating", "message": f"Génération du {viz_intent.viz_type}..."}

        # Determine action
        action = viz_intent.action or "new"

        # "Modify intelligent": keep 'modify' (in-place tweak on the current sheet) only for
        # sort / aggregation / title-only tweaks. Anything that changes the data slice
        # (filters), the chart type (viz_type), or the structural fields (x/y/color) becomes
        # a NEW sheet so the previous chart is never overwritten — the user sees both the
        # original perspective and the new one side by side.
        if action == "modify":
            _prev_intent = session_state.last_intent
            if _prev_intent is None or _is_structural_change(viz_intent, _prev_intent):
                action = "new"
                logger.info("Action modify→new (structural change / no previous chart): adding a new sheet")

        # Resolve Server datasource
        server_ds_content_url = None
        server_ds_name = None
        blend_secondary_content_url = None
        blend_secondary_name = None
        blend_linking_fields = None
        blend_secondary_metadata: DataSourceMetadata | None = None

        # Auto-select datasource if LLM didn't pick one but datasources are available
        if not viz_intent.datasource_luid and available_datasources:
            # Use ALL session datasources (not just ranked top 3) for best field match
            all_ds = session_state.available_datasources or available_datasources
            best = _auto_select_best_datasource(
                viz_intent, all_ds, prior_luids=prior_luids, question=req.question,
            )
            viz_intent = viz_intent.model_copy(update={"datasource_luid": best.luid})
            logger.info("Auto-selected datasource '%s' by field match (LLM returned no datasource_luid)", best.datasource_name)

        if viz_intent.datasource_luid:
            try:
                server_ds_content_url = await get_datasource_content_url(viz_intent.datasource_luid)
            except Exception as exc:
                logger.exception("get_datasource_content_url failed for primary luid=%s", viz_intent.datasource_luid)
                yield {"type": "error", "code": 503,
                       "message": "Tableau Server est temporairement indisponible. Réessayez dans quelques instants.",
                       "suggestion": f"Detail: {type(exc).__name__}: {str(exc)[:200]}"}
                return
            # Find datasource name from cached schemas
            all_ds = session_state.available_datasources or available_datasources
            for ds in (all_ds or []):
                if ds.luid == viz_intent.datasource_luid:
                    server_ds_name = ds.datasource_name
                    break

            # Handle secondary datasource for blending
            if viz_intent.secondary_datasource_luid:
                try:
                    blend_secondary_content_url = await get_datasource_content_url(viz_intent.secondary_datasource_luid)
                except Exception as exc:
                    logger.exception("get_datasource_content_url failed for secondary luid=%s", viz_intent.secondary_datasource_luid)
                    yield {"type": "error", "code": 503,
                           "message": "Tableau Server est temporairement indisponible. Réessayez dans quelques instants.",
                           "suggestion": f"Detail: {type(exc).__name__}: {str(exc)[:200]}"}
                    return
                for ds in (available_datasources or []):
                    if ds.luid == viz_intent.secondary_datasource_luid:
                        blend_secondary_name = ds.datasource_name
                        break
                # Detect linking fields
                from twb_generator import _detect_linking_fields
                primary_schema = next((ds for ds in available_datasources if ds.luid == viz_intent.datasource_luid), None)
                secondary_schema = next((ds for ds in available_datasources if ds.luid == viz_intent.secondary_datasource_luid), None)
                if primary_schema and secondary_schema:
                    blend_linking_fields = _detect_linking_fields(primary_schema, secondary_schema)
                    blend_secondary_metadata = secondary_schema
                    if not blend_linking_fields:
                        yield {"type": "result", "response": ChatResponse(
                            session_id=req.session_id,
                            trace_id=trace_id,
                            mode="clarification",
                            clarification_needed=f"Cannot blend datasources '{server_ds_name}' and '{blend_secondary_name}': no common fields found. Which field should link them?",
                            message="Clarification needed for blending.",
                        )}
                        return

        # Use selected datasource schema as effective metadata for validation.
        # When blending, merge primary + secondary fields so validation and
        # twilize's field_registry see every available field — otherwise a
        # secondary-only field (e.g. vehicle_type in the vehicles DS while
        # primary is trips) gets fuzzy-corrected to the closest primary field
        # (vehicle_id) and the chart renders on the wrong dimension.
        effective_metadata = req.metadata
        if viz_intent.datasource_luid and available_datasources:
            for ds in available_datasources:
                if ds.luid == viz_intent.datasource_luid:
                    effective_metadata = ds
                    break
            if blend_secondary_metadata and effective_metadata is not None:
                primary_field_names_lower = {f.name.lower() for f in effective_metadata.fields}
                merged_fields = list(effective_metadata.fields)
                for f in blend_secondary_metadata.fields:
                    if f.name.lower() not in primary_field_names_lower:
                        merged_fields.append(f)
                effective_metadata = effective_metadata.model_copy(update={"fields": merged_fields})

        # Re-run field validation with effective metadata (Server schema)
        if effective_metadata and effective_metadata != req.metadata:
            viz_intent = auto_correct_intent_fields(viz_intent, effective_metadata)
            field_error = validate_intent_fields(viz_intent, effective_metadata)
            if field_error:
                yield {"type": "result", "response": ChatResponse(
                    session_id=req.session_id,
                    trace_id=trace_id,
                    message=field_error,
                    warning=field_error,
                    mode="clarification",
                    clarification_needed=field_error,
                )}
                return

        # FIX-044: detect calc fields that reference secondary-datasource-only fields.
        # effective_metadata is the merged primary+secondary schema at this point, so
        # we re-fetch the pure primary from available_datasources for the comparison.
        # Tableau blend gives only aggregate access to secondary fields; row-level
        # formulas mixing fields from two separate datasources are unsupported.
        if blend_secondary_metadata and viz_intent.calculated_fields:
            from twb_generator import _detect_cross_datasource_calc_fields
            pure_primary = next(
                (ds for ds in (available_datasources or []) if ds.luid == viz_intent.datasource_luid),
                req.metadata,
            )
            bad_calc_names = _detect_cross_datasource_calc_fields(
                viz_intent, pure_primary, blend_secondary_metadata,
            )
            if bad_calc_names:
                names_str = ", ".join(f'"{n}"' for n in bad_calc_names)
                sec = blend_secondary_name or "secondary datasource"
                prim = server_ds_name or "primary datasource"
                msg = (
                    f"Cannot compute {names_str}: this metric requires fields from both "
                    f'"{prim}" and "{sec}" at row level. Tableau blending only provides '
                    f"aggregated access to the secondary datasource — row-level formulas "
                    f"mixing fields from two separate datasources are not supported in Tableau. "
                    f"Options: (1) ask your admin to create a pre-joined datasource that "
                    f"combines both tables; (2) I can display each metric as a separate KPI instead."
                )
                yield {"type": "result", "response": ChatResponse(
                    session_id=req.session_id,
                    trace_id=trace_id,
                    mode="clarification",
                    clarification_needed=msg,
                    message=msg,
                )}
                return

        # Generate or update .twb file
        mode = "new_workbook"
        inplace_modify = False
        try:
            blend_kwargs = {
                "blend_secondary_content_url": blend_secondary_content_url,
                "blend_secondary_name": blend_secondary_name,
                "blend_linking_fields": blend_linking_fields,
                "blend_secondary_metadata": blend_secondary_metadata,
            }
            if action == "modify" and session_state.session_workbook_path and Path(session_state.session_workbook_path).exists():
                # In-place tweak: replace ONLY the current sheet, preserving all others
                session_wb = session_state.session_workbook_path
                old_title = session_state.last_intent.title if session_state.last_intent else None
                modify_sheet_in_existing(
                    session_wb, viz_intent, effective_metadata,
                    old_title=old_title,
                    server_ds_content_url=server_ds_content_url,
                    server_ds_name=server_ds_name,
                    **blend_kwargs,
                )
                filename = Path(session_wb).name
                mode = "sheet_added"
                inplace_modify = True
            elif session_state.session_workbook_path and Path(session_state.session_workbook_path).exists():
                # Accumulate: add a new sheet to the session workbook (never overwrites)
                existing_path = session_state.session_workbook_path
                add_sheet_to_existing(existing_path, viz_intent, effective_metadata,
                                     server_ds_content_url=server_ds_content_url,
                                     server_ds_name=server_ds_name,
                                     **blend_kwargs)
                filename = Path(existing_path).name
                mode = "sheet_added"
            elif req.workbook_name:
                existing = settings.output_dir / f"{req.workbook_name}.twb"
                if existing.exists():
                    add_sheet_to_existing(str(existing), viz_intent, effective_metadata,
                                         server_ds_content_url=server_ds_content_url,
                                         server_ds_name=server_ds_name,
                                         **blend_kwargs)
                    filename = existing.name
                    mode = "sheet_added"
                else:
                    filename, _ = generate_twb(
                        viz_intent, effective_metadata,
                        server_ds_content_url=server_ds_content_url,
                        server_ds_name=server_ds_name,
                        **blend_kwargs,
                    )
            else:
                filename, _ = generate_twb(
                    viz_intent, effective_metadata,
                    server_ds_content_url=server_ds_content_url,
                    server_ds_name=server_ds_name,
                    **blend_kwargs,
                )
        except ValueError as exc:
            yield {"type": "error", "code": 400, "message": str(exc),
                   "suggestion": _make_suggestion(req.question, "field_mismatch", effective_metadata)}
            return

        # LLM-as-a-Judge validation (skip for sheet_added — TWB already existed)
        judge_score: float | None = None
        judge_feedback: str | None = None
        warning: str | None = None
        twb_full_path = str(settings.output_dir / filename)

        if mode != "sheet_added":
            yield {"type": "status", "step": "validating", "message": "Validation qualité..."}
            # Hybrid judge: Python quick_validate first, LLM judge only if needed
            quick_score, quick_feedback = quick_validate(viz_intent, effective_metadata, req.question)
            if quick_score >= QUICK_VALIDATE_SKIP_THRESHOLD:
                judge_score = quick_score
                judge_feedback = quick_feedback
                logger.info("Hybrid judge: quick_validate passed (%.2f) — skipping LLM judge", quick_score)
            else:
                judge_score, judge_feedback = await judge_viz(
                    viz_intent, twb_full_path, req.question, effective_metadata,
                    provider_override=settings.judge_provider,
                    model_override=settings.judge_model_id,
                )
                # Real self-correction: feed the judge feedback back to the LLM once,
                # regenerate, re-judge, and adopt the revision only if it scores at
                # least as well as the original (regression guard).
                if judge_score < settings.judge_threshold and settings.judge_max_retries > 0:
                    revised = await _revise_intent_with_feedback(
                        viz_intent, judge_feedback or "", req.question, effective_metadata,
                    )
                    if revised is not None:
                        try:
                            new_filename, _ = generate_twb(
                                revised, effective_metadata,
                                server_ds_content_url=server_ds_content_url,
                                server_ds_name=server_ds_name,
                                **blend_kwargs,
                            )
                            new_twb_path = str(settings.output_dir / new_filename)
                            new_score, new_feedback = await judge_viz(
                                revised, new_twb_path, req.question, effective_metadata,
                                provider_override=settings.judge_provider,
                                model_override=settings.judge_model_id,
                            )
                            if new_score >= judge_score:
                                logger.info("Self-correction improved score %.2f → %.2f", judge_score, new_score)
                                _superseded = twb_full_path
                                viz_intent = revised
                                filename = new_filename
                                twb_full_path = new_twb_path
                                judge_score = new_score
                                judge_feedback = new_feedback
                            else:
                                logger.info("Self-correction did not improve (%.2f < %.2f) — keeping original", new_score, judge_score)
                                _superseded = new_twb_path
                            # Drop whichever .twb we are not keeping, so the failed
                            # attempt doesn't linger as an orphan.
                            if _superseded != twb_full_path:
                                try:
                                    os.remove(_superseded)
                                except OSError:
                                    pass
                        except ValueError:
                            pass

            add_generation_span(
                lf_trace, "judge_validation",
                {"viz_intent": viz_intent.model_dump()},
                {"judge_score": judge_score, "judge_feedback": judge_feedback},
            )
            if judge_score is not None:
                score_trace(lf_trace, judge_score, judge_feedback or "")
            if judge_score is not None and judge_score < settings.judge_threshold:
                warning = f"Qualité partielle — quality score: {judge_score:.2f}"

            # RAG: store high-scoring generations as few-shot examples (off the event loop)
            if judge_score is not None:
                await asyncio.to_thread(
                    store_successful_generation,
                    question=req.question,
                    viz_intent_dict=viz_intent.model_dump(),
                    judge_score=judge_score,
                    trace_id=trace_id,
                    datasource_name=effective_metadata.datasource_name if effective_metadata else "",
                )

        # Publish to Tableau Server
        yield {"type": "status", "step": "publishing", "message": "Publication sur Tableau Server..."}
        view_url = None
        if twb_full_path:
            # Same-workbook accumulation: use session-consistent workbook name.
            # Defer session-state mutations until after publish_workbook() succeeds,
            # otherwise a failed first publish leaves session_workbook_path pointing
            # at a workbook that never existed on the server and every later turn
            # enters the overwrite branch and fails too.
            publish_path = twb_full_path
            is_first_publish = session_state.session_workbook_path is None
            pending_first_publish: tuple[str, str, str] | None = None  # (session_wb_path, session_wb_name, throwaway_path)
            if session_state.session_workbook_path:
                # Re-publish the accumulated session workbook (overwrite=true)
                publish_path = session_state.session_workbook_path
            else:
                # First publish in session — create the session-named copy on disk
                # (TSC matches by file name when overwriting). Do NOT touch
                # session_state yet — only commit if publish_workbook() returns ok.
                session_wb_name = f"Analyse_{req.session_id[:8]}"
                session_wb_path = str(settings.output_dir / f"{session_wb_name}.twb")
                shutil.copy2(twb_full_path, session_wb_path)
                publish_path = session_wb_path
                pending_first_publish = (session_wb_path, session_wb_name, twb_full_path)

            try:
                # First publish: no overwrite (creates new workbook)
                # Subsequent: overwrite=true (replaces existing workbook)
                workbook_luid = await publish_workbook(
                    publish_path, settings.tableau_default_project_id,
                    overwrite=not is_first_publish,
                )
                session_state.published_workbook_luid = workbook_luid
                view_url = await get_view_url(workbook_luid)

                # Commit first-publish state ONLY after publish succeeded.
                if pending_first_publish is not None:
                    session_wb_path, session_wb_name, throwaway_path = pending_first_publish
                    session_state.session_workbook_path = session_wb_path
                    session_state.session_workbook_name = session_wb_name
                    # In new_workbook mode the source is a throwaway uuid-named file —
                    # it's now copied into the session workbook, so drop it and repoint
                    # the response to the session workbook (what we publish + re-download).
                    # In sheet_added mode the source is a named/existing workbook we keep.
                    if mode == "new_workbook" and throwaway_path != session_wb_path:
                        try:
                            os.remove(throwaway_path)
                        except OSError:
                            pass
                        filename = f"{session_wb_name}.twb"
                        twb_full_path = session_wb_path
            except Exception as e:
                logger.warning("Failed to publish to Tableau Server: %s", e)
                if not warning:
                    warning = f"Publication échouée: {e}"
                # Roll back the unpublished Analyse_<sid>.twb copy so the next turn
                # retries as a clean first publish. The throwaway uuid .twb stays on
                # disk so the existing download URL still works.
                if pending_first_publish is not None:
                    session_wb_path, _, _ = pending_first_publish
                    try:
                        os.remove(session_wb_path)
                    except OSError:
                        pass

        # Update the session cart (in-place modify replaces the current chart; else append)
        if inplace_modify and session_state.cart:
            session_state.cart[-1] = viz_intent
        else:
            session_state.cart.append(viz_intent)

        # Record the turn with a readable memory line (never raw JSON), compact if long, persist
        _turn_kind = "modify" if inplace_modify else "create"
        session_state.turns.append(ConversationTurn(
            user_message=req.question,
            resolved_intent=viz_intent,
            twb_path=twb_full_path,
            kind=_turn_kind,
            assistant_text=_summarize_chart_turn(viz_intent, _turn_kind),
        ))
        await _maybe_compact_session(session_state)
        await _persist_session_state(session_state)

        latency_ms = time.monotonic() * 1000 - start_ms
        status_str = "partial" if warning else "success"

        end_trace(lf_trace)

        log_trace(
            trace_id=trace_id,
            session_id=req.session_id,
            question=req.question,
            viz_type=viz_intent.viz_type,
            title=viz_intent.title,
            model_id=settings.model_id,
            latency_ms=latency_ms,
            token_usage={},
            judge_score=judge_score,
            judge_feedback=judge_feedback,
            status=status_str,
        )
        async with AsyncSessionLocal() as db:
            await log_generation(
                db=db,
                trace_id=trace_id,
                session_id=req.session_id,
                question=req.question,
                viz_type=viz_intent.viz_type,
                title=viz_intent.title,
                model_id=settings.model_id,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                judge_score=judge_score,
                judge_feedback=judge_feedback,
                status=status_str,
            )

        msg = (
            f"Sheet '{viz_intent.title}' added to existing workbook"
            if mode == "sheet_added"
            else f"Generated {viz_intent.viz_type.replace('_', ' ').title()} — {viz_intent.title}"
        )

        yield {"type": "result", "response": ChatResponse(
            session_id=req.session_id,
            trace_id=trace_id,
            viz_intent=viz_intent,
            twb_filename=filename,
            twb_download_url=f"/download/{filename}",
            message=msg,
            warning=warning,
            mode=mode,
            judge_score=judge_score,
            judge_feedback=judge_feedback,
            view_url=view_url,
        )}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming entry point — drains the shared pipeline and returns the final
    ChatResponse. Progress events are ignored; an error event becomes an HTTPException."""
    async for ev in _run_chat_pipeline(req):
        if ev["type"] == "error":
            raise HTTPException(status_code=ev["code"], detail=ev["message"])
        if ev["type"] == "result":
            return ev["response"]
    raise HTTPException(status_code=500, detail="Chat pipeline produced no result")


# ---------------------------------------------------------------------------
# Chat/stream — SSE streaming endpoint
# ---------------------------------------------------------------------------

async def _sse_stream(req: ChatRequest):
    """Thin SSE adapter over the shared chat pipeline. Maps each pipeline event
    dict to an SSE frame; the event names/payloads are unchanged from before the
    P1 refactor (status/intent/error/result/done)."""

    def emit(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async for ev in _run_chat_pipeline(req):
        t = ev["type"]
        if t == "status":
            yield emit("status", {"step": ev["step"], "message": ev["message"]})
        elif t == "intent":
            yield emit("intent", {"viz_intent": ev["viz_intent"]})
        elif t == "error":
            yield emit("error", {
                "code": ev["code"],
                "message": ev["message"],
                "suggestion": ev.get("suggestion", ""),
            })
            return
        elif t == "result":
            yield emit("result", ev["response"].model_dump())
            yield emit("done", {})
            return


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _sse_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Download generated .twb
# ---------------------------------------------------------------------------

@app.get("/download/{filename}")
async def download(filename: str):
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = settings.output_dir / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(filepath),
        media_type="application/octet-stream",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post("/session/reset")
async def reset_session(body: dict):
    session_id = body.get("session_id", "")
    _session_states.pop(session_id, None)
    await delete_session_memory(session_id)
    return {"status": "ok", "session_id": session_id}


# ---------------------------------------------------------------------------
# Session cart — chart accumulation
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/charts")
async def get_session_charts(session_id: str):
    """Return the accumulated viz intents for a session (the 'cart')."""
    session_state = await _hydrate_session_state(session_id)
    return {"session_id": session_id, "charts": [v.model_dump() for v in session_state.cart]}


@app.post("/download/{session_id}")
async def download_session_workbook(session_id: str, req: DownloadRequest):
    """Generate and download a single .twb containing all charts in the session cart."""
    # Use charts from request body if provided (allows client-side removes),
    # otherwise fall back to server-side cart.
    if req.charts is not None:
        try:
            viz_intents = [VizIntent(**c) for c in req.charts]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid chart data: {exc}")
    else:
        viz_intents = (await _hydrate_session_state(session_id)).cart

    if not viz_intents:
        raise HTTPException(status_code=404, detail="No charts in session cart")

    # Resolve Server datasource from first viz that has a datasource_luid
    server_ds_content_url = None
    server_ds_name = None
    blend_secondary_content_url = None
    blend_secondary_name = None
    blend_linking_fields = None

    session_state = await _hydrate_session_state(session_id)
    available_datasources = session_state.available_datasources if session_state else []

    for viz in viz_intents:
        if viz.datasource_luid:
            try:
                server_ds_content_url = await get_datasource_content_url(viz.datasource_luid)
            except Exception as e:
                logger.warning("Could not resolve datasource content URL: %s", e)
            for ds in available_datasources:
                if ds.luid == viz.datasource_luid:
                    server_ds_name = ds.datasource_name
                    break

            if viz.secondary_datasource_luid:
                try:
                    blend_secondary_content_url = await get_datasource_content_url(viz.secondary_datasource_luid)
                except Exception as e:
                    logger.warning("Could not resolve secondary datasource content URL: %s", e)
                for ds in available_datasources:
                    if ds.luid == viz.secondary_datasource_luid:
                        blend_secondary_name = ds.datasource_name
                        break
                from twb_generator import _detect_linking_fields
                primary_schema = next((ds for ds in available_datasources if ds.luid == viz.datasource_luid), None)
                secondary_schema = next((ds for ds in available_datasources if ds.luid == viz.secondary_datasource_luid), None)
                if primary_schema and secondary_schema:
                    blend_linking_fields = _detect_linking_fields(primary_schema, secondary_schema)
            break  # use datasource from first viz that has one

    try:
        filename, out_path = generate_multi_sheet_twb(
            viz_intents, req.metadata,
            server_ds_content_url=server_ds_content_url,
            server_ds_name=server_ds_name,
            blend_secondary_content_url=blend_secondary_content_url,
            blend_secondary_name=blend_secondary_name,
            blend_linking_fields=blend_linking_fields,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FileResponse(
        str(out_path),
        media_type="application/octet-stream",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback")
async def feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    fb = Feedback(
        trace_id=req.trace_id,
        score=req.score,
        comment=req.comment or "",
    )
    db.add(fb)
    await db.commit()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Monitoring dashboard
# ---------------------------------------------------------------------------

@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    metrics = await get_monitoring_metrics(db)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return templates.TemplateResponse(
        request,
        "monitoring.html",
        {"metrics": metrics, "now": now},
    )


@app.get("/monitoring/metrics")
async def monitoring_metrics(db: AsyncSession = Depends(get_db)):
    return await get_monitoring_metrics(db)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

@app.get("/metadata/validate")
async def validate_metadata(metadata: DataSourceMetadata | None = None):
    if metadata is None:
        return {"valid": True, "message": "No metadata to validate"}
    return {"valid": True, "fields_count": len(metadata.fields)}


# ---------------------------------------------------------------------------
# Serve Tableau Extension static files at /extension
# ---------------------------------------------------------------------------

_ext_dir = Path("extension")
if _ext_dir.exists():
    app.mount("/extension", StaticFiles(directory="extension", html=True), name="extension")
