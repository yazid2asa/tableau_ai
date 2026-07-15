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
    get_dimension_members,
    query_datasource_aggregate,
    query_dimension_pairs,
)

logger = logging.getLogger(__name__)

# FIX-054: per-process cache of a dimension's distinct members, keyed by
# (datasource_luid, field_name_lower). Filter-value correction consults this so
# we fetch each dimension's domain from the VizQL Data Service at most once.
_member_cache: dict[tuple[str, str], list[str]] = {}

# Caches the LLM's semantic resolution of a filter value that matched no real
# member → the real member it denotes, keyed by (luid, field_lower, value_norm).
# "West" → "Ouest" is resolved once per session, then served from here.
_value_resolution_cache: dict[tuple[str, str, str], str | None] = {}


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


_FORMULA_REF_RE = _re.compile(r"\[([^\[\]]+)\]")
_FORMULA_AGG_RE = _re.compile(
    r"\b(?:SUM|AVG|MIN|MAX|MEDIAN|COUNT|COUNTD|STDEV|STDEVP|VAR|VARP|ATTR|PERCENTILE)\s*\(",
    _re.IGNORECASE,
)


def _correct_calc_field_formulas(viz: VizIntent, effective_ds: DataSourceMetadata) -> VizIntent:
    """Validate & fuzzy-correct every [Field] reference INSIDE calculated-field
    formulas (FIX-057). ``auto_correct_intent_fields`` fixes x/y/color and filter
    field names but never looked inside formulas, so a typo'd ref ([Sale] for
    [Sales]) published a broken pill — "The calculation contains errors".

    Per formula ref, in order: exact match against the datasource captions (or
    another calc field of the same intent) → normalized match → difflib fuzzy.
    A ref that resolves to nothing:
      - if the calc field is USED on a shelf (x/y/color) → ``clarification_needed``
        (asking beats publishing a knowingly broken chart);
      - if unused → the calc field is silently dropped (same spirit as FIX-044's
        cross-datasource drop).
    Formulas mixing aggregated and row-level refs (``SUM([A])/[B]`` — Tableau
    rejects with "Cannot mix aggregate and non-aggregate") are flagged in the log;
    FIX-009 handles the *usage* side, this covers the formula body.
    """
    if not viz.calculated_fields or not effective_ds or not effective_ds.fields:
        return viz

    available = [f.name for f in effective_ds.fields]
    available_set = set(available)
    calc_names = {cf.name for cf in viz.calculated_fields}
    norm_map = {_normalize(n): n for n in available}
    shelf_norms = {_normalize(v) for v in (viz.x_field, viz.y_field, viz.color_field or "") if v}

    new_cfs = []
    changed = False
    for cf in viz.calculated_fields:
        formula = cf.formula or ""
        broken: list[str] = []

        def _fix_ref(m) -> str:
            ref = m.group(1)
            if ref in available_set or ref in calc_names:
                return m.group(0)
            exact = norm_map.get(_normalize(ref))
            if exact:
                return f"[{exact}]"
            fuzzy = _auto_correct_field(ref, available)
            if fuzzy:
                return f"[{fuzzy}]"
            broken.append(ref)
            return m.group(0)

        new_formula = _FORMULA_REF_RE.sub(_fix_ref, formula)

        if broken:
            if _normalize(cf.name) in shelf_norms:
                preview = ", ".join(available[:8]) + ("…" if len(available) > 8 else "")
                logger.info("Calc field '%s' has unresolvable refs %s — asking for clarification",
                            cf.name, broken)
                return viz.model_copy(update={"clarification_needed": (
                    f"Le champ calculé « {cf.name} » référence "
                    f"{', '.join('[' + b + ']' for b in broken)} qui n'existe pas dans la source de données. "
                    f"Champs disponibles : {preview}. Quel champ faut-il utiliser ?"
                )})
            logger.info("Dropping unused calc field '%s' — unresolvable refs %s", cf.name, broken)
            changed = True
            continue

        # Aggregate/row-level mix check (log-only): iteratively strip aggregate
        # calls; any [ref] left outside an aggregate while the formula uses
        # aggregates means Tableau will reject it.
        if _FORMULA_AGG_RE.search(new_formula):
            residue, prev = new_formula, None
            while prev != residue:
                prev = residue
                residue = _re.sub(
                    r"\b(?:SUM|AVG|MIN|MAX|MEDIAN|COUNT|COUNTD|STDEV|STDEVP|VAR|VARP|ATTR|PERCENTILE)\s*\([^()]*\)",
                    "", residue, flags=_re.IGNORECASE)
            if _FORMULA_REF_RE.search(residue):
                logger.warning("Calc field '%s' mixes aggregated and row-level refs — "
                               "Tableau may reject it: %s", cf.name, new_formula)

        if new_formula != formula:
            logger.info("Calc formula correction: '%s' → '%s'", formula, new_formula)
            cf = cf.model_copy(update={"formula": new_formula})
            changed = True
        new_cfs.append(cf)

    if changed:
        viz = viz.model_copy(update={"calculated_fields": new_cfs})
    return viz


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


def _resolve_blend_linking_field(requested: str | None,
                                 primary: DataSourceMetadata,
                                 secondary: DataSourceMetadata) -> list[str] | None:
    """C5: resolve a user-named blend linking field against BOTH schemas.

    Normalized comparison (spaces/underscores/case) — stricter generalization of
    `_detect_linking_fields`' lower().strip() match, so "vehicle_id" resolves to
    the caption "Vehicle Id". Returns [primary_caption] when the field exists in
    both datasources, else None (caller falls back to auto-detection)."""
    if not requested:
        return None
    req_norm = _normalize(requested)
    p = next((f.name for f in primary.fields if _normalize(f.name) == req_norm), None)
    s = next((f.name for f in secondary.fields if _normalize(f.name) == req_norm), None)
    if p and s:
        logger.info("Blend linking field set from intent: %s", p)
        return [p]
    logger.warning("blend_linking_field '%s' not found in both datasources — auto-detecting",
                   requested)
    return None


_AGG_LABELS_FR = {
    "SUM": "Total", "AVG": "Moyenne", "MEDIAN": "Médiane",
    "COUNT": "Nombre", "COUNTD": "Nombre distinct", "MIN": "Minimum", "MAX": "Maximum",
}


def _fmt_number(v) -> str:
    """Human-readable number: thousands separators, trim trailing zeros."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return f"{int(f):,}".replace(",", " ")
    return f"{f:,.2f}".replace(",", " ")


def _format_data_answer(rows: list[dict], measure: str, aggregation: str,
                        group_by: str | None, filters: list[dict] | None) -> str:
    """Render VDS aggregate rows as a readable chat answer (C1 — query_data).

    Handles the VDS response generically: the measure column is whichever key
    holds a numeric value; the group column is the remaining key. Group results
    are sorted descending by the measure."""
    label = _AGG_LABELS_FR.get((aggregation or "SUM").upper(), aggregation or "SUM")
    flt_txt = ""
    if filters:
        parts = []
        for f in filters:
            vals = f.get("values") or []
            if f.get("field") and vals:
                neg = "≠ " if f.get("exclude") else ""
                parts.append(f"{f['field']} = {neg}{', '.join(str(v) for v in vals)}")
        if parts:
            flt_txt = f" (filtré : {' ; '.join(parts)})"

    def _measure_key(row: dict) -> str | None:
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return k
        # fall back: any value parseable as float
        for k, v in row.items():
            try:
                float(v)
                return k
            except (TypeError, ValueError):
                continue
        return None

    if not group_by:
        row = rows[0]
        mk = _measure_key(row)
        val = row.get(mk) if mk else None
        if val is None:
            return ("Je n'ai pas pu obtenir la valeur demandée depuis la source de données. "
                    "Voulez-vous que je génère un chart KPI à la place ?")
        return f"{label} de {measure}{flt_txt} : **{_fmt_number(val)}**"

    lines = [f"{label} de {measure} par {group_by}{flt_txt} :"]
    keyed: list[tuple[str, float, str]] = []
    for row in rows:
        mk = _measure_key(row)
        if mk is None:
            continue
        group_val = next((str(v) for k, v in row.items() if k != mk), "?")
        try:
            num = float(row[mk])
        except (TypeError, ValueError):
            continue
        keyed.append((group_val, num, _fmt_number(row[mk])))
    keyed.sort(key=lambda t: t[1], reverse=True)
    for g, _n, disp in keyed[:20]:
        lines.append(f"  • {g} : {disp}")
    if len(keyed) > 20:
        lines.append(f"  … ({len(keyed) - 20} autres)")
    return "\n".join(lines)


def _resolve_field_in(ds: DataSourceMetadata, field: str | None) -> str | None:
    """Resolve a user/LLM-worded field to the datasource's real caption
    (exact → normalized → fuzzy), or None if the DS doesn't own it."""
    if not field:
        return None
    names = [f.name for f in ds.fields]
    norm_map = {_normalize(n): n for n in names}
    return (field if field in names
            else norm_map.get(_normalize(field)) or _auto_correct_field(field, names))


async def _snap_filter_values(ds: DataSourceMetadata, raw_filters: list[dict]) -> list[dict]:
    """Correct filter field names + snap values to the DS's real members
    (FIX-054 machinery) → VDS-shaped ``{"field","values","exclude"}`` dicts."""
    out: list[dict] = []
    for f in raw_filters:
        fld = _resolve_field_in(ds, f.get("field"))
        vals = [str(v) for v in (f.get("values") or []) if v is not None]
        if not fld or not vals:
            continue
        try:
            members = _member_cache.get((ds.luid, fld.lower()))
            if members is None:
                members = await get_dimension_members(ds.luid, fld)
                _member_cache[(ds.luid, fld.lower())] = members
            if members:
                vals = [(_match_member(v, members) or v) for v in vals]
        except Exception:
            pass
        out.append({"field": fld, "values": vals, "exclude": bool(f.get("exclude", False))})
    return out


# Cross-datasource query_data guards: keep chat answers readable (aligned with
# the C1 top-20 rule) and the linking SET filter within a sane payload size.
_CROSS_DS_MAX_GROUPS = 20
_CROSS_DS_MAX_LINKS = 1000


async def _answer_cross_ds_question(
    measure_ds: DataSourceMetadata,
    datasources: list[DataSourceMetadata],
    r_measure: str,
    aggregation: str,
    own_filters: list[dict],
    foreign_group: str | None,
    foreign_filters: list[dict],
) -> str:
    """C1b — answer a factual question whose measure lives in one datasource
    and whose group-by / filter dimension lives in ANOTHER, linked by a common
    field ("combien de voyages des véhicules Ford ?", "coût moyen par marque ?").

    Tableau blending cannot render this as a single correct number for
    non-additive aggregations (measured — see FIX-062), but the join is exact
    in Python via the VizQL Data Service:

      1. the dimension-owning DS resolves the foreign fields —
         filter-only → the SET of linking values matching the filter (the
         vehicle_ids of brand Ford); group-by → the linking values of EACH
         group value (brand → [vehicle_ids]) via ``query_dimension_pairs``;
      2. the measure-owning DS aggregates with ``linking IN (set)`` — once for
         a filter-only question, once per group otherwise (asyncio.gather).

    Exact for every aggregation, COUNTD included (each query is a plain
    aggregate over the measure DS's rows — no re-aggregation of
    pre-aggregates). Always returns a user-facing string; never degrades to a
    silently unfiltered/ungrouped answer.
    """
    from twb_generator import _detect_linking_fields

    agg = (aggregation or "SUM").upper()
    foreign_fields = [f.get("field") for f in foreign_filters if f.get("field")]
    if foreign_group:
        foreign_fields.append(foreign_group)

    # 1. Partner datasource: owns every foreign field AND shares a linking field.
    dim_ds = None
    linking_caption = None
    for cand in datasources:
        if cand.luid == measure_ds.luid:
            continue
        if not all(_resolve_field_in(cand, f) for f in foreign_fields):
            continue
        links = _detect_linking_fields(measure_ds, cand)
        if links:
            dim_ds, linking_caption = cand, links[0]
            break
    if dim_ds is None:
        missing = ", ".join(str(f) for f in foreign_fields)
        return (f"Je ne trouve pas « {missing} » dans {measure_ds.datasource_name}, ni de "
                "source liée qui le contienne — impossible de croiser les données.")

    link_in_dim = _resolve_field_in(dim_ds, linking_caption) or linking_caption
    link_in_measure = _resolve_field_in(measure_ds, linking_caption) or linking_caption
    r_foreign_group = _resolve_field_in(dim_ds, foreign_group)

    # 2. Snap foreign filter values against the dimension DS's real members.
    dim_vds_filters = await _snap_filter_values(dim_ds, foreign_filters)
    display_filters = own_filters + dim_vds_filters

    def _fail() -> str:
        return (f"Je n'ai pas pu croiser {measure_ds.datasource_name} et "
                f"{dim_ds.datasource_name} — voulez-vous que je génère un chart à la place ?")

    def _first_number(row: dict):
        return next((v for v in row.values()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)), None)

    if r_foreign_group:
        pairs = await query_dimension_pairs(
            dim_ds.luid, link_in_dim, r_foreign_group, filters=dim_vds_filters)
        groups: dict[str, list[str]] = {}
        for row in pairs:
            link_v, group_v = row.get(link_in_dim), row.get(r_foreign_group)
            if link_v is None or group_v is None:
                continue
            groups.setdefault(str(group_v), []).append(str(link_v))
        if not groups:
            return _fail()
        if len(groups) > _CROSS_DS_MAX_GROUPS:
            return (f"« {r_foreign_group} » compte {len(groups)} valeurs — trop pour une "
                    "réponse en chat. Voulez-vous que je génère un chart à la place ?")
        if sum(len(v) for v in groups.values()) > _CROSS_DS_MAX_LINKS:
            return _fail()

        async def _one(gval: str, link_vals: list[str]):
            rows = await query_datasource_aggregate(
                measure_ds.luid, r_measure, agg, group_by=None,
                filters=own_filters + [{"field": link_in_measure, "values": link_vals}])
            return gval, rows

        results = await asyncio.gather(
            *(_one(g, l) for g, l in sorted(groups.items())), return_exceptions=True)
        out_rows: list[dict] = []
        for res in results:
            if isinstance(res, BaseException) or not res[1]:
                continue
            val = _first_number(res[1][0])
            if val is not None:
                out_rows.append({r_foreign_group: res[0], f"{agg}({r_measure})": val})
        if not out_rows:
            return _fail()
        return _format_data_answer(out_rows, r_measure, agg, r_foreign_group, display_filters)

    # Filter-only: resolve the linking SET on the dimension DS, then ONE
    # aggregate on the measure DS — exact for every aggregation.
    link_rows = await query_dimension_pairs(
        dim_ds.luid, link_in_dim, None, filters=dim_vds_filters)
    link_vals = sorted({str(r[link_in_dim]) for r in link_rows
                        if r.get(link_in_dim) is not None})
    if not link_vals:
        flt_txt = " ; ".join(
            f"{f['field']} = {', '.join(f['values'])}" for f in dim_vds_filters) or "ce filtre"
        return f"Aucune valeur de « {link_in_dim} » ne correspond à {flt_txt} — résultat : 0."
    if len(link_vals) > _CROSS_DS_MAX_LINKS:
        return _fail()
    rows = await query_datasource_aggregate(
        measure_ds.luid, r_measure, agg, group_by=None,
        filters=own_filters + [{"field": link_in_measure, "values": link_vals}])
    if not rows:
        return _fail()
    return _format_data_answer(rows, r_measure, agg, None, display_filters)


async def _answer_data_question(qargs: dict, datasources: list[DataSourceMetadata],
                                question: str | None) -> str:
    """Handle a `query_data` tool call end-to-end (C1): pick the datasource,
    normalize/fuzzy-correct field names, snap filter values to real members
    (same FIX-054 machinery as charts), query the VizQL Data Service, and
    format the answer. When the group-by / filter fields live in a DIFFERENT
    datasource than the measure, routes to the cross-datasource join (C1b —
    `_answer_cross_ds_question`) instead of silently dropping them. Always
    returns a user-facing string — degrades to a friendly message rather
    than raising."""
    measure = (qargs.get("measure") or "").strip()
    aggregation = (qargs.get("aggregation") or "SUM").upper()
    group_by = (qargs.get("group_by") or None)
    filters = list(qargs.get("filters") or [])
    if not measure:
        return "Quelle mesure faut-il calculer ? (ex. : total des ventes, coût moyen…)"
    if not datasources:
        return ("Je n'ai pas accès aux sources de données pour le moment — "
                "réessayez dans quelques instants.")

    # Pick the datasource: LLM-provided luid if valid, else best field match
    # (the MEASURE's owner weighs double — it must run the aggregate query;
    # foreign group/filter fields are joined in from the partner DS).
    ds = next((d for d in datasources if d.luid == qargs.get("datasource_luid")), None)
    if ds is None:
        best, best_score = None, -1
        for cand in datasources:
            cand_names = {_normalize(f.name) for f in cand.fields}
            score = (2 if _normalize(measure) in cand_names else 0) \
                + (1 if group_by and _normalize(group_by) in cand_names else 0)
            if score > best_score:
                best, best_score = cand, score
        ds = best or datasources[0]

    r_measure = _resolve_field_in(ds, measure)
    if r_measure is None:
        # The LLM may have pointed at the dimension-owning DS — re-pick by
        # measure ownership before giving up (the group joins back via C1b).
        owner = next((c for c in datasources
                      if c.luid != ds.luid and _resolve_field_in(c, measure)), None)
        if owner is not None:
            ds = owner
            r_measure = _resolve_field_in(ds, measure)
    if r_measure is None:
        names = [f.name for f in ds.fields]
        preview = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        return (f"Je ne trouve pas de champ « {measure} » dans {ds.datasource_name}. "
                f"Champs disponibles : {preview}")
    r_group = _resolve_field_in(ds, group_by)

    # Split filters by ownership: fields the measure DS can't resolve belong
    # to a partner DS (C1b). Before this split they were silently DROPPED —
    # "combien de voyages des véhicules Ford" answered the unfiltered total.
    own_raw = [f for f in filters if not f.get("field") or _resolve_field_in(ds, f["field"])]
    foreign_raw = [f for f in filters if f.get("field") and not _resolve_field_in(ds, f["field"])]
    foreign_group = group_by if (group_by and r_group is None) else None

    vds_filters = await _snap_filter_values(ds, own_raw)

    if foreign_group or foreign_raw:
        return await _answer_cross_ds_question(
            ds, datasources, r_measure, aggregation,
            vds_filters, foreign_group, foreign_raw)

    try:
        rows = await query_datasource_aggregate(
            ds.luid, r_measure, aggregation, group_by=r_group, filters=vds_filters)
    except Exception as exc:
        logger.warning("query_data VDS call failed: %s", exc)
        rows = []
    if not rows:
        return ("Je n'ai pas pu interroger la source de données "
                f"({ds.datasource_name}) — voulez-vous que je génère un chart KPI à la place ?")
    return _format_data_answer(rows, r_measure, aggregation, r_group, vds_filters)


def _match_sheet_title(requested: str | None, titles: list[str]) -> str | None:
    """Match a user-worded sheet reference to a real worksheet title.

    Order: exact → normalized → difflib → unique substring. None target →
    the most recent sheet. Never guesses between several substring hits."""
    if not titles:
        return None
    if not requested:
        return titles[-1]
    if requested in titles:
        return requested
    norm_map = {_normalize(t): t for t in titles}
    hit = norm_map.get(_normalize(requested))
    if hit:
        return hit
    close = difflib.get_close_matches(_normalize(requested), list(norm_map.keys()), n=1, cutoff=0.6)
    if close:
        return norm_map[close[0]]
    subs = [t for t in titles if _normalize(requested) in _normalize(t)]
    return subs[0] if len(subs) == 1 else None


async def _handle_manage_worksheet(margs: dict, session_state: SessionState) -> str:
    """Handle a `manage_worksheet` tool call (C2/C3): delete / rename / undo on
    the session workbook, then republish. Always returns a user-facing string.

    Undo semantics (v1): if the last chart turn has an earlier intent with the
    same title (an in-place modify), regenerate that previous version in place
    (single-datasource only); otherwise delete the most recent sheet. The undo
    turn records the restored/now-last intent so conversational continuity
    (last_intent) keeps pointing at a sheet that actually exists."""
    from twb_generator import (
        list_worksheet_titles, delete_sheet_from_workbook, rename_sheet_in_workbook,
    )
    op = (margs.get("operation") or "").lower()
    session_wb = session_state.session_workbook_path
    if not session_wb or not Path(session_wb).exists():
        return "Aucun classeur dans cette session pour le moment — créez d'abord un chart."
    titles = list_worksheet_titles(session_wb)
    if not titles:
        return "Le classeur de la session ne contient aucune feuille."

    async def _republish() -> str | None:
        try:
            await publish_workbook(session_wb, settings.tableau_default_project_id, overwrite=True)
            return None
        except Exception as exc:
            logger.exception("manage_worksheet republish failed")
            return (f" ⚠️ La republication sur Tableau Server a échoué "
                    f"({type(exc).__name__}) — le changement sera repris à la prochaine publication.")

    def _drop_cart_entry(title: str) -> None:
        for i, v in enumerate(session_state.cart):
            if _normalize(v.title) == _normalize(title):
                del session_state.cart[i]
                return

    if op == "delete":
        target = _match_sheet_title(margs.get("sheet_title"), titles)
        if target is None:
            return (f"Je ne trouve pas de feuille correspondant à « {margs.get('sheet_title')} ». "
                    f"Feuilles actuelles : {', '.join(titles)}")
        if len(titles) <= 1:
            return ("Impossible de supprimer la dernière feuille — un classeur Tableau doit en "
                    "garder au moins une. Utilisez « nouvelle session » pour repartir de zéro.")
        if not delete_sheet_from_workbook(session_wb, target):
            return f"La suppression de « {target} » a échoué."
        _drop_cart_entry(target)
        warn = await _republish()
        return f"Feuille « {target} » supprimée du classeur.{warn or ''}"

    if op == "rename":
        new_title = (margs.get("new_title") or "").strip()
        if not new_title:
            return "Quel nouveau nom faut-il donner à la feuille ?"
        target = _match_sheet_title(margs.get("sheet_title"), titles)
        if target is None:
            return (f"Je ne trouve pas de feuille correspondant à « {margs.get('sheet_title')} ». "
                    f"Feuilles actuelles : {', '.join(titles)}")
        final = rename_sheet_in_workbook(session_wb, target, new_title)
        if final is None:
            return f"Le renommage de « {target} » a échoué."
        for i, v in enumerate(session_state.cart):
            if _normalize(v.title) == _normalize(target):
                session_state.cart[i] = v.model_copy(update={"title": final})
                break
        warn = await _republish()
        return f"Feuille « {target} » renommée en « {final} ».{warn or ''}"

    if op == "undo":
        last_intent = session_state.last_intent
        if last_intent is None:
            return "Rien à annuler — aucun chart n'a encore été créé dans cette session."
        current_title = _match_sheet_title(last_intent.title, titles) or titles[-1]
        # An earlier intent with the same title ⇒ the last action was an in-place
        # modify — restore that previous version (single-DS only in v1).
        prev_same = next(
            (t.resolved_intent for t in reversed(session_state.turns)
             if t.resolved_intent is not None
             and t.resolved_intent is not last_intent
             and _normalize(t.resolved_intent.title) == _normalize(last_intent.title)),
            None,
        )
        if prev_same is not None and not prev_same.secondary_datasource_luid:
            try:
                ds = next((d for d in (session_state.available_datasources or [])
                           if d.luid == prev_same.datasource_luid), None)
                cu = await get_datasource_content_url(prev_same.datasource_luid) if prev_same.datasource_luid else None
                modify_sheet_in_existing(
                    session_wb, prev_same, ds, old_title=current_title,
                    server_ds_content_url=cu,
                    server_ds_name=ds.datasource_name if ds else None,
                )
                if session_state.cart:
                    session_state.cart[-1] = prev_same
                session_state.turns.append(ConversationTurn(
                    user_message="(undo)", kind="modify", resolved_intent=prev_same,
                    assistant_text=f"Modification annulée — « {prev_same.title} » restauré.",
                ))
                warn = await _republish()
                return (f"Dernière modification annulée — « {prev_same.title} » est revenu "
                        f"à sa version précédente.{warn or ''}")
            except Exception as exc:
                logger.exception("undo-restore failed, falling back to delete")
        # Otherwise: the last action added this sheet — remove it.
        if len(titles) <= 1:
            return ("Impossible d'annuler : c'est la seule feuille du classeur. "
                    "Utilisez « nouvelle session » pour repartir de zéro.")
        if not delete_sheet_from_workbook(session_wb, current_title):
            return f"L'annulation a échoué (feuille « {current_title} » introuvable)."
        _drop_cart_entry(current_title)
        restored = session_state.cart[-1] if session_state.cart else None
        session_state.turns.append(ConversationTurn(
            user_message="(undo)", kind="answer", resolved_intent=restored,
            assistant_text=f"Dernier chart « {current_title} » supprimé (undo).",
        ))
        warn = await _republish()
        return f"Dernier chart annulé — feuille « {current_title} » supprimée.{warn or ''}"

    return "Opération non reconnue — je peux supprimer, renommer une feuille, ou annuler (undo)."


async def _handle_create_dashboard(dargs: dict, session_state: SessionState) -> str:
    """Handle a `create_dashboard` tool call (C4): assemble existing sheets into
    a grid dashboard inside the session workbook, then republish."""
    from twb_generator import list_worksheet_titles, add_dashboard_to_workbook
    session_wb = session_state.session_workbook_path
    if not session_wb or not Path(session_wb).exists():
        return "Aucun classeur dans cette session pour le moment — créez d'abord des charts."
    titles = list_worksheet_titles(session_wb)
    if not titles:
        return "Le classeur de la session ne contient aucune feuille."

    requested = [t for t in (dargs.get("sheet_titles") or []) if t]
    if requested:
        picked: list[str] = []
        missing: list[str] = []
        for r in requested:
            hit = _match_sheet_title(r, titles)
            if hit and hit not in picked:
                picked.append(hit)
            elif not hit:
                missing.append(r)
        if missing:
            return (f"Je ne trouve pas de feuille correspondant à : {', '.join(missing)}. "
                    f"Feuilles actuelles : {', '.join(titles)}")
    else:
        picked = titles

    title = (dargs.get("title") or "").strip() or "Dashboard"
    try:
        final = add_dashboard_to_workbook(session_wb, title, picked)
    except Exception as exc:
        logger.exception("create_dashboard failed")
        return f"La création du dashboard a échoué ({type(exc).__name__})."
    try:
        await publish_workbook(session_wb, settings.tableau_default_project_id, overwrite=True)
    except Exception as exc:
        logger.exception("create_dashboard republish failed")
        return (f"Dashboard « {final} » créé avec {len(picked)} feuille(s), mais la republication "
                f"a échoué ({type(exc).__name__}) — il sera publié à la prochaine génération.")
    return (f"Dashboard « {final} » créé avec {len(picked)} feuille(s) : "
            f"{', '.join(picked)}. Il est publié sur Tableau Server.")


def _extract_intent_from_text(content: str) -> dict | None:
    """Fallback parser for providers whose tool-calling is flaky (FIX-058).

    When Google rate-limits, the OpenRouter fallback sometimes answers with the
    intent JSON as plain TEXT instead of a generate_chart tool call — the pipeline
    then wrongly returned it as a conversation bubble and no chart was generated.
    This recovers the intent from the text, accepting it ONLY when it unambiguously
    looks like a chart intent (a JSON object carrying both `viz_type` and
    `x_field`), so genuine conversational answers that merely contain braces are
    never hijacked into a chart."""
    if not content or "viz_type" not in content:
        return None
    start = content.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(content[start:])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("viz_type") or "x_field" not in data:
        return None
    return data


def _norm_value(s: str) -> str:
    """Case- and accent-insensitive normalization for matching filter values to
    real datasource members ("OUest"/"ouest"/"OUEST" → "ouest")."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold().strip()


def _match_member(value: str, members: list[str]) -> str | None:
    """Map a requested filter value to the real datasource member it denotes.

    Order: exact → case/accent-insensitive → fuzzy (difflib on normalized forms).
    Returns the real member string, or None when nothing plausible matches
    (caller then warns/clarifies instead of emitting a chart-emptying filter)."""
    if not members:
        return None
    if value in members:
        return value
    nv = _norm_value(value)
    norm_to_member: dict[str, str] = {}
    for m in members:
        norm_to_member.setdefault(_norm_value(m), m)
    if nv in norm_to_member:
        return norm_to_member[nv]
    # Conservative fuzzy: only snap on a high-confidence typo (e.g. "Oest"→"Ouest").
    # A loose cutoff would wrongly map a translated/invented value ("West"→"Est");
    # better to return None and let the caller warn than to silently pick the wrong member.
    close = difflib.get_close_matches(nv, list(norm_to_member.keys()), n=1, cutoff=0.88)
    if close:
        return norm_to_member[close[0]]
    return None


# Short / stopword tokens that must never be treated as a member mention when
# recovering a filter value from the question (French "est" = the verb "is",
# English "in"/"on", etc.). A member named exactly one of these is skipped for
# recovery — the LLM almost always emits the correct value for it anyway.
_MEMBER_RECOVERY_STOPWORDS = {
    "est", "et", "ou", "de", "des", "du", "la", "le", "les", "un", "une",
    "is", "in", "on", "or", "and", "the", "by", "for", "to", "of", "a",
}


def _recover_member_from_question(question: str | None, members: list[str]) -> str | None:
    """Last-resort recovery when the LLM rewrote a filter value into something that
    no longer matches the data (e.g. it *translated* the user's "OUest" → "West",
    which `_match_member` rightly refuses to fuzzy-map to "Est"). The user's own
    word is still in the question, so scan it for a real member the user literally
    wrote. Returns a member ONLY when exactly one matches — never guesses between
    several — so it can pick the right value without risking a wrong one."""
    if not question or not members:
        return None
    qn = _norm_value(question)
    qtokens = set(_re.findall(r"[a-z0-9]+", qn))
    hits: list[str] = []
    for m in members:
        mn = _norm_value(m)
        if not mn:
            continue
        if " " in m or "-" in m:
            matched = mn in qn  # multi-word member → substring on the normalized question
        else:
            matched = mn in qtokens and mn not in _MEMBER_RECOVERY_STOPWORDS and len(mn) >= 3
        if matched and m not in hits:
            hits.append(m)
    return hits[0] if len(hits) == 1 else None


async def _resolve_member_via_llm(
    field: str, members: list[str], question: str | None, failed_value: str, luid: str,
) -> str | None:
    """Resolve a filter value that matched no real member to the member the user
    *meant*, using the LLM with the field's actual domain in context (FIX-054c).

    The user doesn't know the datasource's exact records — they may ask for the
    "West region" when the data stores "Ouest". Deterministic string matching can
    never bridge that (it would wrongly snap "West"→"Est"), but the LLM, shown the
    real members, knows "West" denotes "Ouest". This is the semantic equivalent of
    capturing the user's *intent* rather than echoing their literal word.

    Defensive: the LLM's answer is accepted ONLY if it is exactly one of the real
    members (re-validated through `_match_member`); anything else (incl. "NONE" or a
    hallucinated value) → None, so a wrong member is never shipped.

    Reliability (FIX-054c hardening): the call is non-deterministic, so one bad
    answer must not blank the chart — an invalid/failed attempt is retried ONCE,
    and only SUCCESSES are cached (a cached failure used to make the miss
    permanent for the session, reproducing the very bug this fix closes)."""
    if not members or not failed_value:
        return None
    cache_key = (luid, field.lower(), _norm_value(failed_value))
    if cache_key in _value_resolution_cache:
        return _value_resolution_cache[cache_key]

    allowed = ", ".join(members)
    sys_msg = (
        "You map a user's requested filter value to the value that actually exists "
        "in a dataset column (the user may not know the exact stored value, or may "
        "use another language/synonym). Reply with ONLY one value copied verbatim "
        "from the allowed list, or the single word NONE. No explanation, no quotes."
    )
    user_msg = (
        f'Column "{field}" allows exactly these values: {allowed}.\n'
        f'User request: "{question or failed_value}"\n'
        f'The user referred to "{failed_value}" for this column, but it is not in the '
        f'allowed list. Which ONE allowed value did they most likely mean? '
        f'Reply with the exact allowed value, or NONE.'
    )
    resolved: str | None = None
    for attempt in (1, 2):
        try:
            resp = await call_llm([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ])
            ans = (resp.content or "").strip().strip('"').strip()
            if ans and ans.upper() != "NONE":
                resolved = _match_member(ans, members)  # accept only a real member
            elif ans.upper() == "NONE":
                # A deliberate NONE is an answer, not a flake — don't retry it.
                break
        except Exception as exc:
            logger.warning("filter-value LLM resolution attempt %d failed for %s '%s': %s",
                           attempt, field, failed_value, exc)
        if resolved is not None:
            logger.info("filter-value LLM resolution: %s '%s' → '%s' (attempt %d)",
                        field, failed_value, resolved, attempt)
            break
    if resolved is not None:
        _value_resolution_cache[cache_key] = resolved  # cache successes only
    return resolved


async def _correct_filter_values(
    viz: VizIntent,
    available_datasources: list[DataSourceMetadata] | None,
    question: str | None = None,
) -> tuple[VizIntent, str | None]:
    """Validate/correct member-based filter VALUES against the real datasource
    domain (FIX-054). String filters in Tableau are case-sensitive, so a value the
    LLM mis-cased / mis-accented / invented ("OUest" vs "Ouest") publishes fine but
    selects 0 members → a blank chart. For each eq/in filter on a dimension, fetch
    that dimension's distinct members (cached) and snap each value to the real one.
    When a value has no plausible match, two fallbacks run in order: (FIX-054b)
    recover the member the user literally wrote in `question`; then (FIX-054c) ask
    the LLM to map the value to the member the user *meant*, given the field's real
    domain — this captures intent across language/synonyms ("West" → "Ouest") which
    string matching can't. Only if all of these fail do we return a human-readable
    warning so the caller can surface it instead of shipping an empty chart.

    Returns (possibly-corrected viz, warning-or-None). Degrades to a no-op when
    members can't be fetched (VDS disabled / offline) so it never blocks generation.
    """
    luid = viz.datasource_luid
    if not viz.filters or not luid or not available_datasources:
        return viz, None
    ds = next((d for d in available_datasources if d.luid == luid), None)
    if ds is None:
        return viz, None
    # Map each dimension to the datasource that OWNS it. A blend chart can filter
    # on a secondary-owned dimension ("vehicle_type" while primary is trips) —
    # its members live in the SECONDARY datasource, so fetching them with the
    # primary luid returned nothing and the value was never protected (FIX-054
    # hardening, part of the FIX-054c reliability pass).
    dim_luids: dict[str, str] = {f.name: luid for f in ds.fields if f.role == "dimension"}
    if viz.secondary_datasource_luid:
        sec = next((d for d in available_datasources
                    if d.luid == viz.secondary_datasource_luid), None)
        if sec is not None:
            for fld in sec.fields:
                if fld.role == "dimension" and fld.name not in dim_luids:
                    dim_luids[fld.name] = sec.luid

    warnings: list[str] = []
    notes: list[str] = []   # user-visible transparency for semantic corrections
    new_filters = []
    changed = False
    for f in viz.filters:
        # neq/not_in are corrected too: a mis-cased EXCLUDED value silently
        # excludes nothing (the inverse failure of the eq blank chart) — FIX-055.
        if f.op not in ("eq", "in", "neq", "not_in") or f.field not in dim_luids:
            new_filters.append(f)
            continue
        f_luid = dim_luids[f.field]
        cache_key = (f_luid, f.field.lower())
        members = _member_cache.get(cache_key)
        if members is None:
            try:
                members = await get_dimension_members(f_luid, f.field)
            except Exception as exc:
                logger.warning("filter-value correction: member fetch failed for %s.%s: %s",
                               ds.datasource_name, f.field, exc)
                members = []
            _member_cache[cache_key] = members
        if not members:
            new_filters.append(f)
            continue

        if f.op in ("eq", "neq") and f.value is not None:
            corrected = _match_member(str(f.value), members)
            if corrected is None:
                corrected = _recover_member_from_question(question, members)
                if corrected is not None:
                    logger.info("filter-value recovery from question: %s eq '%s' → '%s'",
                                f.field, f.value, corrected)
            if corrected is None:  # semantic/translation mapping against the real domain
                corrected = await _resolve_member_via_llm(
                    f.field, members, question, str(f.value), f_luid)
            if corrected is None:
                warnings.append(
                    f"« {f.value} » n'existe pas dans {f.field} "
                    f"(valeurs possibles : {', '.join(members[:8])}{'…' if len(members) > 8 else ''})"
                )
            elif corrected != f.value:
                logger.info("filter-value correction: %s %s '%s' → '%s'", f.field, f.op, f.value, corrected)
                # Semantic change (not a mere case/accent fix) → tell the user which
                # member was actually used, so the interpretation is visible.
                if _norm_value(corrected) != _norm_value(str(f.value)):
                    notes.append(f"filtre {f.field} : « {f.value} » interprété comme « {corrected} »")
                f = f.model_copy(update={"value": corrected})
                changed = True
        elif f.op in ("in", "not_in") and f.values:
            new_vals = []
            for v in f.values:
                c = _match_member(str(v), members)
                if c is None:
                    c = _recover_member_from_question(question, members)
                    if c is not None:
                        logger.info("filter-value recovery from question: %s in '%s' → '%s'",
                                    f.field, v, c)
                if c is None:  # semantic/translation mapping against the real domain
                    c = await _resolve_member_via_llm(f.field, members, question, str(v), f_luid)
                if c is None:
                    new_vals.append(v)
                    warnings.append(
                        f"« {v} » n'existe pas dans {f.field} "
                        f"(valeurs possibles : {', '.join(members[:8])}{'…' if len(members) > 8 else ''})"
                    )
                else:
                    if _norm_value(c) != _norm_value(str(v)):
                        notes.append(f"filtre {f.field} : « {v} » interprété comme « {c} »")
                    if c not in new_vals:  # recovery can map several failed values to one member
                        new_vals.append(c)
            if new_vals != list(f.values):
                logger.info("filter-value correction: %s in %s → %s", f.field, f.values, new_vals)
                f = f.model_copy(update={"values": new_vals})
                changed = True
        new_filters.append(f)

    if changed:
        viz = viz.model_copy(update={"filters": new_filters})
    combined = warnings + notes
    return viz, ("; ".join(combined) if combined else None)


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
        # --- 2b. Calculated-field formula refs (FIX-057) — same effective_ds
        # (primary + secondary merged when blending), so a formula referencing a
        # blended field is validated against the full field universe.
        viz_intent = _correct_calc_field_formulas(viz_intent, effective_ds)
        if viz_intent.clarification_needed:
            return viz_intent

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

    # --- 3b. Value-slot must hold a measure (FIX-053) ---
    # For chart types that aggregate y_field as the value, a *dimension* in y_field
    # is turned into SUM(<dimension>) by the builders → an invalid red pill / blank
    # cell (the reported "list sales and profit by category and region" table that
    # rendered SUM(Region)). When the measure is sitting in color_field instead,
    # swap them: the measure becomes the value and the 2nd dimension moves to color
    # (rows/columns for a text table → a valid crosstab). heatmap/kpi/gantt are
    # excluded — their y_field is intentionally a dimension or handled elsewhere.
    _VALUE_ON_Y = {"bar_chart", "line_chart", "area", "pie", "treemap", "text", "combo", "scatter"}
    _NUMERIC_AGG = {"SUM", "AVG", "MIN", "MAX"}
    if viz_intent.viz_type in _VALUE_ON_Y and viz_intent.y_field:
        y_info = _find_field_info(viz_intent.y_field, available_datasources, viz_intent.datasource_luid)
        c_info = (_find_field_info(viz_intent.color_field, available_datasources, viz_intent.datasource_luid)
                  if viz_intent.color_field else None)
        # Only act when y is a real datasource dimension (a calc-field name resolves
        # to None here and is left alone — it may legitimately be a measure).
        if y_info and y_info.role == "dimension":
            if c_info and c_info.role == "measure":
                logger.info("Self-correction: value-slot — swapped y_field(dim '%s')/color_field(measure '%s')",
                            viz_intent.y_field, viz_intent.color_field)
                updates["y_field"] = viz_intent.color_field
                updates["color_field"] = viz_intent.y_field
            elif viz_intent.aggregation in _NUMERIC_AGG:
                # No measure available to swap in — count the dimension instead of
                # SUM-ing it, so we never emit SUM(<dimension>).
                logger.info("Self-correction: value-slot — y_field '%s' is a dimension with no measure; "
                            "aggregation %s → COUNTD", viz_intent.y_field, viz_intent.aggregation)
                updates["aggregation"] = "COUNTD"

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


def _is_structural_change(new: VizIntent, prev: VizIntent) -> bool:
    """True when a follow-up should become a NEW sheet rather than an in-place tweak.

    ONLY a change of the structural fields x/y/color (added/swapped dimension or
    measure) triggers a new sheet — a different field layout is a different
    analytical question. Everything else (filters, viz_type, sort, aggregation,
    title) refines the SAME question and updates the current sheet in place
    (FIX-012 — see FIXES.md: promoting filter/viz_type changes to "structural"
    was reverted twice; users asking "filtre par 2024" expect THEIR chart to be
    filtered, not a near-duplicate sheet added next to it).
    Comparison is normalized to tolerate caption/physical spelling differences.
    """
    return (
        _normalize(new.x_field or "") != _normalize(prev.x_field or "")
        or _normalize(new.y_field or "") != _normalize(prev.y_field or "")
        or _normalize(new.color_field or "") != _normalize(prev.color_field or "")
    )


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

        # Branch: query_data tool — answer a factual question with real numbers
        # from the VizQL Data Service, no chart generated (C1).
        if llm_response.has_tool_call and llm_response.tool_calls[0].name == "query_data":
            yield {"type": "status", "step": "querying",
                   "message": "Interrogation de la source de données..."}
            answer_text = await _answer_data_question(
                llm_response.tool_calls[0].arguments or {},
                session_state.available_datasources or available_datasources or [],
                req.question,
            )
            session_state.turns.append(ConversationTurn(
                user_message=req.question, kind="answer", assistant_text=answer_text,
            ))
            await _persist_session_state(session_state)
            end_trace(lf_trace)
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=answer_text,
                mode="conversation",
            )}
            return

        # Branch: manage_worksheet tool — delete/rename a sheet or undo (C2/C3).
        if llm_response.has_tool_call and llm_response.tool_calls[0].name == "manage_worksheet":
            yield {"type": "status", "step": "managing",
                   "message": "Mise à jour du classeur..."}
            answer_text = await _handle_manage_worksheet(
                llm_response.tool_calls[0].arguments or {}, session_state)
            session_state.turns.append(ConversationTurn(
                user_message=req.question, kind="answer", assistant_text=answer_text,
            ))
            await _persist_session_state(session_state)
            end_trace(lf_trace)
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=answer_text,
                mode="conversation",
            )}
            return

        # Branch: create_dashboard tool — assemble sheets into a dashboard (C4).
        if llm_response.has_tool_call and llm_response.tool_calls[0].name == "create_dashboard":
            yield {"type": "status", "step": "dashboard",
                   "message": "Assemblage du dashboard..."}
            answer_text = await _handle_create_dashboard(
                llm_response.tool_calls[0].arguments or {}, session_state)
            session_state.turns.append(ConversationTurn(
                user_message=req.question, kind="answer", assistant_text=answer_text,
            ))
            await _persist_session_state(session_state)
            end_trace(lf_trace)
            yield {"type": "result", "response": ChatResponse(
                session_id=req.session_id,
                trace_id=trace_id,
                message=answer_text,
                mode="conversation",
            )}
            return

        # Branch: conversation (text-only) vs chart generation (tool call)
        rescued_intent: dict | None = None
        if not llm_response.has_tool_call:
            # FIX-058: the OpenRouter fallback sometimes emits the intent JSON as
            # plain text instead of a tool call — rescue it so a rate-limited
            # primary provider still produces a chart instead of a JSON bubble.
            rescued_intent = _extract_intent_from_text(llm_response.content or "")
            if rescued_intent is not None:
                logger.info("Intent rescued from text response (provider tool-calling fallback)")
        if not llm_response.has_tool_call and rescued_intent is None:
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

        intent_data = rescued_intent if rescued_intent is not None else llm_response.first_tool_args
        reasoning_text = "" if rescued_intent is not None else (llm_response.content or "")

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

        # Validate/correct filter VALUES against the real datasource members so a
        # mis-cased / mis-accented / invented value (e.g. "OUest" vs "Ouest") doesn't
        # publish a filter that selects 0 members → a blank chart (FIX-054).
        filter_value_warning: str | None = None
        try:
            viz_intent, filter_value_warning = await _correct_filter_values(
                viz_intent, all_session_ds, req.question)
        except Exception as exc:
            logger.warning("filter-value correction skipped: %s", exc)

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
                    # C5: a user-named linking field ("lie-les sur vehicle_id") takes
                    # precedence over auto-detection — this also makes the "which
                    # field should link them?" clarification actually answerable.
                    blend_linking_fields = _resolve_blend_linking_field(
                        viz_intent.blend_linking_field, primary_schema, secondary_schema)
                    if not blend_linking_fields:
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
        warning: str | None = filter_value_warning  # FIX-054: surface unresolved filter values
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
                _q = f"Qualité partielle — quality score: {judge_score:.2f}"
                warning = (warning + " · " + _q) if warning else _q

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
