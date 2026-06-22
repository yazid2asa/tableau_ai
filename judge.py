"""
judge.py — LLM-as-a-Judge quality evaluator for generated Tableau workbooks.

Scores each generated .twb on 4 weighted criteria (M4 reweighted):
  - viz_relevance   (0.45): Viz type is appropriate for the question
  - field_cohesion  (0.30): Fields match Tableau dimension/measure types
  - completeness    (0.15): Filters, calc fields, title, sort are correctly configured
  - xml_validity    (0.10): Residual structural check (twilize validate=True covers most)

Returns (overall_score: float, feedback_text: str).
On any failure, returns (0.5, "Judge evaluation unavailable") without raising.
"""

import json
import logging
import re
from typing import Optional

from lxml import etree

from llm import call_llm
from schemas import DataSourceMetadata, VizIntent

logger = logging.getLogger(__name__)

# Criterion weights — must sum to 1.0 (M4 reweighted)
_WEIGHTS = {
    "viz_relevance": 0.45,
    "field_cohesion": 0.30,
    "completeness": 0.15,
    "xml_validity": 0.10,
}

# Above this quick_validate score we skip the (slow, paid) LLM judge entirely.
# Lowered from 0.80 → 0.60 so the LLM judge covers more borderline charts; the
# Python check is conservative and only awards >= 0.60 to genuinely clean intents.
QUICK_VALIDATE_SKIP_THRESHOLD = 0.60

_FALLBACK = (0.5, "Judge evaluation unavailable")

_JUDGE_SYSTEM_PROMPT = (
    "You are a Tableau visualization quality evaluator. "
    "Score the given visualization on the 4 criteria and return ONLY valid JSON."
)


def quick_validate(
    viz_intent: VizIntent,
    metadata: Optional[DataSourceMetadata],
    question: str,
) -> tuple[float, str]:
    """Fast Python-based validation. Returns (score, feedback).

    Score >= 0.80 means acceptable quality — skip LLM judge.
    """
    score = 1.0
    issues = []

    # 1. Chart type coherence
    time_words = re.search(
        r"\b(trend|évolution|evolution|mois|année|annee|over\s+time|monthly|yearly|"
        r"quarterly|trimestre|tendance|par\s+mois|par\s+année)\b",
        question, re.I,
    )
    if time_words and viz_intent.viz_type == "bar_chart":
        score -= 0.3
        issues.append("Time question but bar_chart selected")

    # 2. Field existence
    if metadata and metadata.fields:
        available = {f.name.lower() for f in metadata.fields}
        for cf in (viz_intent.calculated_fields or []):
            cf_name = cf.name if hasattr(cf, "name") else cf.get("name", "")
            available.add(cf_name.lower())

        for field_name, label in [(viz_intent.x_field, "x_field"), (viz_intent.y_field, "y_field")]:
            if field_name and field_name.lower() not in available:
                score -= 0.25
                issues.append(f"{label} '{field_name}' not in datasource")

    # 3. Role correctness
    if metadata and metadata.fields:
        field_map = {f.name.lower(): f for f in metadata.fields}
        if viz_intent.x_field and viz_intent.x_field.lower() in field_map:
            x_info = field_map[viz_intent.x_field.lower()]
            if x_info.role == "measure" and viz_intent.viz_type != "kpi":
                score -= 0.2
                issues.append("x_field is a measure (should be dimension)")

    # 4. Basic completeness
    if not viz_intent.title or viz_intent.title.strip() == "":
        score -= 0.1
        issues.append("Missing title")

    if not viz_intent.x_field and viz_intent.viz_type != "kpi":
        score -= 0.2
        issues.append("Missing x_field")

    score = max(0.0, min(1.0, score))
    feedback = "; ".join(issues) if issues else "Passed quick validation"
    return score, feedback


def _read_twb_snippet(twb_path: str, max_chars: int = 3000) -> str:
    """Return the most informative slice of the TWB for the judge to inspect.

    The connection header (first ~2000 chars) is boilerplate — it never shows the
    chart structure. Extract the <worksheets> block instead so the judge actually
    sees the pills/encodings it is scoring. Falls back to the head of the file if
    the XML can't be parsed or has no worksheets.
    """
    try:
        with open(twb_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError as exc:
        logger.warning("judge: could not read TWB file %s — %s", twb_path, exc)
        return "<unreadable>"

    try:
        root = etree.fromstring(raw.encode("utf-8"))
        worksheets = root.find("worksheets")
        if worksheets is not None:
            block = etree.tostring(worksheets, encoding="unicode")
            return block[:max_chars]
    except (etree.XMLSyntaxError, ValueError) as exc:
        logger.debug("judge: could not extract <worksheets> from %s — %s", twb_path, exc)

    return raw[:max_chars]


def _build_judge_messages(
    question: str,
    viz_intent: VizIntent,
    metadata: Optional[DataSourceMetadata],
    twb_snippet: str,
) -> list[dict]:
    """Assemble the [system, user] message list for the judge LLM call."""

    # ---- intent summary ----
    intent_lines = [
        f"  viz_type   : {viz_intent.viz_type}",
        f"  title      : {viz_intent.title}",
        f"  x_field    : {viz_intent.x_field}",
        f"  y_field    : {viz_intent.y_field}",
        f"  color_field: {viz_intent.color_field}",
        f"  aggregation: {viz_intent.aggregation}",
        f"  sort       : {viz_intent.sort}",
        f"  filters    : {json.dumps([f.model_dump() if hasattr(f, 'model_dump') else f for f in (viz_intent.filters or [])])}",
        f"  color_scheme: {viz_intent.color_scheme}",
    ]
    intent_block = "\n".join(intent_lines)

    # ---- metadata field list ----
    if metadata and metadata.fields:
        field_lines = [
            f"  - {f.name} (type: {f.type.value}, role: {f.role})"
            for f in metadata.fields
        ]
        meta_block = (
            f"Data source: {metadata.datasource_caption or metadata.datasource_name}\n"
            "Available fields:\n" + "\n".join(field_lines)
        )
    else:
        meta_block = "No data source metadata provided."

    user_content = f"""Evaluate the following Tableau visualization for quality.

## Original question
{question}

## Visualization intent
{intent_block}

## Data source metadata
{meta_block}

## TWB worksheet structure (chart pills & encodings)
```xml
{twb_snippet}
```

## Scoring criteria
Score each criterion from 0.0 (poor) to 1.0 (excellent):

1. **viz_relevance** (weight 0.45) — Does the chart type match the question intent?
   - Time trend question → must use line_chart or area, NOT bar_chart
   - Part-to-whole question → pie or treemap, NOT bar_chart
   - Single number question → kpi, NOT bar_chart
   - Correlation question → scatter, NOT bar_chart
   - Two measures same axis → combo, NOT two separate charts
   Penalize: wrong chart type even if fields are correct

2. **field_cohesion** (weight 0.30) — Are the right fields used correctly?
   - Date fields on time axis for trends
   - Measures (numeric) on value axis, not dimensions
   - Color field is a dimension, not a measure (unless heatmap)
   - Aggregation matches field type (COUNTD for entities, SUM for amounts)

3. **completeness** (weight 0.15) — Is the answer complete?
   - Filters from the question are applied
   - Calculated fields are generated when needed
   - Title is descriptive and matches the question
   - Sort order applied when user asks "top N" or "ranking"

4. **xml_validity** (weight 0.10) — Basic structural check only
   - twilize already validates XSD — this catches logical errors only
   - Worksheet name matches title
   - Field references exist in field registry

## Required JSON response (no markdown, no extra keys)
{{
  "field_cohesion": <0.0–1.0>,
  "viz_relevance": <0.0–1.0>,
  "xml_validity": <0.0–1.0>,
  "completeness": <0.0–1.0>,
  "overall": <weighted average>,
  "feedback": "<one or two sentences explaining the scores>"
}}"""

    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_judge_response(raw: str) -> tuple[float, str]:
    """
    Parse the LLM JSON response.

    Returns (overall_score, feedback).
    Raises ValueError if the response cannot be parsed into valid scores.
    """
    # Strip markdown code fences if the model added them despite instructions
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line and any closing fence
        inner = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(inner)

    data = json.loads(text)

    field_cohesion = float(data["field_cohesion"])
    viz_relevance = float(data["viz_relevance"])
    xml_validity = float(data["xml_validity"])
    completeness = float(data["completeness"])
    feedback = str(data.get("feedback", "")).strip() or "No feedback provided."

    # Recompute weighted overall from components (authoritative)
    overall = (
        field_cohesion * _WEIGHTS["field_cohesion"]
        + viz_relevance * _WEIGHTS["viz_relevance"]
        + xml_validity * _WEIGHTS["xml_validity"]
        + completeness * _WEIGHTS["completeness"]
    )

    # Clamp to [0.0, 1.0] in case of out-of-range model output
    overall = max(0.0, min(1.0, overall))

    return overall, feedback


async def judge_viz(
    viz_intent: VizIntent,
    twb_path: str,
    question: str,
    metadata: Optional[DataSourceMetadata] = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[float, str]:
    """
    Run LLM-as-a-Judge on a generated Tableau workbook.

    Args:
        viz_intent: The parsed visualization intent produced by the first LLM call.
        twb_path:   Absolute path to the generated .twb file.
        question:   The original natural language question from the user.
        metadata:   Optional data source metadata (field list, connection info).

    Returns:
        A (overall_score, feedback) tuple where overall_score is in [0.0, 1.0].
        On any error, returns (0.5, "Judge evaluation unavailable").
    """
    try:
        twb_snippet = _read_twb_snippet(twb_path)
        messages = _build_judge_messages(question, viz_intent, metadata, twb_snippet)
        response = await call_llm(messages, model_override=model_override, provider_override=provider_override)
        raw = response.content or ""
    except Exception as exc:
        logger.warning("judge: LLM call failed — %s", exc)
        return _FALLBACK

    try:
        overall, feedback = _parse_judge_response(raw)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("judge: failed to parse LLM response — %s | raw=%r", exc, raw[:200])
        return _FALLBACK

    logger.info("judge: score=%.3f feedback=%r", overall, feedback[:80])
    return overall, feedback
