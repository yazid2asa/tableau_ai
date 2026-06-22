from schemas import DataSourceMetadata, VizIntent

SYSTEM_PROMPT = """You are an expert Tableau data visualization assistant. You help users understand their data and create charts.

You have two capabilities:
1. TALK naturally — answer questions, explain concepts, suggest approaches, ask for clarification
2. CREATE CHARTS — when the user wants a visualization, use the generate_chart tool

DATASOURCE SELECTION: If multiple datasources are available, select the one whose fields best match the question. Field matching is authoritative — never pick by name alone.

WHEN TO USE generate_chart:
- User asks to show, create, make, display, visualize, chart, graph, or plot data
- User asks for a KPI, total, trend, comparison, distribution, proportion, ranking
- User asks to modify an existing chart (add filter, change type, sort, rename, etc.)
- User describes data they want to see (e.g. "ventes par région", "sales by category")

WHEN TO JUST RESPOND IN TEXT (do NOT call generate_chart):
- User asks about your capabilities ("can you...?", "tu peut...?", "est-ce possible de...?")
- User asks for help, explanation, or advice ("c'est quoi un treemap ?", "what chart should I use?")
- User greets you or makes small talk
- User asks about available fields or datasources
- The question is ambiguous and you need clarification before generating

RESPONSE STYLE (when responding in text):
- Never expose internal identifiers (LUIDs, UUIDs, content URLs) in your responses to the user
- When listing datasources, mention only the name and key fields
- Keep responses concise and natural — no code formatting unless the user asks for it

CHART TYPE DECISION (when using generate_chart):
- Time trend (évolution, tendance, par mois, over time, monthly, yearly) + date field → line_chart
- Proportion/share/répartition/part de/camembert/pie/donut → pie (ALWAYS, never override to bar_chart)
- Single number/KPI (total, combien, how many, overall) → kpi
- Correlation between 2 numeric fields → scatter
- Two measures, different scales → combo
- Matrix/crosstab (par X et par Y) → heatmap
- Hierarchical/nested breakdown (part of a whole with nesting) → treemap
- Table/list/detail view, show rows, breakdown with multiple dimensions + measure → text
- Default comparison/ranking → bar_chart

FIELD RULES (when using generate_chart):
- x_field and y_field MUST exactly match available field names — EXACT spelling, casing, separators
- x_field = dimension (category, date, name). y_field = measure (number, amount)
- For kpi: x_field = the measure, y_field = ""
- For pie: x_field = slices (dimension), y_field = size (measure)
- For heatmap: x_field = first dimension, y_field = second dimension, color_field = measure
- If a needed metric doesn't exist, create a calculated_field with Tableau formula syntax ([Field] brackets)
- Aggregation: monetary → SUM, entity counts → COUNT/COUNTD, ratios/averages → AVG, default → SUM
- CROSS-DATASOURCE CALC FIELDS ARE INVALID: a calculated field formula can ONLY reference fields from the primary datasource (datasource_luid). NEVER write a formula that mixes fields from primary AND secondary datasources (e.g., [Cargo Weight Kg]/[capacity_kg] where capacity_kg is in the secondary) — Tableau rejects this with "The calculation contains errors". If a metric requires fields from both datasources, place them separately on x_field/y_field/color_field shelves; do not combine them in a calculated_field formula.

IMPLICIT FILTERS — extract even when not explicitly stated:
- "rentable"/"profitable" → {field: Profit-like, op: gt, value: 0}
- "top N" → {field: x_field, op: top_n, value: N, by: y_field}
- Year "2024", "en 2023" → {field: date_field, op: year, value: YEAR}

CONTINUITY: If previous_intent is provided and the user is modifying it (add filter, change type, sort, rename, etc.), set action="modify" and merge changes onto the previous intent.
- Filter follow-ups ("filtré par X", "filter by Y", "only Z", "sans les W", "exclude V", "for 2024 only") → keep previous_intent.x_field / y_field / color_field unchanged and ADD the filter to the `filters` array. Never drop existing filters that the user did not ask to remove.
- Chart-type follow-ups ("en barres", "show as line") → keep all fields and filters; change only viz_type.
- Sort follow-ups → keep all fields/filters; change `sort` only.
- Returning an empty `filters` array on a filter follow-up is a bug — the filter the user just asked for must appear there.

MEMORY: You are given the recent conversation, a summary of earlier turns, and the list of charts already built this session. Use them — when the user refers back to something asked or created earlier, answer from that context instead of starting over.

Respond in the same language as the user."""


GENERATE_CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_chart",
        "description": (
            "Generate a Tableau visualization. Call this when the user wants to create, "
            "modify, or update a chart. Do NOT call this for questions about capabilities, "
            "explanations, or greetings."
        ),
        "parameters": {
            "type": "object",
            "required": ["viz_type", "title", "x_field", "action"],
            "properties": {
                "viz_type": {
                    "type": "string",
                    "enum": ["bar_chart", "line_chart", "pie", "scatter", "area",
                             "heatmap", "treemap", "kpi", "combo", "gantt", "text"],
                    "description": "Chart type.",
                },
                "title": {
                    "type": "string",
                    "description": "Descriptive chart title.",
                },
                "x_field": {
                    "type": "string",
                    "description": "Dimension field (categories, dates). For KPI: the measure field.",
                },
                "y_field": {
                    "type": "string",
                    "description": "Measure field (numbers, amounts). Empty string for KPI.",
                    "default": "",
                },
                "color_field": {
                    "type": ["string", "null"],
                    "description": "Optional color encoding field.",
                    "default": None,
                },
                "filters": {
                    "type": "array",
                    "description": "Filter specifications.",
                    "default": [],
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "op": {"type": "string", "enum": [
                                "eq", "in", "gt", "gte", "lt", "lte", "between",
                                "year", "quarter", "month",
                                "last_n_days", "last_n_months",
                                "top_n", "bottom_n", "not_null",
                            ]},
                            "value": {},
                            "values": {"type": "array"},
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "by": {"type": "string"},
                        },
                    },
                },
                "calculated_fields": {
                    "type": "array",
                    "description": "On-the-fly calculated fields using Tableau formula syntax with [Field] brackets.",
                    "default": [],
                    "items": {
                        "type": "object",
                        "required": ["name", "formula"],
                        "properties": {
                            "name": {"type": "string"},
                            "formula": {"type": "string"},
                            "datatype": {"type": "string", "default": "real"},
                            "role": {"type": "string", "default": "measure"},
                        },
                    },
                },
                "sort": {
                    "type": ["string", "null"],
                    "enum": ["ascending", "descending", None],
                    "description": "Sort order. Use descending for top N, rankings.",
                    "default": None,
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["SUM", "AVG", "COUNT", "COUNTD", "MIN", "MAX"],
                    "default": "SUM",
                },
                "color_scheme": {
                    "type": "string",
                    "default": "tableau10",
                },
                "action": {
                    "type": "string",
                    "enum": ["new", "modify", "clarify"],
                    "description": "'new' for new chart, 'modify' to update existing, 'clarify' when ambiguous.",
                },
                "datasource_luid": {
                    "type": ["string", "null"],
                    "description": "LUID of the Tableau Server datasource to use.",
                    "default": None,
                },
                "secondary_datasource_luid": {
                    "type": ["string", "null"],
                    "description": "LUID of a secondary datasource for data blending.",
                    "default": None,
                },
                "clarification_needed": {
                    "type": ["string", "null"],
                    "description": "Question to ask user when intent is ambiguous (use with action='clarify').",
                    "default": None,
                },
            },
        },
    },
}

TOOLS = [GENERATE_CHART_TOOL]


def _describe_filter(f) -> str:
    """Render one FilterSpec as a single readable line for the LLM prompt.

    Output is one line per filter: "Field — operator = value(s)".
    Pure read-only formatting — never alters the filter.
    """
    field = f.field or "?"
    op = (f.op or "").lower()
    if op in ("in",) and f.values is not None:
        return f"{field} — in {list(f.values)}"
    if op == "between" and f.min is not None and f.max is not None:
        return f"{field} — between [{f.min}, {f.max}]"
    if op in ("top_n", "bottom_n") and f.by:
        return f"{field} — {op} {f.value} by {f.by}"
    if op in ("not_null",):
        return f"{field} — not null"
    if f.value is not None:
        return f"{field} — {op} = {f.value}"
    return f"{field} — {op}"


def _format_existing_filters_section(filters) -> str:
    """Build the explicit 'EXISTING FILTERS' section so the LLM cannot lose
    previous filters when the user adds/removes/changes one in a follow-up.

    Returns "" when there are no filters to inject — caller decides whether
    to append it. The section is intentionally separate from the previous_intent
    JSON dump because filters buried inside JSON are routinely dropped by the LLM.
    """
    if not filters:
        return ""
    lines = ["", "=== EXISTING FILTERS ON THE CHART YOU MAY BE MODIFYING ==="]
    lines.append("The previous chart currently has these filters applied:")
    for i, f in enumerate(filters, 1):
        lines.append(f"  {i}. {_describe_filter(f)}")
    lines.append("")
    lines.append("Rules when the user asks to modify:")
    lines.append("  - \"add filter X\" / \"filtre par X\" / \"only X\" → KEEP all filters above, APPEND the new one.")
    lines.append("  - \"remove filter Y\" / \"sans Y\" / \"exclude Y\" → keep all filters EXCEPT the one matching Y.")
    lines.append("  - \"change filter Z to W\" / \"plutôt W\" → replace the matching filter, keep the others.")
    lines.append("  - If the user is starting a totally new chart (action=\"new\"), ignore these filters.")
    lines.append("")
    lines.append("Returning an empty `filters` array on a follow-up that adds/keeps a filter is a bug.")
    return "\n".join(lines) + "\n"


def _format_rag_context(
    knowledge: list[str],
    examples: list[dict],
) -> str:
    """Format retrieved RAG knowledge and examples into a prompt section."""
    parts: list[str] = []

    if knowledge:
        parts.append("=== TABLEAU GUIDANCE (use as advisory, reason past when needed) ===")
        for i, chunk in enumerate(knowledge, 1):
            parts.append(f"{i}. {chunk}")

    if examples:
        parts.append("\n=== SIMILAR PAST GENERATIONS (high quality — use as reference) ===")
        parts.append(
            "IMPORTANT: The examples below use field names from their original datasources. "
            "You MUST substitute with field names from the CURRENT datasource provided above. "
            "Never copy field names from examples — always use the actual available fields."
        )
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i} — Question: {ex['question']}")
            parts.append(f"  Score: {ex['judge_score']:.2f}")
            # Format the viz_intent compactly
            intent = ex.get("viz_intent", {})
            compact = {k: v for k, v in intent.items() if v is not None and v != [] and v != ""}
            parts.append(f"  Intent: {compact}")

    return "\n".join(parts)


def build_intent_prompt(
    question: str,
    metadata: DataSourceMetadata | None,
    history: list[dict],
    previous_intent: VizIntent | None = None,
    rag_knowledge: list[str] | None = None,
    rag_examples: list[dict] | None = None,
    available_datasources: list[DataSourceMetadata] | None = None,
    session_summary: str = "",
    charts_so_far: list[str] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Conversation history — the caller already trims to the desired window
    for msg in history:
        messages.append(msg)

    content = f"Question: {question}\n\n"
    if metadata:
        content += f"Data source: {metadata.datasource_caption or metadata.datasource_name}\n"
        content += "Available fields:\n"
        for f in metadata.fields:
            content += f"  - {f.name} (type: {f.type.value}, role: {f.role})\n"
    else:
        content += "No data source metadata provided — infer reasonable field names from the question.\n"

    if available_datasources and len(available_datasources) > 0:
        # Server mode: inject all datasource schemas for LLM to choose from
        parts = []
        parts.append("\n--- Available datasources (Tableau Server) ---")
        parts.append(
            "HARD RULE: the datasource you pick MUST contain every field you put in "
            "x_field, y_field, AND color_field. Do NOT pick a datasource just because "
            "its name matches a word in the question — verify the fields are actually "
            "listed below it."
        )
        parts.append(
            "If the fields you need live in TWO different datasources, set BOTH "
            "datasource_luid (primary, holds the measure) AND secondary_datasource_luid "
            "(holds the missing dimension). They will be blended on any shared field."
        )
        parts.append(
            "Worked example — question \"Trip count by brand\": `trip_id` is in the "
            "trips datasource and `brand` is in the vehicles datasource → primary=trips "
            "(for the count), secondary=vehicles (for brand), blend key=vehicle_id."
        )
        for ds in available_datasources:
            luid = getattr(ds, 'luid', None) or 'unknown'
            field_list = ", ".join(f.name for f in ds.fields) if ds.fields else "(no fields)"
            parts.append(f"\nDatasource: {ds.datasource_name}")
            parts.append(f"  Fields: {field_list}")
            parts.append(f"  [internal: luid={luid}]")
        content += "\n".join(parts) + "\n"

    if session_summary:
        content += f"\n--- Earlier in this conversation (summary) ---\n{session_summary}\n"

    if charts_so_far:
        content += "\n--- Charts already in this workbook ---\n"
        for i, c in enumerate(charts_so_far, 1):
            content += f"  {i}. {c}\n"
        content += "If the user refers to one of these, set action='modify'; for a new topic, set action='new'.\n"

    if previous_intent:
        content += "\nPrevious chart (last VizIntent):\n"
        content += previous_intent.model_dump_json(indent=2) + "\n"
        content += "If the user is modifying the above chart, set action='modify' and merge changes into it.\n"
        # Lift filters out of the JSON dump into an explicit, human-readable
        # section — buried inside the JSON the LLM routinely drops them on
        # follow-ups like "filtre par East" (it returns only the new filter).
        content += _format_existing_filters_section(previous_intent.filters or [])

    # Inject RAG context — advisory guidance, not hard rules
    if rag_knowledge or rag_examples:
        rag_text = _format_rag_context(rag_knowledge or [], rag_examples or [])
        if rag_text:
            content += "\n" + rag_text + "\n"

    messages.append({"role": "user", "content": content})
    return messages
