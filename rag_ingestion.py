"""
rag_ingestion.py — One-time data ingestion pipeline for ChromaDB RAG collections.

Seeds two persistent ChromaDB collections on first startup (count == 0):
  - tableau_rules     : Tableau docs (4 pages) + GitHub markdown + 13-chunk fallback
  - golden_examples   : 30 curated Superstore question/intent pairs

All HTTP fetches are graceful — failures log a warning and skip that source.
The 13-chunk fallback guarantees tableau_rules is never empty even when all HTTP fails.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_MIN_CHUNK = 80
_MAX_CHUNK = 2000

# ---------------------------------------------------------------------------
# Tableau docs URLs
# ---------------------------------------------------------------------------

_TABLEAU_DOC_URLS = [
    (
        "tableau_functions",
        "https://help.tableau.com/current/pro/desktop/en-us/functions_all_categories.htm",
    ),
    (
        "tableau_calc_fields",
        "https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_create.htm",
    ),
    (
        "tableau_filtering",
        "https://help.tableau.com/current/pro/desktop/en-us/filtering.htm",
    ),
    (
        "tableau_charttypes",
        "https://help.tableau.com/current/pro/desktop/en-us/charttypes.htm",
    ),
]

# ---------------------------------------------------------------------------
# GitHub sources
# ---------------------------------------------------------------------------

_GITHUB_DIRECT_URLS = [
    (
        "github_tableau_study",
        "https://raw.githubusercontent.com/mainkoon81/Study-Tableau/master/README.md",
        {"source": "github", "repo": "mainkoon81/Study-Tableau"},
    ),
]

_GITHUB_REPO_SCAN_URL = (
    "https://api.github.com/repos/PacktPublishing/Tableau-10-Complete-Reference/contents/"
)
_GITHUB_REPO_MAX_FILES = 10

# ---------------------------------------------------------------------------
# Fallback: 13 hardcoded knowledge chunks (guaranteed baseline)
# ---------------------------------------------------------------------------

_FALLBACK_CHUNKS = [
    {
        "id": "chart_bar",
        "text": (
            "Bar chart: Use for categorical comparison — one dimension on x-axis, one numeric measure on y-axis. "
            "Best when comparing discrete categories (products, regions, segments). "
            "NEVER use bar chart for time trends — use line_chart or area instead. "
            "Sort descending when user asks for 'top N' or 'ranking'."
        ),
    },
    {
        "id": "chart_line_area",
        "text": (
            "Line chart / Area chart: Use for trends over time. x_field must be a date or datetime field. "
            "Prefer line_chart for rate-of-change visibility, area for cumulative volume. "
            "Wrap date field in YEAR() for yearly, MONTH() for monthly on the columns shelf. "
            "Use color_field for a dimension breakdown (e.g., Region, Category)."
        ),
    },
    {
        "id": "chart_pie_treemap",
        "text": (
            "Pie chart: Use for part-to-whole with 6 or fewer categories. x_field=dimension (slices), y_field=measure (size). "
            "NEVER use pie for more than 8 categories — use treemap instead. "
            "Treemap: hierarchical part-to-whole with many categories. size=measure, detail=dimension. "
            "Keywords: share, percentage, proportion, repartition, distribution."
        ),
    },
    {
        "id": "chart_scatter_heatmap",
        "text": (
            "Scatter plot: Use for correlation between two numeric fields. Both axes are measures. "
            "Optional color_field as a dimension for grouping. Keywords: correlation, relationship, X vs Y. "
            "Heatmap: Use for intensity across two categorical dimensions. Square mark, measure as color intensity. "
            "x_field=first dimension (columns), y_field=second dimension (rows), color=aggregated measure."
        ),
    },
    {
        "id": "chart_combo_kpi",
        "text": (
            "Combo (dual-axis): Use when comparing two measures with different scales on the same chart. "
            "x_field=shared dimension, y_field=primary measure (bars), color_field=secondary measure (line). "
            "KPI: Use ONLY for a single aggregated number with NO dimension breakdown. "
            "x_field=the field to aggregate, y_field=empty string. "
            "If ANY dimension is needed, use bar_chart or text instead of kpi."
        ),
    },
    {
        "id": "chart_text_gantt",
        "text": (
            "Text table: Use when the user asks to 'show', 'list', or 'display' values by a dimension. "
            "x_field=row dimension (required), y_field=measure, color_field=optional column dimension. "
            "Gantt chart: Use for timelines and schedules. x_field=task/category dimension, "
            "y_field=duration measure, color_field=start date field (REQUIRED)."
        ),
    },
    {
        "id": "aggregation_rules",
        "text": (
            "Aggregation selection rules: "
            "SUM — default for monetary amounts, totals (Sales, Profit, Revenue). "
            "AVG — for rates, percentages, ratios, margins, scores, averages. "
            "COUNTD — for counting distinct entities (unique customers, unique products, unique orders). "
            "COUNT — for counting rows/records/occurrences/transactions. "
            "MIN/MAX — for extremes (earliest date, highest score, lowest price). "
            "Never aggregate dimension fields (strings, dates used as axis)."
        ),
    },
    {
        "id": "filter_categorical",
        "text": (
            "Categorical filters: "
            "eq (exact match) — 'only West', 'uniquement Furniture' → {field, op:'eq', value:'West'}. "
            "in (multiple values) — 'Furniture and Technology', 'regions East et West' → {field, op:'in', values:[...]}. "
            "Implicit filters: 'profitable'→Profit>0, 'growing'→growth_field>0, 'exclude returns'→Sales>0."
        ),
    },
    {
        "id": "filter_numeric_range",
        "text": (
            "Numeric filters: "
            "gt/gte/lt/lte — 'greater than 10000', 'at least 500' → {field, op:'gt', value:10000}. "
            "between — 'between 0 and 5000' → {field, op:'between', min:0, max:5000}. "
            "not_null — 'exclude empty values', 'sans valeurs nulles' → {field, op:'not_null'}."
        ),
    },
    {
        "id": "filter_date",
        "text": (
            "Date filters: "
            "year — 'in 2024', 'en 2023' → {field:[date_field], op:'year', value:2024}. "
            "quarter — 'Q3', 'third quarter' → {field:[date_field], op:'quarter', value:3}. "
            "month — 'in March', 'en janvier' → {field:[date_field], op:'month', value:3}. "
            "last_n_days/last_n_months — 'last 30 days', 'les 6 derniers mois' → {field, op:'last_n_months', value:6}."
        ),
    },
    {
        "id": "filter_ranking",
        "text": (
            "Ranking filters: "
            "top_n — 'top 10 customers', 'les 5 meilleurs' → {field:[x_field], op:'top_n', value:10, by:[y_field]}. "
            "bottom_n — '5 worst performers', 'les 5 pires' → {field:[x_field], op:'bottom_n', value:5, by:[y_field]}. "
            "When top_n is used, also set sort:'descending' on the viz intent."
        ),
    },
    {
        "id": "calc_field_margin_avg",
        "text": (
            "Calculated field patterns — financial metrics: "
            "Profit margin / taux de marge → SUM([Profit])/SUM([Sales]), datatype:real, role:measure. "
            "Average order value / panier moyen → SUM([Sales])/COUNTD([Order_ID]), datatype:real, role:measure. "
            "Profit per customer → SUM([Profit])/COUNTD([Customer]), datatype:real, role:measure. "
            "Percent of total / part du total → SUM([Sales])/TOTAL(SUM([Sales])), datatype:real, role:measure."
        ),
    },
    {
        "id": "calc_field_rank_running",
        "text": (
            "Calculated field patterns — analytical: "
            "Rank / classement → RANK(SUM([Sales])), datatype:integer, role:measure. "
            "Running sum / cumulative → RUNNING_SUM(SUM([Sales])), datatype:real, role:measure. "
            "YoY growth / croissance → (SUM([Sales])-LOOKUP(SUM([Sales]),-1))/ABS(LOOKUP(SUM([Sales]),-1)), datatype:real, role:measure. "
            "When a calculated field is created, use its name as y_field (or x_field for KPI)."
        ),
    },
]

# ---------------------------------------------------------------------------
# Golden examples — 30 curated Superstore question/intent pairs
# ---------------------------------------------------------------------------

_GOLDEN_EXAMPLES = [
    {
        "question": "Show total sales by category as a bar chart sorted from highest to lowest",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Category",
            "x_field": "Category", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Which region has the highest profit? Show me a bar chart",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Profit by Region",
            "x_field": "Region", "y_field": "Profit", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Rank sub-categories by sales from best to worst",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Sub-Category",
            "x_field": "Sub_Category", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show me how sales have evolved over the years as a line chart",
        "viz_intent": {
            "viz_type": "line_chart", "title": "Sales Over Time",
            "x_field": "Order_Date", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Monthly sales trend for 2023",
        "viz_intent": {
            "viz_type": "line_chart", "title": "Monthly Sales 2023",
            "x_field": "Order_Date", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Order_Date", "op": "year", "value": 2023}],
            "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show cumulative sales volume over time as an area chart",
        "viz_intent": {
            "viz_type": "area", "title": "Cumulative Sales Over Time",
            "x_field": "Order_Date", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "What is the proportion of sales by category? Show as a pie chart",
        "viz_intent": {
            "viz_type": "pie", "title": "Sales Distribution by Category",
            "x_field": "Category", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show the breakdown of sales by sub-category as a treemap",
        "viz_intent": {
            "viz_type": "treemap", "title": "Sales by Sub-Category",
            "x_field": "Sub_Category", "y_field": "Sales", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Is there a correlation between sales and profit? Scatter plot please",
        "viz_intent": {
            "viz_type": "scatter", "title": "Sales vs Profit Correlation",
            "x_field": "Sales", "y_field": "Profit", "color_field": "Category",
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show a heatmap of sales by category and region",
        "viz_intent": {
            "viz_type": "heatmap", "title": "Sales Heatmap by Category and Region",
            "x_field": "Category", "y_field": "Region", "color_field": "Sales",
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "What is the total sales amount?",
        "viz_intent": {
            "viz_type": "kpi", "title": "Total Sales",
            "x_field": "Sales", "y_field": "", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "How many distinct customers do we have in total?",
        "viz_intent": {
            "viz_type": "kpi", "title": "Total Distinct Customers",
            "x_field": "Customer", "y_field": "", "color_field": None,
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "COUNTD", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show the top 10 customers by total sales",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Top 10 Customers by Sales",
            "x_field": "Customer", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Customer", "op": "top_n", "value": 10, "by": "Sales"}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Which are the top 5 sub-categories by profit?",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Top 5 Sub-Categories by Profit",
            "x_field": "Sub_Category", "y_field": "Profit", "color_field": None,
            "filters": [{"field": "Sub_Category", "op": "top_n", "value": 5, "by": "Profit"}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show sales by region for the year 2023",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Region 2023",
            "x_field": "Region", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Order_Date", "op": "year", "value": 2023}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Sales by category in Q3",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Category Q3",
            "x_field": "Category", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Order_Date", "op": "quarter", "value": 3}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show sales for the Furniture category only",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Furniture Sales by Sub-Category",
            "x_field": "Sub_Category", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Category", "op": "eq", "value": "Furniture"}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Compare sales for Furniture and Technology categories by region",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Region — Furniture and Technology",
            "x_field": "Region", "y_field": "Sales", "color_field": "Category",
            "filters": [{"field": "Category", "op": "in", "values": ["Furniture", "Technology"]}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show only profitable products — products where profit is positive",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Profitable Sub-Categories",
            "x_field": "Sub_Category", "y_field": "Profit", "color_field": None,
            "filters": [{"field": "Profit", "op": "gt", "value": 0}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show profit margin by category",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Profit Margin by Category",
            "x_field": "Category", "y_field": "Taux de Marge", "color_field": None,
            "filters": [], "calculated_fields": [
                {"name": "Taux de Marge", "formula": "SUM([Profit])/SUM([Sales])",
                 "datatype": "real", "role": "measure"},
            ],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "What is the average order value by region?",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Average Order Value by Region",
            "x_field": "Region", "y_field": "Panier Moyen", "color_field": None,
            "filters": [], "calculated_fields": [
                {"name": "Panier Moyen", "formula": "SUM([Sales])/COUNTD([Order_ID])",
                 "datatype": "real", "role": "measure"},
            ],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show cumulative running sum of sales over time",
        "viz_intent": {
            "viz_type": "line_chart", "title": "Running Sum of Sales",
            "x_field": "Order_Date", "y_field": "Ventes Cumulées", "color_field": None,
            "filters": [], "calculated_fields": [
                {"name": "Ventes Cumulées", "formula": "RUNNING_SUM(SUM([Sales]))",
                 "datatype": "real", "role": "measure"},
            ],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show sales as bars and profit as a line on the same chart by category",
        "viz_intent": {
            "viz_type": "combo", "title": "Sales and Profit by Category",
            "x_field": "Category", "y_field": "Sales", "color_field": "Profit",
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Display a table of sales by category and region",
        "viz_intent": {
            "viz_type": "text", "title": "Sales by Category and Region",
            "x_field": "Category", "y_field": "Sales", "color_field": "Region",
            "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Now add a filter to show only the West region",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sales by Category — West Region",
            "x_field": "Category", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Region", "op": "eq", "value": "West"}],
            "calculated_fields": [],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "modify", "clarification_needed": None,
        },
    },
    {
        "question": "What do you mean by performance — profit total, profit margin, or sales ranking?",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "", "x_field": "", "y_field": "",
            "color_field": None, "filters": [], "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "clarify",
            "clarification_needed": "Que voulez-vous dire par performance: profit total, taux de marge, ou classement des ventes?",
        },
    },
    {
        "question": "Show year-over-year sales growth by region",
        "viz_intent": {
            "viz_type": "line_chart", "title": "YoY Sales Growth by Region",
            "x_field": "Order_Date", "y_field": "Croissance YoY", "color_field": "Region",
            "filters": [], "calculated_fields": [
                {"name": "Croissance YoY",
                 "formula": "(SUM([Sales])-LOOKUP(SUM([Sales]),-1))/ABS(LOOKUP(SUM([Sales]),-1))",
                 "datatype": "real", "role": "measure"},
            ],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Which are the 5 worst sub-categories by profit?",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Bottom 5 Sub-Categories by Profit",
            "x_field": "Sub_Category", "y_field": "Profit", "color_field": None,
            "filters": [{"field": "Sub_Category", "op": "bottom_n", "value": 5, "by": "Profit"}],
            "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Show sales trend for the last 6 months",
        "viz_intent": {
            "viz_type": "line_chart", "title": "Sales — Last 6 Months",
            "x_field": "Order_Date", "y_field": "Sales", "color_field": None,
            "filters": [{"field": "Order_Date", "op": "last_n_months", "value": 6}],
            "calculated_fields": [],
            "sort": None, "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
    {
        "question": "Rank sub-categories by profit margin from best to worst",
        "viz_intent": {
            "viz_type": "bar_chart", "title": "Sub-Category Profit Margin Ranking",
            "x_field": "Sub_Category", "y_field": "Classement Marge", "color_field": None,
            "filters": [], "calculated_fields": [
                {"name": "Classement Marge",
                 "formula": "RANK(SUM([Profit])/SUM([Sales]))",
                 "datatype": "integer", "role": "measure"},
            ],
            "sort": "descending", "aggregation": "SUM", "color_scheme": "tableau10",
            "action": "new", "clarification_needed": None,
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return slug[:max_len].strip("_")


def _chunk_html(slug: str, html: str, url: str) -> list[tuple[str, str, dict]]:
    """Parse HTML and return (id, content, metadata) chunks split at H2/H3 headings."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("rag_ingestion: beautifulsoup4 not installed — skipping HTML parse")
        return []

    soup = BeautifulSoup(html, "html.parser")
    chunks: list[tuple[str, str, dict]] = []

    for heading in soup.find_all(["h2", "h3"]):
        heading_text = heading.get_text(strip=True)
        if not heading_text:
            continue

        body_parts: list[str] = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ("h1", "h2", "h3"):
                break
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                body_parts.append(text)

        body = " ".join(body_parts)[:_MAX_CHUNK]
        content = f"{heading_text}\n{body}".strip()
        if len(content) < _MIN_CHUNK:
            continue

        chunk_id = f"{slug}_{_slugify(heading_text)}"
        chunks.append((chunk_id, content, {"source": "tableau_docs", "url": url}))

    return chunks


def _chunk_markdown(slug: str, markdown: str, meta: dict) -> list[tuple[str, str, dict]]:
    """Split markdown at ## / ### headings and return (id, content, metadata) chunks."""
    sections = re.split(r"\n(?=#{2,3} )", markdown)
    chunks: list[tuple[str, str, dict]] = []

    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("#").strip()
        if not heading:
            continue
        body = " ".join(ln.strip() for ln in lines[1:])[:_MAX_CHUNK]
        content = f"{heading}\n{body}".strip()
        if len(content) < _MIN_CHUNK:
            continue
        chunk_id = f"{slug}_{_slugify(heading)}"
        chunks.append((chunk_id, content, meta))

    return chunks


async def _fetch(client, url: str) -> str | None:
    """Fetch a URL with a 10s timeout. Returns text on 200, None otherwise."""
    try:
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        logger.warning("rag_ingestion: HTTP %d for %s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.warning("rag_ingestion: fetch failed for %s — %s", url, exc)
        return None


def _safe_upsert(col, ids: list[str], documents: list[str], metadatas: list[dict]) -> int:
    """Upsert documents into collection, swallowing errors. Returns count added."""
    if not ids:
        return 0
    try:
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)
    except Exception as exc:
        logger.warning("rag_ingestion: upsert failed — %s", exc)
        return 0


def _add_chunks(col, chunks: list[tuple[str, str, dict]]) -> int:
    if not chunks:
        return 0
    ids = [c[0] for c in chunks]
    docs = [c[1] for c in chunks]
    metas = [c[2] for c in chunks]
    return _safe_upsert(col, ids, docs, metas)


# ---------------------------------------------------------------------------
# Source ingestion helpers
# ---------------------------------------------------------------------------

async def _ingest_tableau_docs(client, rules_col) -> int:
    """Fetch 4 Tableau help pages, chunk by H2/H3, upsert into tableau_rules."""
    total = 0
    for slug, url in _TABLEAU_DOC_URLS:
        html = await _fetch(client, url)
        if html is None:
            continue
        chunks = _chunk_html(slug, html, url)
        added = _add_chunks(rules_col, chunks)
        logger.info("rag_ingestion: tableau_docs %s → %d chunks", slug, added)
        total += added
    return total


async def _ingest_github(client, rules_col) -> int:
    """Fetch GitHub markdown sources, chunk by ##/###, upsert into tableau_rules."""
    total = 0

    # Direct markdown URLs
    for slug, url, meta in _GITHUB_DIRECT_URLS:
        md = await _fetch(client, url)
        if md is None:
            continue
        chunks = _chunk_markdown(slug, md, meta)
        added = _add_chunks(rules_col, chunks)
        logger.info("rag_ingestion: github direct %s → %d chunks", slug, added)
        total += added

    # Repo scan for .md files
    repo_listing = await _fetch(client, _GITHUB_REPO_SCAN_URL)
    if repo_listing:
        try:
            items = json.loads(repo_listing)
            md_files = [
                item for item in items
                if isinstance(item, dict) and item.get("name", "").endswith(".md")
            ][:_GITHUB_REPO_MAX_FILES]

            for item in md_files:
                download_url = item.get("download_url")
                if not download_url:
                    continue
                md = await _fetch(client, download_url)
                if md is None:
                    continue
                slug = f"github_packt_{_slugify(item['name'])}"
                meta = {
                    "source": "github",
                    "repo": "PacktPublishing/Tableau-10-Complete-Reference",
                }
                chunks = _chunk_markdown(slug, md, meta)
                added = _add_chunks(rules_col, chunks)
                logger.info("rag_ingestion: github packt %s → %d chunks", item["name"], added)
                total += added
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("rag_ingestion: github repo scan parse failed — %s", exc)

    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_ingestion(tableau_rules_col, golden_examples_col) -> None:
    """One-time data ingestion pipeline. Safe to call on every startup — guards on count == 0.

    Args:
        tableau_rules_col:  ChromaDB collection for Tableau knowledge (from rag.get_tableau_rules_collection())
        golden_examples_col: ChromaDB collection for curated examples (from rag.get_golden_examples_collection())
    """
    # --- tableau_rules ---
    if tableau_rules_col is not None and tableau_rules_col.count() == 0:
        logger.info("rag_ingestion: seeding tableau_rules (first run)…")
        try:
            import httpx
            async with httpx.AsyncClient(
                headers={"User-Agent": "text-to-viz-agent/1.0"},
            ) as client:
                docs_added = await _ingest_tableau_docs(client, tableau_rules_col)
                github_added = await _ingest_github(client, tableau_rules_col)
                logger.info(
                    "rag_ingestion: external sources — docs=%d github=%d",
                    docs_added, github_added,
                )
        except ImportError:
            logger.warning("rag_ingestion: httpx not installed — skipping external sources")
        except Exception as exc:
            logger.warning("rag_ingestion: external ingestion failed — %s", exc)

        # Always apply fallback to ensure collection is non-empty
        if tableau_rules_col.count() == 0:
            added = _add_chunks(
                tableau_rules_col,
                [(c["id"], c["text"], {"source": "fallback"}) for c in _FALLBACK_CHUNKS],
            )
            logger.info("rag_ingestion: applied fallback seed — %d chunks", added)
        else:
            # Also upsert fallback chunks alongside external docs (they complement each other)
            _add_chunks(
                tableau_rules_col,
                [(c["id"], c["text"], {"source": "fallback"}) for c in _FALLBACK_CHUNKS],
            )
            logger.info(
                "rag_ingestion: tableau_rules seeded — total=%d", tableau_rules_col.count()
            )

    # --- golden_examples ---
    if golden_examples_col is not None and golden_examples_col.count() == 0:
        logger.info("rag_ingestion: seeding golden_examples (%d pairs)…", len(_GOLDEN_EXAMPLES))
        try:
            ids = [f"golden_{i:02d}" for i in range(len(_GOLDEN_EXAMPLES))]
            docs = [ex["question"] for ex in _GOLDEN_EXAMPLES]
            metas = [
                {
                    "viz_intent_json": json.dumps(ex["viz_intent"], ensure_ascii=False),
                    "judge_score": 1.0,
                }
                for ex in _GOLDEN_EXAMPLES
            ]
            added = _safe_upsert(golden_examples_col, ids, docs, metas)
            logger.info("rag_ingestion: golden_examples seeded — %d pairs", added)
        except Exception as exc:
            logger.warning("rag_ingestion: golden_examples seeding failed — %s", exc)
