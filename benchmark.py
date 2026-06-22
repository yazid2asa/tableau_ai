"""
benchmark.py -- Model Evaluation Benchmark for Text-to-Viz Agent.

Runs 20 fixed questions against one or more LLM models via OpenRouter,
scores the VizIntent output, and prints a comparison table.

Usage:
    # Test current model (from .env MODEL_ID):
    python benchmark.py

    # Test specific models:
    python benchmark.py --models "google/gemini-2.0-flash-exp:free" "minimax/minimax-m2.5:free"

    # Quick test (5 questions only):
    python benchmark.py --quick
"""

import argparse
import asyncio
import json

from config import settings
from llm import call_llm
from prompts import build_intent_prompt
from schemas import (
    DataSourceMetadata,
    FieldInfo,
    FieldType,
    VizIntent,
)

# ---------------------------------------------------------------------------
# Standard test datasource with common fields
# ---------------------------------------------------------------------------

TEST_DATASOURCE = DataSourceMetadata(
    datasource_name="Benchmark_Sales",
    luid="bench-ds-001",
    fields=[
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension"),
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sub-Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Segment", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="City", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Product Name", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Customer Name", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Quantity", type=FieldType.INTEGER, role="measure"),
        FieldInfo(name="Discount", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure"),
    ],
)

# ---------------------------------------------------------------------------
# 20 benchmark questions (bilingual FR / EN)
# ---------------------------------------------------------------------------

BENCHMARK_QUESTIONS = [
    # --- Chart type selection (8 questions) ---
    {
        "question": "Montre l'evolution des ventes par mois",
        "expected_viz_type": "line_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "description": "Time series -> should pick line_chart, not bar",
    },
    {
        "question": "Quelle est la repartition des ventes par categorie ?",
        "expected_viz_type": "pie",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "description": "Distribution/repartition -> pie",
    },
    {
        "question": "Show the correlation between profit and sales",
        "expected_viz_type": "scatter",
        "expected_x_role": "measure",
        "expected_y_role": "measure",
        "description": "Correlation -> scatter",
    },
    {
        "question": "Combien de commandes au total ?",
        "expected_viz_type": "kpi",
        "expected_x_role": "measure",
        "expected_y_role": None,
        "description": "Single number -> kpi",
    },
    {
        "question": "Compare revenue and quantity by region",
        "expected_viz_type": "combo",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "description": "Two measures different scales -> combo",
    },
    {
        "question": "Top 10 des produits par chiffre d'affaires",
        "expected_viz_type": "bar_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "expected_has_filter": True,
        "description": "Ranking -> bar_chart with top_n filter",
    },
    {
        "question": "Show a heatmap of sales by region and category",
        "expected_viz_type": "heatmap",
        "expected_x_role": "dimension",
        "expected_y_role": "dimension",
        "description": "Explicit heatmap request",
    },
    {
        "question": "Breakdown of profit by category and sub-category",
        "expected_viz_type": "treemap",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "description": "Hierarchical breakdown -> treemap",
    },
    # --- Field role correctness (4 questions) ---
    {
        "question": "Ventes par region",
        "expected_viz_type": "bar_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "expected_aggregation": "SUM",
        "description": "Basic: dimension on x, measure on y",
    },
    {
        "question": "Average discount by segment",
        "expected_viz_type": "bar_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "expected_aggregation": "AVG",
        "description": "AVG aggregation for rate/ratio field",
    },
    {
        "question": "Nombre de clients par ville",
        "expected_viz_type": "bar_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "expected_aggregation": "COUNTD",
        "description": "Count of entities -> COUNTD",
    },
    {
        "question": "Monthly trend of order count",
        "expected_viz_type": "line_chart",
        "expected_x_role": "dimension",
        "expected_y_role": "measure",
        "description": "Time + count -> line_chart",
    },
    # --- Implicit filters (4 questions) ---
    {
        "question": "Produits rentables par categorie",
        "expected_viz_type": "bar_chart",
        "expected_has_filter": True,
        "description": "rentable -> implicit Profit > 0 filter",
    },
    {
        "question": "Sales in 2024 by quarter",
        "expected_viz_type": "bar_chart",
        "expected_has_filter": True,
        "description": "Year mention -> implicit year filter",
    },
    {
        "question": "Top 5 customers by revenue",
        "expected_viz_type": "bar_chart",
        "expected_has_filter": True,
        "description": "Top N -> implicit top_n filter",
    },
    {
        "question": "Montre les ventes des clients actifs",
        "expected_viz_type": "bar_chart",
        "expected_has_filter": True,
        "description": "actifs -> implicit status filter",
    },
    # --- Conversational continuity (4 questions) ---
    {
        "question": "Ajoute un filtre par region West",
        "expected_action": "modify",
        "description": "Modification keyword -> action=modify",
        "has_previous": True,
    },
    {
        "question": "Change le type en camembert",
        "expected_action": "modify",
        "description": "Change keyword -> modify",
        "has_previous": True,
    },
    {
        "question": "Montre le profit par segment",
        "expected_action": "new",
        "description": "New unrelated question -> action=new",
        "has_previous": True,
    },
    {
        "question": "Qu'est-ce que tu veux dire par region ?",
        "expected_action": "clarify",
        "description": "Ambiguous -> action=clarify",
        "has_previous": True,
    },
]

# ---------------------------------------------------------------------------
# VizIntent parsing helper
# ---------------------------------------------------------------------------


def parse_viz_intent(raw: str) -> VizIntent:
    """Parse LLM raw response into VizIntent."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(inner)
    data = json.loads(text)
    return VizIntent(**data)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Points per criterion
_PTS_VIZ_TYPE = 40
_PTS_ROLE = 10
_PTS_AGGREGATION = 15
_PTS_FILTER = 15
_PTS_ACTION = 10


def score_response(
    viz_intent: VizIntent,
    expected: dict,
    datasource: DataSourceMetadata,
) -> dict[str, int]:
    """Score a VizIntent against expected outputs. Returns dict of criteria -> points."""
    scores: dict[str, int] = {}

    # 1. Chart type (40 points)
    if "expected_viz_type" in expected:
        scores["viz_type"] = (
            _PTS_VIZ_TYPE if viz_intent.viz_type == expected["expected_viz_type"] else 0
        )

    # 2. Field roles (10 pts each)
    if "expected_x_role" in expected and viz_intent.x_field:
        x_info = next(
            (f for f in datasource.fields if f.name == viz_intent.x_field), None
        )
        scores["x_role"] = (
            _PTS_ROLE if (x_info and x_info.role == expected["expected_x_role"]) else 0
        )

    if "expected_y_role" in expected:
        if expected["expected_y_role"] is None:
            scores["y_role"] = _PTS_ROLE if viz_intent.y_field == "" else 0
        elif viz_intent.y_field:
            y_info = next(
                (f for f in datasource.fields if f.name == viz_intent.y_field), None
            )
            scores["y_role"] = (
                _PTS_ROLE
                if (y_info and y_info.role == expected["expected_y_role"])
                else 0
            )

    # 3. Aggregation (15 points)
    if "expected_aggregation" in expected:
        scores["aggregation"] = (
            _PTS_AGGREGATION
            if viz_intent.aggregation == expected["expected_aggregation"]
            else 0
        )

    # 4. Filters (15 points)
    if "expected_has_filter" in expected:
        has_filters = len(viz_intent.filters) > 0
        scores["has_filter"] = (
            _PTS_FILTER if has_filters == expected["expected_has_filter"] else 0
        )

    # 5. Action (10 points)
    if "expected_action" in expected:
        scores["action"] = (
            _PTS_ACTION if viz_intent.action == expected["expected_action"] else 0
        )

    return scores


def _max_for_criteria(scores: dict[str, int]) -> int:
    """Compute max possible score based on which criteria were evaluated."""
    max_pts = 0
    for key in scores:
        if key == "viz_type":
            max_pts += _PTS_VIZ_TYPE
        elif key in ("x_role", "y_role"):
            max_pts += _PTS_ROLE
        elif key == "aggregation":
            max_pts += _PTS_AGGREGATION
        elif key == "has_filter":
            max_pts += _PTS_FILTER
        elif key == "action":
            max_pts += _PTS_ACTION
    return max_pts


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    models: list[str],
    questions: list[dict],
    quick: bool = False,
) -> None:
    """Run benchmark for each model and print results."""

    if quick:
        questions = questions[:5]

    results: dict[str, dict] = {}  # model -> {total, max, details}

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model}")
        print(f"{'=' * 60}")

        model_results: dict = {"total": 0, "max": 0, "details": []}

        for i, q in enumerate(questions, 1):
            # Build previous intent for continuity questions
            previous_intent = None
            if q.get("has_previous"):
                previous_intent = VizIntent(
                    viz_type="bar_chart",
                    title="Previous Chart",
                    x_field="Category",
                    y_field="Sales",
                    aggregation="SUM",
                )

            messages = build_intent_prompt(
                q["question"],
                TEST_DATASOURCE,
                [],
                previous_intent=previous_intent,
                available_datasources=[TEST_DATASOURCE],
            )

            try:
                raw = await call_llm(messages, model_override=model)
                viz_intent = parse_viz_intent(raw)

                # Apply self-correction (lazy import to avoid circular / startup side-effects)
                from main import _validate_and_correct_intent

                viz_intent = _validate_and_correct_intent(
                    viz_intent, [TEST_DATASOURCE], q["question"]
                )

                scores = score_response(viz_intent, q, TEST_DATASOURCE)
                total = sum(scores.values())
                max_possible = _max_for_criteria(scores)

                status = "PASS" if (max_possible > 0 and total >= max_possible * 0.7) else ("FAIL" if max_possible > 0 else "SKIP")
                print(f"  [{status}] Q{i}: {q['description']} -- {total}/{max_possible}")

                if status == "FAIL":
                    print(
                        f"         Got: viz_type={viz_intent.viz_type}, "
                        f"x={viz_intent.x_field}, y={viz_intent.y_field}, "
                        f"action={viz_intent.action}"
                    )
                    print(
                        f"         Filters: {len(viz_intent.filters)}, "
                        f"Agg: {viz_intent.aggregation}"
                    )

                model_results["total"] += total
                model_results["max"] += max_possible
                model_results["details"].append(
                    {"q": i, "score": total, "max": max_possible, "status": status}
                )

            except Exception as exc:
                print(f"  [ERROR] Q{i}: {q['description']} -- {exc}")
                model_results["details"].append(
                    {"q": i, "score": 0, "max": 0, "status": "ERROR"}
                )

        results[model] = model_results

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<45} {'Score':>8} {'%':>6} {'Pass':>6}")
    print(f"{'-' * 45} {'-' * 8} {'-' * 6} {'-' * 6}")
    for model, r in sorted(
        results.items(), key=lambda x: x[1]["total"], reverse=True
    ):
        pct = (r["total"] / r["max"] * 100) if r["max"] > 0 else 0
        passed = sum(1 for d in r["details"] if d["status"] == "PASS")
        total_q = len(r["details"])
        print(
            f"{model:<45} {r['total']:>4}/{r['max']:<4} {pct:>5.1f}% "
            f"{passed:>3}/{total_q}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Viz Model Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to test (default: current MODEL_ID from .env)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 5 questions for quick testing",
    )
    args = parser.parse_args()

    models = args.models or [settings.model_id]

    asyncio.run(run_benchmark(models, BENCHMARK_QUESTIONS, quick=args.quick))
