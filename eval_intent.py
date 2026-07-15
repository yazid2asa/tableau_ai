"""Class #2 + #4 measurement harness — question → VizIntent accuracy + RAG A/B.

Unlike eval_harness.py (Phase 1: generate → publish → verify the filter on the
real Server), this harness measures the *upstream* decision quality: does the
LLM + deterministic post-correction pick the right `viz_type` and field roles for
a natural-language question? It runs the **real production intent path**:

    build_intent_prompt → call_llm(tools=TOOLS) → VizIntent
        → auto_correct_intent_fields → _validate_and_correct_intent

against a labelled question set covering every viz_type, and reports per-question
PASS/FAIL + an accuracy %. Two switches:

  --rag on|off    inject (or not) the RAG few-shot examples  → Class #4 A/B
  --provider P    google (default) | groq | openrouter       → measure the primary

Usage:
    python eval_intent.py                      # google, RAG on
    python eval_intent.py --rag off            # google, RAG off (for the A/B)
    python eval_intent.py --quick              # first 8 questions
    python eval_intent.py --gen-check          # also generate the table case & assert no SUM(dim)

LLM calls are spaced + retried with backoff (the gemini free tier rate-limits on
bursts); the OpenRouter fallback is disabled during a run so we measure the
primary model cleanly rather than the fallback's (poor) tool-use.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from schemas import VizIntent, DataSourceMetadata, FieldInfo, FieldType
from prompts import build_intent_prompt, TOOLS
from llm import call_llm

# ---------------------------------------------------------------------------
# Labelled question set — covers every viz_type + field-role + the table case.
# Built against the real `ventes` (Sales/Profit/Region/Category/Order Date/…)
# and `trips` (Trip Date/Status/Duration Hours/…) schemas.
# ---------------------------------------------------------------------------


@dataclass
class Q:
    question: str
    expect_viz: str
    note: str = ""
    expect_x_role: Optional[str] = None     # "dimension" | "measure"
    expect_y_role: Optional[str] = None     # "dimension" | "measure" | "" (none/kpi)
    expect_filter: Optional[bool] = None
    ds: str = "ventes"                       # which real datasource the question targets


QUESTIONS: list[Q] = [
    # time → line
    Q("Montre l'évolution des ventes par mois", "line_chart", "time trend → line", "dimension", "measure", ds="ventes"),
    Q("Monthly revenue trend over time", "line_chart", "time trend → line", "dimension", "measure", ds="ventes"),
    # part-to-whole → pie / treemap
    Q("Quelle est la répartition des ventes par catégorie ?", "pie", "share → pie", "dimension", "measure", ds="ventes"),
    Q("Breakdown of profit by category and sub-category", "treemap", "hierarchical → treemap", "dimension", "measure", ds="ventes"),
    # correlation → scatter
    Q("Show the correlation between sales and profit", "scatter", "correlation → scatter", "measure", "measure", ds="ventes"),
    # single number → kpi
    Q("Combien de ventes au total ?", "kpi", "single number → kpi", "measure", "", ds="ventes"),
    Q("What is the total profit overall?", "kpi", "single number → kpi", "measure", "", ds="ventes"),
    # two measures, different scales → combo
    Q("Compare sales and quantity by region", "combo", "two measures → combo", "dimension", "measure", ds="ventes"),
    # matrix → heatmap
    Q("Show a heatmap of sales by region and category", "heatmap", "explicit heatmap", "dimension", "dimension", ds="ventes"),
    # ranking → bar + top_n
    Q("Top 10 customers by sales", "bar_chart", "ranking → bar + top_n", "dimension", "measure", True, ds="ventes"),
    # default comparison → bar
    Q("Sales by region", "bar_chart", "default comparison → bar", "dimension", "measure", ds="ventes"),
    Q("Profit by segment", "bar_chart", "default comparison → bar", "dimension", "measure", ds="ventes"),
    # field-role / aggregation
    Q("Average discount by segment", "bar_chart", "AVG ratio", "dimension", "measure", ds="ventes"),
    Q("Nombre de clients distincts par région", "bar_chart", "COUNTD entities", "dimension", "measure", ds="ventes"),
    # the reported list/table defect → text, must NOT put a dimension in the measure slot
    Q("List sales and profit by category and region", "text", "table/list → text (2 dims + measures)", "dimension", None, ds="ventes"),
    Q("Table of sales by category", "text", "table → text", "dimension", None, ds="ventes"),
    # gantt (trips has Trip Date + Status + Duration Hours)
    Q("Gantt of trip duration by status", "gantt", "gantt", ds="trips"),
    # implicit filters
    Q("Produits rentables par catégorie", "bar_chart", "rentable → Profit>0 filter", "dimension", "measure", True, ds="ventes"),
    Q("Sales in 2024 by category", "bar_chart", "year filter", "dimension", "measure", True, ds="ventes"),
    Q("Orders with profit between 0 and 500", "bar_chart", "between filter", expect_filter=True, ds="ventes"),
]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class IResult:
    question: str
    note: str
    expect_viz: str
    got_viz: str = "?"
    got_x: str = ""
    got_y: str = ""
    viz_ok: bool = False
    role_ok: Optional[bool] = None
    filter_ok: Optional[bool] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.error:
            return False
        if not self.viz_ok:
            return False
        if self.role_ok is False:
            return False
        if self.filter_ok is False:
            return False
        return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _role_of(field_name: str, ds: DataSourceMetadata) -> Optional[str]:
    if not field_name:
        return None
    for f in ds.fields:
        if f.name.lower() == field_name.lower():
            return f.role
    return None


async def _call_with_backoff(messages, provider: str, model: Optional[str] = None, attempts: int = 6):
    delay = 6.0
    last = None
    for i in range(attempts):
        try:
            return await call_llm(messages, provider_override=provider, model_override=model, tools=TOOLS)
        except Exception as exc:
            last = exc
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                await asyncio.sleep(delay)
                delay = min(delay * 1.6, 30)
                continue
            raise
    raise last


async def run_intent_matrix(questions: list[Q], ds_by_name: dict[str, DataSourceMetadata],
                            available: list[DataSourceMetadata], rag_on: bool,
                            provider: str, spacing: float, model: Optional[str] = None) -> list[IResult]:
    from main import auto_correct_intent_fields, _validate_and_correct_intent
    rag_fetch = None
    if rag_on:
        from main import _fetch_rag_async
        rag_fetch = _fetch_rag_async

    results: list[IResult] = []
    for q in questions:
        ds = ds_by_name.get(q.ds) or available[0]
        res = IResult(question=q.question, note=q.note, expect_viz=q.expect_viz)
        try:
            rag_examples = await rag_fetch(q.question, ds.datasource_name) if rag_fetch else []
            messages = build_intent_prompt(q.question, ds, [], rag_examples=rag_examples,
                                           available_datasources=available)
            resp = await _call_with_backoff(messages, provider, model)
            if not resp.has_tool_call:
                res.error = "no tool_call (text response)"
                results.append(res)
                await asyncio.sleep(spacing)
                continue
            viz = VizIntent(**resp.first_tool_args)
            viz = auto_correct_intent_fields(viz, ds)
            viz = _validate_and_correct_intent(viz, available, q.question)

            res.got_viz = viz.viz_type
            res.got_x, res.got_y = viz.x_field, viz.y_field
            res.viz_ok = (viz.viz_type == q.expect_viz)
            if q.expect_x_role is not None:
                xr = _role_of(viz.x_field, ds)
                # unknown (calc field) → can't penalize
                res.role_ok = (xr is None) or (xr == q.expect_x_role)
            if q.expect_y_role is not None and res.role_ok is not False:
                if q.expect_y_role == "":
                    res.role_ok = (viz.y_field == "")
                elif q.expect_y_role == "dimension":
                    # heatmap: y is a 2nd dimension (unknown calc name also acceptable)
                    yr = _role_of(viz.y_field, ds)
                    res.role_ok = (res.role_ok in (None, True)) and (yr in ("dimension", None))
                else:  # expect a measure value-slot
                    yr = _role_of(viz.y_field, ds)
                    # measure, OR an unknown calc-field name, OR a dimension counted
                    # via COUNT/COUNTD (COUNTD(dim) is a valid value) → acceptable.
                    y_ok = (yr == "measure") or (yr is None) or (
                        yr == "dimension" and viz.aggregation in ("COUNT", "COUNTD"))
                    res.role_ok = (res.role_ok in (None, True)) and y_ok
            if q.expect_filter is not None:
                res.filter_ok = (len(viz.filters) > 0) == q.expect_filter
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {str(exc)[:120]}"
        results.append(res)
        await asyncio.sleep(spacing)
    return results


def print_report(results: list[IResult], label: str) -> str:
    lines = [f"\n{'='*100}", f"INTENT ACCURACY — {label}", "-"*100,
             f"{'Q':<46} {'EXPECT':<11} {'GOT':<11} {'VIZ':<4} {'ROLE':<5} {'FILT':<5}"]
    for r in results:
        viz = "OK" if r.viz_ok else "X"
        role = "-" if r.role_ok is None else ("OK" if r.role_ok else "X")
        filt = "-" if r.filter_ok is None else ("OK" if r.filter_ok else "X")
        q = (r.question[:44] + "..") if len(r.question) > 46 else r.question
        tail = f"  <- {r.error}" if r.error else ("" if r.ok else f"  (x={r.got_x} y={r.got_y})")
        lines.append(f"{q:<46} {r.expect_viz:<11} {r.got_viz:<11} {viz:<4} {role:<5} {filt:<5}{tail}")
    n = len(results)
    viz_ok = sum(1 for r in results if r.viz_ok)
    full_ok = sum(1 for r in results if r.ok)
    lines.append("-"*100)
    lines.append(f"  viz_type accuracy: {viz_ok}/{n} ({100*viz_ok/n:.0f}%)   |   full (viz+role+filter): {full_ok}/{n} ({100*full_ok/n:.0f}%)")
    lines.append("="*100)
    out = "\n".join(lines)
    print(out)
    return out


async def main_async(args) -> int:
    # Disable every OTHER provider during measurement so a primary-provider 429
    # is retried (with backoff) instead of silently producing another model's
    # output. Keyed on the measured provider, so `--provider openrouter` no
    # longer self-sabotages by blanking its own key (the historical B2 trap).
    if not args.allow_fallback:
        for prov, attr in (("openrouter", "openrouter_api_key"),
                           ("google", "google_api_key"),
                           ("mistral", "mistral_api_key"),
                           ("groq", "groq_api_key")):
            if prov != args.provider:
                setattr(settings, attr, "")

    # Real datasource schemas (mirror production); fall back to a built-in ventes/trips
    # schema if the Server is unreachable so the harness still measures intent.
    available: list[DataSourceMetadata] = []
    try:
        from tableau_server import signin, get_all_datasource_schemas
        await signin()
        schemas = await get_all_datasource_schemas()
        available = [d for d in schemas if d.datasource_name in ("ventes", "trips")]
    except Exception as exc:
        print(f"(Server fetch failed: {type(exc).__name__}; using built-in schema)  ")
    if not available:
        available = _builtin_schemas()
    ds_by_name = {d.datasource_name: d for d in available}

    questions = QUESTIONS[:8] if args.quick else QUESTIONS
    label = f"provider={args.provider} rag={args.rag}"
    print(f"Running {len(questions)} questions  ({label})  datasources={list(ds_by_name)}")

    t0 = time.time()
    results = await run_intent_matrix(questions, ds_by_name, available,
                                      rag_on=(args.rag == "on"), provider=args.provider,
                                      spacing=args.spacing, model=args.model)
    elapsed = time.time() - t0
    report = print_report(results, label + f"  ({elapsed:.0f}s, {elapsed/max(1,len(results)):.1f}s/q)")

    if args.gen_check:
        report += "\n" + _gen_check_table_case(ds_by_name.get("ventes") or available[0])
        print(report.rsplit("\n", 1)[-1])

    try:
        out = settings.output_dir / f"eval_intent_{args.provider}_rag-{args.rag}.txt"
        out.write_text(report, encoding="utf-8")
        print(f"\nReport: {out}")
    except Exception:
        pass
    n = len(results)
    return 0 if sum(1 for r in results if r.ok) == n else 1


def _builtin_schemas() -> list[DataSourceMetadata]:
    ventes = DataSourceMetadata(datasource_name="ventes", luid="ventes", fields=[
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sub Category", type=FieldType.STRING, role="dimension", local_name="Sub_Category"),
        FieldInfo(name="Segment", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Customer", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Order ID", type=FieldType.STRING, role="dimension", local_name="Order_ID"),
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension", local_name="Order_Date"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Quantity", type=FieldType.INTEGER, role="measure"),
    ])
    trips = DataSourceMetadata(datasource_name="trips", luid="trips", fields=[
        FieldInfo(name="Trip Date", type=FieldType.DATE, role="dimension", local_name="trip_date"),
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension", local_name="status"),
        FieldInfo(name="Duration Hours", type=FieldType.FLOAT, role="measure", local_name="duration_hours"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure", local_name="cost_eur"),
    ])
    return [ventes, trips]


def _gen_check_table_case(ds: DataSourceMetadata) -> str:
    """Generate the 'list by category and region' table case and assert the
    generated .twb does NOT aggregate a dimension (SUM(<dimension>))."""
    import lxml.etree as ET
    from twb_generator import generate_twb
    from main import _validate_and_correct_intent
    viz = VizIntent(viz_type="text", title="Sales and Profit by Category and Region",
                    x_field="Category", y_field="Region", color_field="Profit",
                    action="new", datasource_luid=ds.luid)
    # Mirror production: the correction runs BEFORE generate_twb.
    viz = _validate_and_correct_intent(viz, [ds], "list sales and profit by category and region")
    try:
        _fn, path = generate_twb(viz, ds, server_ds_content_url="ventes", server_ds_name="ventes")
    except Exception as exc:
        return f"GEN-CHECK table case: ERROR {exc}"
    txt = open(str(path), encoding="utf-8").read()
    try:
        os.remove(str(path))
    except OSError:
        pass
    dim_names = [f.local_name or f.name for f in ds.fields if f.role == "dimension"]
    bad = [d for d in dim_names if f"sum:{d.lower()}" in txt.lower() or f"SUM([{d}])" in txt]
    verdict = "FAIL — aggregates a dimension: " + str(bad) if bad else "PASS — no SUM(<dimension>)"
    return f"GEN-CHECK table case (y=Region dimension): {verdict}"


def main() -> int:
    p = argparse.ArgumentParser(description="Class #2/#4 intent-accuracy + RAG A/B harness")
    p.add_argument("--rag", choices=["on", "off"], default="on")
    p.add_argument("--provider", default=settings.llm_provider)
    p.add_argument("--model", default=None, help="model override (e.g. llama-3.3-70b-versatile for groq)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--spacing", type=float, default=5.0, help="seconds between LLM calls (rate-limit safety)")
    p.add_argument("--allow-fallback", action="store_true", help="don't disable OpenRouter fallback")
    p.add_argument("--gen-check", action="store_true", help="also generate the table case and assert no SUM(dimension)")
    args = p.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
