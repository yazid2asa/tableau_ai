"""End-to-end evaluation harness for the Text-to-Viz agent (Step-0 safety net).

Unlike benchmark.py (which stops at the VizIntent JSON against a synthetic
datasource), this harness exercises the *generation + publish* path against the
**real published datasources** on the connected Tableau Server:

    VizIntent  →  generate_twb()/add_sheet_to_existing()  →  publish to Server
               →  (a) publish succeeds  (b) worksheet parses  (c) filter ACTUALLY
                  restricts the data (verified by reading the published view's CSV)

It is the regression guarantee for:
  * Class #1 — workbooks that pass twilize's XSD but crash Tableau's native parser
               at publish time ("... could not be parsed; the workbook may be
               malformed"). Only a real publish catches this.
  * Class #3 — filters/calc fields that show a card but do not restrict the data.
               Verified by pulling the published view's summary CSV and asserting
               the data satisfies the requested predicate.

Three tiers, each independently switchable:
  Tier 1 (always)        structural XML checks on the generated .twb (fast, offline)
  Tier 2 (--publish)     publish each .twb to the Server — catches native-parse crash
  Tier 3 (--verify-data) read the published view CSV and assert the filter applied

Single command:
    py -3 eval_harness.py                 # tier 1+2+3, full matrix, then cleanup
    py -3 eval_harness.py --no-publish     # tier 1 only (offline, fast iteration)
    py -3 eval_harness.py --matrix filter  # only the filter-operator matrix
    py -3 eval_harness.py --case combo_filter_year_2025 --keep
    py -3 eval_harness.py --rag off        # (Phase 2 toggle; no effect on generation)

A representative slice is wired into pytest via tests/test_eval_harness.py
(tier-1 structural checks only — no network).
"""
from __future__ import annotations

import argparse
import asyncio
import csv as csvmod
import io
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lxml.etree as etree

from config import settings
from schemas import VizIntent, FilterSpec, DataSourceMetadata
from twb_generator import (
    generate_twb,
    add_sheet_to_existing,
    modify_sheet_in_existing,
    _detect_linking_fields,
)

# ---------------------------------------------------------------------------
# Tableau Server helpers (read-only extras layered on tableau_server.py)
# ---------------------------------------------------------------------------

_TSC_AVAILABLE = True
try:
    from tableau_server import (
        signin,
        get_all_datasource_schemas,
        get_datasource_content_url,
        publish_workbook,
        _ensure_signed_in,
    )
except Exception:  # pragma: no cover
    _TSC_AVAILABLE = False


def _read_view_csv(workbook_luid: str, sheet_name: Optional[str] = None) -> str:
    """Read a published view's summary data as CSV text (Tier-3 ground truth)."""
    server = _ensure_signed_in()
    wb = server.workbooks.get_by_id(workbook_luid)
    server.workbooks.populate_views(wb)
    view = None
    if sheet_name:
        view = next((v for v in wb.views if v.name == sheet_name), None)
    if view is None:
        view = wb.views[-1] if wb.views else None
    if view is None:
        raise RuntimeError("no views on workbook")
    server.views.populate_csv(view)
    return b"".join(view.csv).decode("utf-8", errors="replace")


def _delete_workbook(workbook_luid: str) -> None:
    server = _ensure_signed_in()
    server.workbooks.delete(workbook_luid)


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One generation step. `verify` asserts on the published view CSV (Tier 3)."""
    viz: VizIntent
    # The user's original question, fed to filter-value correction so it can recover
    # the member the user literally wrote when the LLM translated it (FIX-054b,
    # "West" emitted for a question that says "OUest"). None → no recovery context.
    question: Optional[str] = None
    verify: Optional[Callable[[list[dict]], Optional[str]]] = None  # returns failure reason or None
    expect_sheets: int = 1
    # When True, also publish an UNFILTERED twin of this turn's viz and assert the
    # filtered result differs from it. Used for measure-range filters (gt/lt/between)
    # which restrict at row level — the per-row aggregate changes even when the set
    # of categories does not, so "differs from unfiltered" is the correct proof that
    # the filter is active (vs. a wrong assumption about aggregate-level semantics).
    baseline_compare: bool = False
    # A filtered view that renders ZERO marks is a FAIL (the filter blanked the chart,
    # e.g. a mis-cased member value — FIX-054), UNLESS emptiness is legitimately
    # expected here (e.g. a relative-date window with no data in range).
    allow_empty: bool = False
    # In-place modify (the production `action="modify"` path): title of the existing
    # sheet this turn REPLACES via modify_sheet_in_existing. None → normal add-sheet.
    modify_of: Optional[str] = None


@dataclass
class Case:
    name: str
    group: str               # viz | filter | multiturn | blend
    turns: list[Turn]
    needs_blend: bool = False
    secondary_name: Optional[str] = None  # blend secondary datasource name


@dataclass
class CaseResult:
    name: str
    group: str
    structural: str = "?"     # PASS | FAIL
    publish: str = "-"        # PASS | FAIL | SKIP
    data: str = "-"           # VERIFIED | CONTRADICTED | UNVERIFIED | SKIP
    reason: str = ""
    intent_summary: str = ""

    @property
    def ok(self) -> bool:
        if self.structural == "FAIL":
            return False
        if self.publish == "FAIL":
            return False
        if self.data == "CONTRADICTED":
            return False
        return True


# ---------------------------------------------------------------------------
# Tier 1 — structural XML validation (offline, always)
# ---------------------------------------------------------------------------


def _structural_check(twb_path: str, all_intents: list[VizIntent]) -> Optional[str]:
    """Parse the generated .twb and assert the invariants that, when violated,
    crash Tableau's native parser or silently drop the filter. Returns a failure
    reason string, or None if every invariant holds."""
    try:
        root = etree.parse(twb_path).getroot()
    except Exception as exc:
        return f"XML not well-formed: {exc}"

    worksheets = root.findall(".//worksheets/worksheet")
    names = [w.get("name", "") for w in worksheets]

    # (A) Duplicate worksheet names → "could not be parsed; workbook may be malformed"
    if len(names) != len(set(names)):
        dup = [n for n in names if names.count(n) > 1]
        return f"duplicate worksheet name(s): {sorted(set(dup))}"

    # (B) Empty worksheet (no <view> or no marks) → often a half-built sheet
    for w in worksheets:
        if w.find(".//view") is None:
            return f"worksheet '{w.get('name')}' has no <view>"

    # (C) quantitative date instance still ending in :ok] (FIX-005 regression)
    for ci in root.iter("column-instance"):
        if ci.get("type") == "quantitative" and (ci.get("name") or "").endswith(":ok]"):
            return f"quantitative column-instance still ':ok]': {ci.get('name')}"

    # (D) same instance name declared with two different types (combo+date hazard)
    inst_types: dict[str, set] = {}
    for ci in root.iter("column-instance"):
        nm = ci.get("name")
        if nm:
            inst_types.setdefault(nm, set()).add(ci.get("type"))
    for nm, types in inst_types.items():
        if len(types) > 1:
            return f"column-instance '{nm}' declared with conflicting types {sorted(types)}"

    # (E) every requested filter must produce a <filter> AND a matching <slices> entry
    for w in worksheets:
        view = w.find(".//view")
        if view is None:
            continue
        filters = view.findall("filter")
        if not filters:
            continue
        slice_cols = {c.text for c in view.findall("slices/column") if c.text}
        for f in filters:
            col = f.get("column")
            if col and col not in slice_cols:
                return (f"worksheet '{w.get('name')}': filter on {col} "
                        f"has no matching <slices><column> (filter card shows but data unfiltered)")
    return None


# ---------------------------------------------------------------------------
# Tier 3 — data-verification predicates
# ---------------------------------------------------------------------------


def _parse_csv(text: str) -> list[dict]:
    return list(csvmod.DictReader(io.StringIO(text)))


def _num(s: str) -> float:
    return float(str(s).replace(",", "").replace('"', "").strip() or 0)


def _col(rows: list[dict], substr: str) -> Optional[str]:
    """Find the CSV column whose header contains substr (case-insensitive)."""
    for k in (rows[0].keys() if rows else []):
        if substr.lower() in k.lower():
            return k
    return None


# ---------------------------------------------------------------------------
# Matrix builders (grounded in the real `trips` schema)
# ---------------------------------------------------------------------------

# Discovered from the live data (probe): Status ∈ {Cancelled, Completed, Delayed,
# In Progress}; Trip Date years ∈ {2024, 2025}.
STATUS_VALUES = ["Cancelled", "Completed", "Delayed", "In Progress"]
DATA_YEARS = [2024, 2025]


def _verify_only_status(allowed: set[str]):
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (cannot confirm filter)"
        col = _col(rows, "Status")
        if not col:
            return "no Status column in view CSV"
        seen = {r[col] for r in rows}
        extra = seen - allowed
        if extra:
            return f"rows present for non-selected status values: {sorted(extra)}"
        return None
    return v


def _verify_excludes_status(excluded: set[str]):
    """FIX-055: an exclusion filter must remove the excluded members AND keep
    the others (an over-restricting rewrite that empties the chart is a FAIL
    via the non-empty guard; a no-op rewrite leaves the excluded members)."""
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (exclusion over-restricted)"
        col = _col(rows, "Status")
        if not col:
            return "no Status column in view CSV"
        seen = {r[col] for r in rows}
        leaked = seen & excluded
        if leaked:
            return f"excluded status values still present: {sorted(leaked)} (exclude filter is a no-op)"
        return None
    return v


def _verify_date_between(min_iso: str, max_iso: str):
    """FIX-056: every returned date must fall inside [min_iso, max_iso]."""
    import datetime as _dt
    lo, hi = _dt.date.fromisoformat(min_iso), _dt.date.fromisoformat(max_iso)
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (date_range may have over-restricted)"
        col = _col(rows, "Trip Date") or _col(rows, "Date")
        if not col:
            return "no date column in view CSV"
        bad = [r[col] for r in rows
               if not (lo <= (_to_date(r[col]) or lo) <= hi)]
        if bad:
            return f"rows outside [{min_iso}, {max_iso}]: {bad[:5]}"
        return None
    return v


def _verify_year(year: int):
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (year filter may have over-restricted)"
        col = _col(rows, "Trip Date") or _col(rows, "Year")
        if not col:
            return "no date/year column in view CSV"
        bad = [r[col] for r in rows if str(year) not in str(r[col])]
        if bad:
            return f"rows outside year {year}: {bad[:5]}"
        return None
    return v


def _date_parts(s: str) -> tuple[int, int]:
    """Best-effort (month, year) from a Tableau CSV date cell like '6/1/2024' or '2024'."""
    s = str(s).strip().strip('"')
    if "/" in s:
        p = s.split("/")
        return int(p[0]), int(p[-1])
    if "-" in s:
        p = s.split("-")
        return int(p[1]) if len(p) > 1 else 0, int(p[0])
    return 0, int(s) if s.isdigit() else 0


def _verify_month_part(month: int):
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned"
        col = _col(rows, "Trip Date") or _col(rows, "Month")
        if not col:
            return "no date column in view CSV"
        bad = [r[col] for r in rows if _date_parts(r[col])[0] != month]
        if bad:
            return f"rows outside month {month}: {bad[:5]}"
        return None
    return v


def _verify_quarter_part(q: int):
    months = {1: {1, 2, 3}, 2: {4, 5, 6}, 3: {7, 8, 9}, 4: {10, 11, 12}}[q]
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned"
        col = _col(rows, "Trip Date") or _col(rows, "Quarter")
        if not col:
            return "no date column in view CSV"
        bad = [r[col] for r in rows if _date_parts(r[col])[0] not in months]
        if bad:
            return f"rows outside quarter {q}: {bad[:5]}"
        return None
    return v


def _verify_blend_vehicle_type():
    """Blend is ENGAGED iff the secondary dimension (Vehicle Type) resolved to real
    type values (Truck/Car/…) — i.e. NOT the broken-link fallback ('*' axis) and NOT
    a substitution to the linking key. The mark granularity (one mark per vehicle_id,
    via the FIX-043 Detail LOD) is the documented blend-link design, so we assert on
    the DISTINCT vehicle types, not the raw row count."""
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (blend not engaged)"
        col = _col(rows, "Vehicle Type")
        if not col:
            return f"no 'Vehicle Type' column (blend fell back / substituted link key): {list(rows[0].keys())}"
        distinct = {r[col] for r in rows if r[col]}
        if any(d == "*" for d in distinct):
            return "Vehicle Type axis is '*' (broken blend link)"
        if not (1 <= len(distinct) <= 12):
            return f"{len(distinct)} distinct Vehicle Type values — expected the handful of real types"
        return None
    return v


def _verify_blend_avg_by_brand():
    """FIX-062 (measured): a secondary-measure blend joins per linking value —
    the linking field MUST be in the view (Detail LOD). The no-LOD variant was
    CONTRADICTED here: every brand showed the same GLOBAL average (241.95).
    Guards BOTH failure modes: a broken link ('*' axis / missing Brand column)
    and the silent global-average fake (one repeated value everywhere).
    Per-vehicle mark granularity (more rows than distinct brands) is the
    documented Tableau blending constraint, NOT a failure."""
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (blend not engaged)"
        col = _col(rows, "Brand")
        if not col:
            return f"no 'Brand' column: {list(rows[0].keys())}"
        brands = {r[col] for r in rows if r[col]}
        if any(b == "*" for b in brands):
            return "Brand axis is '*' (broken blend link)"
        if len(brands) < 2:
            return f"only {len(brands)} brand value(s) — blend not joined"
        mcol = next((k for k in rows[0].keys() if k != col), None)
        if mcol:
            vals = {str(r[mcol]).strip() for r in rows}
            if len(vals) <= 1:
                return f"single repeated value {vals} — measure not joined (global aggregate)"
        return None
    return v


def _verify_blend_total_avg():
    """FIX-063: 'AVG by cross-DS dimension' renders the TRUE category value on
    every mark — the TOTAL(<agg>) wrap computed along the linking field. The
    per-brand value must be CONSTANT across that brand's per-vehicle marks
    (raw per-vehicle averages would differ) and vary between brands (a global
    average would repeat one value everywhere). Live-proven equal to the exact
    VDS oracle on 16/16 brands (2026-07-08); this verifier keeps the invariant
    without re-querying VDS (rate-limit safe)."""
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (blend not engaged)"
        bcol = _col(rows, "Brand")
        vcol = _col(rows, "(AVG)")
        if not bcol:
            return f"no 'Brand' column: {list(rows[0].keys())}"
        if not vcol:
            return f"no TOTAL-wrap '(AVG)' calc column: {list(rows[0].keys())}"
        if any(r[bcol] == "*" for r in rows):
            return "Brand axis is '*' (broken blend link)"
        per_brand: dict[str, set[float]] = {}
        for r in rows:
            try:
                per_brand.setdefault(r[bcol], set()).add(round(_num(r[vcol]), 2))
            except ValueError:
                return f"non-numeric value in {vcol}: {r[vcol]!r}"
        multi = {b: sorted(vs) for b, vs in per_brand.items() if len(vs) > 1}
        if multi:
            some = dict(list(multi.items())[:3])
            return f"per-brand value not constant (TOTAL scope wrong): {some}"
        distinct = {next(iter(vs)) for vs in per_brand.values()}
        if len(per_brand) > 1 and len(distinct) <= 1:
            return f"one global value {distinct} repeated — TOTAL partition wrong"
        return None
    return v


def _rel_date_start(days: int = 0, months: int = 0) -> str:
    """ISO start date of a relative-date window ending today (for verification).

    FIX-059 emits LIVE Tableau relative-date filters, whose month periods start
    at the FIRST day of the month N months back (Tableau period semantics), so
    the verifier's window uses day=1 for months. Day windows are exact."""
    import datetime as _dt
    today = _dt.date.today()
    if months:
        total = (today.year * 12 + (today.month - 1)) - months
        y, m = divmod(total, 12)
        return _dt.date(y, m + 1, 1).isoformat()
    return (today - _dt.timedelta(days=days)).isoformat()


def _to_date(s: str):
    """Parse a Tableau CSV date cell ('6/1/2024' M/D/Y, or '2024-06-01', or '2024')."""
    import datetime as _dt
    s = str(s).strip().strip('"')
    if "/" in s:
        mo, dy, yr = (s.split("/") + ["1", "1"])[:3]
        try:
            return _dt.date(int(yr), int(mo), int(dy))
        except ValueError:
            return None
    if "-" in s:
        p = s.split("-")
        try:
            return _dt.date(int(p[0]), int(p[1]) if len(p) > 1 else 1, int(p[2]) if len(p) > 2 else 1)
        except ValueError:
            return None
    if s.isdigit():
        return _dt.date(int(s), 1, 1)
    return None


def _verify_relative_date(min_iso: str):
    """For relative-date filters: every returned date must be >= min_iso.
    Empty result is acceptable (the window may legitimately contain no data).
    A no-op filter (values:[]) returns the FULL range incl. dates before the
    window → caught here as CONTRADICTED."""
    import datetime as _dt
    start = _dt.date.fromisoformat(min_iso)
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return None  # restricted to empty window — acceptable
        col = _col(rows, "Trip Date") or _col(rows, "Year")
        if not col:
            return "no date column in view CSV"
        bad = [r[col] for r in rows if (_to_date(r[col]) or start) < start]
        if bad:
            return f"rows before window start {min_iso}: {bad[:5]} (filter is a no-op)"
        return None
    return v


def _verify_max_rows(n: int):
    def v(rows: list[dict]) -> Optional[str]:
        if len(rows) > n:
            return f"top/bottom-{n} returned {len(rows)} rows (> {n})"
        if not rows:
            return "no rows returned"
        return None
    return v


def _verify_measure_bound(measure_substr: str, lo: Optional[float], hi: Optional[float]):
    def v(rows: list[dict]) -> Optional[str]:
        if not rows:
            return "no rows returned (range filter may have over-restricted)"
        col = _col(rows, measure_substr)
        if not col:
            return f"no {measure_substr} column in view CSV"
        for r in rows:
            try:
                x = _num(r[col])
            except Exception:
                continue
            if lo is not None and x < lo - 1e-6:
                return f"value {x} < min {lo}"
            if hi is not None and x > hi + 1e-6:
                return f"value {x} > max {hi}"
        return None
    return v


def build_matrix(trips_luid: str, which: str) -> list[Case]:
    """Build the deterministic test matrix from the real `trips` schema.

    Intents are hand-built (not LLM-derived) so the harness isolates the
    generation/publish/filter classes (#1, #3) from LLM intent variance (#2)."""
    DS = trips_luid
    cases: list[Case] = []

    def viz(vt, title, x, y="", color=None, filters=None, agg="SUM", sort=None):
        return VizIntent(
            viz_type=vt, title=title, x_field=x, y_field=y, color_field=color,
            filters=filters or [], aggregation=agg, sort=sort,
            action="new", datasource_luid=DS,
        )

    # ----- VIZ-TYPE matrix (every supported viz_type) -----
    if which in ("all", "viz"):
        cases += [
            Case("viz_bar", "viz", [Turn(viz("bar_chart", "Bench Bar Cost by Status", "Status", "Cost Eur"))]),
            Case("viz_line", "viz", [Turn(viz("line_chart", "Bench Line Cost by Date", "Trip Date", "Cost Eur"))]),
            Case("viz_pie", "viz", [Turn(viz("pie", "Bench Pie Cost by Status", "Status", "Cost Eur"))]),
            Case("viz_scatter", "viz", [Turn(viz("scatter", "Bench Scatter Distance vs Cost", "Distance Km", "Cost Eur"))]),
            Case("viz_area", "viz", [Turn(viz("area", "Bench Area Cost by Date", "Trip Date", "Cost Eur"))]),
            Case("viz_heatmap", "viz", [Turn(viz("heatmap", "Bench Heatmap Status x Priority", "Status", "Priority", color="Cost Eur"))]),
            Case("viz_treemap", "viz", [Turn(viz("treemap", "Bench Treemap Cost by City", "Destination City", "Cost Eur"))]),
            Case("viz_text", "viz", [Turn(viz("text", "Bench Text Cost by Status", "Status", "Cost Eur"))]),
            Case("viz_gantt", "viz", [Turn(viz("gantt", "Bench Gantt Duration by Status", "Status", "Duration Hours", color="Trip Date"))]),
            Case("viz_combo", "viz", [Turn(viz("combo", "Bench Combo Cost+Fuel by Date", "Trip Date", "Cost Eur", color="Fuel Consumed Liters"))]),
            Case("viz_kpi", "viz", [Turn(viz("kpi", "Bench KPI Total Cost", "Cost Eur"))]),
        ]

    # ----- FILTER-OPERATOR matrix (all 15 operators) -----
    if which in ("all", "filter"):
        def fcase(name, op, **fkw):
            extra = {k: v for k, v in fkw.items()
                     if k not in ("x", "y", "verify", "field", "by", "baseline_compare", "allow_empty")}
            x = fkw.get("x", "Status")
            y = fkw.get("y", "Cost Eur")
            fld = fkw.get("field", "Status")
            spec = FilterSpec(field=fld, op=op, by=fkw.get("by"), **extra)
            return Case(name, "filter", [Turn(
                viz("bar_chart", f"Bench Filter {name}", x, y, filters=[spec]),
                verify=fkw.get("verify"),
                baseline_compare=fkw.get("baseline_compare", False),
                allow_empty=fkw.get("allow_empty", False),
            )])

        cases += [
            fcase("flt_eq", "eq", field="Status", value="Completed", verify=_verify_only_status({"Completed"})),
            fcase("flt_in", "in", field="Status", values=["Completed", "Delayed"], verify=_verify_only_status({"Completed", "Delayed"})),
            # measure-range filters are row-level → assert the result DIFFERS from the
            # unfiltered twin (the only semantics-agnostic proof the filter is active).
            fcase("flt_gt", "gt", x="Destination City", field="Cost Eur", value=500, baseline_compare=True),
            fcase("flt_gte", "gte", x="Destination City", field="Cost Eur", value=500, baseline_compare=True),
            fcase("flt_lt", "lt", x="Destination City", field="Cost Eur", value=500, baseline_compare=True),
            fcase("flt_lte", "lte", x="Destination City", field="Cost Eur", value=500, baseline_compare=True),
            fcase("flt_between", "between", x="Destination City", field="Cost Eur", min=500, max=2000, baseline_compare=True),
            fcase("flt_year", "year", x="Trip Date", field="Trip Date", value=2025, verify=_verify_year(2025)),
            fcase("flt_quarter", "quarter", x="Trip Date", field="Trip Date", value=1, verify=_verify_quarter_part(1)),
            fcase("flt_month", "month", x="Trip Date", field="Trip Date", value=6, verify=_verify_month_part(6)),
            fcase("flt_last_n_days", "last_n_days", x="Trip Date", field="Trip Date", value=400,
                  verify=_verify_relative_date(_rel_date_start(days=400)), allow_empty=True),
            fcase("flt_last_n_months", "last_n_months", x="Trip Date", field="Trip Date", value=12,
                  verify=_verify_relative_date(_rel_date_start(months=12)), allow_empty=True),
            fcase("flt_top_n", "top_n", x="Destination City", field="Destination City", value=5, by="Cost Eur", verify=_verify_max_rows(5)),
            fcase("flt_bottom_n", "bottom_n", x="Destination City", field="Destination City", value=5, by="Cost Eur", verify=_verify_max_rows(5)),
            fcase("flt_not_null", "not_null", x="Status", field="Status", baseline_compare=False),
            # FIX-055 — exclusion operators ("sans les annulés" / "exclude X")
            fcase("flt_neq", "neq", field="Status", value="Completed",
                  verify=_verify_excludes_status({"Completed"})),
            fcase("flt_not_in", "not_in", field="Status", values=["Completed", "Delayed"],
                  verify=_verify_excludes_status({"Completed", "Delayed"})),
            # FIX-056 — explicit date span ("de mars à juin 2025")
            fcase("flt_date_range", "date_range", x="Trip Date", field="Trip Date",
                  date_min="2025-03-01", date_max="2025-06-30",
                  verify=_verify_date_between("2025-03-01", "2025-06-30")),
        ]

    # ----- MULTI-TURN matrix (create → add filter; the reported crash path) -----
    if which in ("all", "multiturn"):
        # The exact reported flow: combo cost+fuel by date, THEN year=2025 filter.
        combo = viz("combo", "Cout et consommation de carburant par date de voyage",
                    "Trip Date", "Cost Eur", color="Fuel Consumed Liters")
        combo_filtered = viz("combo", "Cout et consommation de carburant par date de voyage",
                             "Trip Date", "Cost Eur", color="Fuel Consumed Liters",
                             filters=[FilterSpec(field="Trip Date", op="year", value=2025)])
        cases.append(Case("multiturn_combo_year_2025", "multiturn", [
            Turn(combo, expect_sheets=1),
            Turn(combo_filtered, verify=_verify_year(2025), expect_sheets=2),
        ]))

        bar = viz("bar_chart", "Bench MT Bar Cost by Status", "Status", "Cost Eur")
        bar_filtered = viz("bar_chart", "Bench MT Bar Cost by Status", "Status", "Cost Eur",
                           filters=[FilterSpec(field="Status", op="eq", value="Completed")])
        cases.append(Case("multiturn_bar_eq", "multiturn", [
            Turn(bar, expect_sheets=1),
            Turn(bar_filtered, verify=_verify_only_status({"Completed"}), expect_sheets=2),
        ]))

        # FIX-012 production flow: "filtre par X" on the CURRENT chart → in-place
        # modify (modify_sheet_in_existing), same datasource. The workbook must
        # still hold ONE sheet and the filter must verifiably restrict the data.
        mod_bar = viz("bar_chart", "Bench MT Modify In Place", "Status", "Cost Eur")
        mod_bar_filtered = viz("bar_chart", "Bench MT Modify In Place", "Status", "Cost Eur",
                               filters=[FilterSpec(field="Status", op="eq", value="Completed")])
        cases.append(Case("multiturn_modify_filter_eq", "multiturn", [
            Turn(mod_bar, expect_sheets=1),
            Turn(mod_bar_filtered, modify_of="Bench MT Modify In Place",
                 verify=_verify_only_status({"Completed"}), expect_sheets=1),
        ]))

    return cases


def build_blend_case(trips: DataSourceMetadata, vehicles: DataSourceMetadata) -> Optional[Case]:
    """Blend: Cargo Weight (trips) by Vehicle Type (vehicles), linked on Vehicle Id."""
    linking = _detect_linking_fields(trips, vehicles)
    if not linking:
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Bench Blend Cargo by Vehicle Type",
        x_field="Vehicle Type", y_field="Cargo Weight Kg",
        action="new", datasource_luid=trips.luid,
        secondary_datasource_luid=vehicles.luid,
    )
    return Case("blend_cargo_by_vehicle_type", "blend",
                [Turn(v, verify=_verify_blend_vehicle_type())],
                needs_blend=True, secondary_name=vehicles.datasource_name)


def build_blend_measure_case(trips: DataSourceMetadata,
                             vehicles: DataSourceMetadata) -> Optional[Case]:
    """FIX-062: secondary-measure blend — vehicles primary (Brand on x),
    AVG(Cost Eur) blended in from trips (the 'Coût moyen par marque' shape).
    Measured: the linking field MUST be in the view (Detail LOD) or every
    brand silently shows the same GLOBAL average; the verifier guards both
    the broken-link and the global-average failure modes."""
    linking = _detect_linking_fields(vehicles, trips)
    if not linking or not any(f.name == "Brand" for f in vehicles.fields):
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Bench Blend Avg Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="AVG",
        action="new", datasource_luid=vehicles.luid,
        secondary_datasource_luid=trips.luid,
    )
    return Case("blend_secondary_measure_avg", "blend",
                [Turn(v, verify=_verify_blend_avg_by_brand())],
                needs_blend=True, secondary_name=trips.datasource_name)


def build_blend_total_case(trips: DataSourceMetadata,
                           vehicles: DataSourceMetadata) -> Optional[Case]:
    """FIX-063: non-additive aggregation by a cross-DS dimension — 'coût moyen
    des voyages par marque'. The TOTAL(AVG) wrap + compute-using(linking) +
    breakdown=off must render one correct mark per brand instead of stacked
    per-vehicle averages."""
    linking = _detect_linking_fields(trips, vehicles)
    if not linking or not any(f.name == "Brand" for f in vehicles.fields):
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Bench Blend Total Avg Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="AVG",
        action="new", datasource_luid=trips.luid,
        secondary_datasource_luid=vehicles.luid,
    )
    return Case("blend_avg_total_by_brand", "blend",
                [Turn(v, verify=_verify_blend_total_avg())],
                needs_blend=True, secondary_name=vehicles.datasource_name)


def build_blend_merge_case(trips: DataSourceMetadata,
                           vehicles: DataSourceMetadata) -> Optional[Case]:
    """FIX-061 regression: the blend arrives on a LATER turn (add-sheet merge path).

    Turn 1 is a plain single-DS sheet; turn 2 adds the blended sheet, driving
    _merge_new_sheet_into_workbook Step 7 — the path that used to re-inject the
    <datasource-relationships> block with caption-form names AFTER every
    caption→physical fix pass had already run. On the real published DSes
    (caption "Vehicle Id" ≠ physical vehicle_id) that declared a phantom link
    column: red-``!`` duplicate in the data pane, blend never engaged, and
    secondary-DS filters silently no-oped (the reported "marque Ford" KPI
    showing all 100 vehicles / 1000 trips)."""
    linking = _detect_linking_fields(trips, vehicles)
    if not linking:
        return None
    t1 = VizIntent(viz_type="bar_chart", title="Bench BlendMerge Cost by Status",
                   x_field="Status", y_field="Cost Eur",
                   action="new", datasource_luid=trips.luid)
    t2 = VizIntent(viz_type="bar_chart", title="Bench BlendMerge Cargo by Vehicle Type",
                   x_field="Vehicle Type", y_field="Cargo Weight Kg",
                   action="new", datasource_luid=trips.luid,
                   secondary_datasource_luid=vehicles.luid)
    return Case("blend_added_as_second_sheet", "blend", [
        Turn(t1, expect_sheets=1),
        Turn(t2, verify=_verify_blend_vehicle_type(), expect_sheets=2),
    ], needs_blend=True, secondary_name=vehicles.datasource_name)


def build_multids_modify_case(trips: DataSourceMetadata,
                              vehicles: DataSourceMetadata) -> Optional[Case]:
    """FIX-012 pre-test: a workbook that accumulated sheets from TWO different
    datasources (trips, then vehicles), then an IN-PLACE modify of the last sheet
    (same x/y/color, one filter added — the exact "filtre par X" follow-up).

    Turn 3 drives modify_sheet_in_existing on the multi-DS workbook, which
    re-enters add_sheet_to_existing's different-datasource branch
    (_merge_new_sheet_into_workbook) — the path the FIX-012 revocation note in
    FIXES.md reports crashed publish with Tableau Cloud's misleading 500000.
    The filter is top_n (member-agnostic) so Tier 3 can verify the filter truly
    restricts the data without knowing the datasource's member values."""
    if not any(f.name == "Vehicle Type" for f in vehicles.fields):
        return None
    measure = next((f.name for f in vehicles.fields if f.role == "measure"), None)
    if measure:
        y, agg = measure, "SUM"
    elif any(f.name == "Vehicle Id" for f in vehicles.fields):
        y, agg = "Vehicle Id", "COUNTD"
    else:
        return None
    t1 = VizIntent(viz_type="bar_chart", title="Bench MultiDS Cost by Status",
                   x_field="Status", y_field="Cost Eur",
                   action="new", datasource_luid=trips.luid)
    t2 = VizIntent(viz_type="bar_chart", title="Bench MultiDS Fleet by Type",
                   x_field="Vehicle Type", y_field=y, aggregation=agg,
                   action="new", datasource_luid=vehicles.luid)
    t3 = t2.model_copy(update={
        "action": "modify",
        "filters": [FilterSpec(field="Vehicle Type", op="top_n", value=2, by=y)],
    })
    return Case("multids_modify_in_place", "multiturn", [
        Turn(t1, expect_sheets=1),
        Turn(t2, expect_sheets=2),
        Turn(t3, modify_of="Bench MultiDS Fleet by Type",
             verify=_verify_max_rows(2), expect_sheets=2),
    ])


def build_class5_case(ventes: DataSourceMetadata) -> Optional[Case]:
    """The reported Class #5 defect: a mis-cased member value ("OUest") publishes a
    filter that selects 0 of 5 Region members → a blank chart. With FIX-054 value
    correction the harness snaps "OUest" → "Ouest" and the view renders bars.
    WITHOUT correction (--no-value-correction) the non-empty guard FAILs it."""
    if not any(f.name == "Region" for f in ventes.fields):
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Sales by Sub-Category for OUest Region",
        x_field="Sub Category", y_field="Sales",
        filters=[FilterSpec(field="Region", op="eq", value="OUest")],  # deliberately mis-cased
        action="new", datasource_luid=ventes.luid,
    )
    # verify=None → only the non-empty guard applies (renders bars ⇒ PASS).
    return Case("class5_miscased_region_ouest", "filter", [Turn(v)])


def build_class5b_case(ventes: DataSourceMetadata) -> Optional[Case]:
    """Class #5 translation sub-case (FIX-054b): the LLM *translated* the user's
    "OUest" into the English "West", which is not a real member (and must NOT be
    fuzzy-mapped to "Est"). Correction recovers "Ouest" from the question text, so
    the view renders bars. WITHOUT correction the filter selects 0 of 5 → the
    non-empty guard FAILs."""
    if not any(f.name == "Region" for f in ventes.fields):
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Sales by Sub-Category for West Region",
        x_field="Sub Category", y_field="Sales",
        filters=[FilterSpec(field="Region", op="eq", value="West")],  # LLM-translated, not a member
        action="new", datasource_luid=ventes.luid,
    )
    return Case("class5b_translated_region_west", "filter",
                [Turn(v, question="Show sales by sub-category only for the OUest region")])


def build_class5c_case(ventes: DataSourceMetadata) -> Optional[Case]:
    """Class #5 intent sub-case (FIX-054c): the user asks for the English "West region"
    not knowing the data stores French "Ouest". Nothing in the question is a member, so
    deterministic recovery can't help — only the LLM (shown the real domain) bridges
    "West" → "Ouest". With correction the view renders bars; without it, 0 of 5 → FAIL.
    NOTE: exercises the live LLM resolver, so it needs a reachable provider."""
    if not any(f.name == "Region" for f in ventes.fields):
        return None
    v = VizIntent(
        viz_type="bar_chart", title="Sales by Sub-Category West Intent",
        x_field="Sub Category", y_field="Sales",
        filters=[FilterSpec(field="Region", op="eq", value="West")],  # English, not a member
        action="new", datasource_luid=ventes.luid,
    )
    return Case("class5c_english_region_west", "filter",
                [Turn(v, question="Show sales by sub-category only for the West region")])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _rows_signature(rows: list[dict]) -> tuple:
    return tuple(sorted(tuple(sorted(r.items())) for r in rows))


async def _baseline_differs(turn: "Turn", primary: DataSourceMetadata, cu: str,
                            filtered_rows: list[dict]) -> Optional[str]:
    """Publish an UNFILTERED twin of the turn's viz and assert the filtered view
    differs from it (proves a row-level measure filter is actually active).
    Returns a failure reason if the data is unchanged, else None."""
    base_viz = turn.viz.model_copy(update={
        "filters": [], "title": (turn.viz.title + " BASE")[:60],
    })
    out = settings.output_dir
    _fn, path = generate_twb(base_viz, primary, server_ds_content_url=cu,
                             server_ds_name=primary.datasource_name)
    base_wb = str(out / "_bench_baseline_tmp.twb")
    shutil.copy2(str(path), base_wb)
    try:
        os.remove(str(path))
    except OSError:
        pass
    luid = await publish_workbook(base_wb, settings.tableau_default_project_id, overwrite=True)
    try:
        await asyncio.sleep(2)
        base_csv = await asyncio.to_thread(_read_view_csv, luid, base_viz.title)
        base_rows = _parse_csv(base_csv)
        if _rows_signature(base_rows) == _rows_signature(filtered_rows):
            return f"filtered result identical to unfiltered ({len(base_rows)} rows) — filter is a no-op"
        return None
    finally:
        try:
            await asyncio.to_thread(_delete_workbook, luid)
        except Exception:
            pass


async def run_case(
    case: Case,
    ds_by_luid: dict[str, DataSourceMetadata],
    content_urls: dict[str, str],
    do_publish: bool,
    do_verify: bool,
    keep: bool,
    value_correction: bool = True,
) -> CaseResult:
    res = CaseResult(name=case.name, group=case.group)
    out = settings.output_dir
    out.mkdir(exist_ok=True)
    wb_name = f"_bench_{case.name}"
    wb_path = str(out / f"{wb_name}.twb")
    if os.path.exists(wb_path):
        os.remove(wb_path)

    primary = ds_by_luid[case.turns[0].viz.datasource_luid]
    res.intent_summary = f"{case.turns[0].viz.viz_type} x={case.turns[0].viz.x_field}"

    # blend kwargs (mirror main.py)
    blend_kwargs: dict = {}
    secondary_meta = None
    if case.needs_blend:
        sec = next((d for d in ds_by_luid.values() if d.datasource_name == case.secondary_name), None)
        if sec:
            secondary_meta = sec
            linking = _detect_linking_fields(primary, sec)
            blend_kwargs = dict(
                blend_secondary_content_url=content_urls[sec.luid],
                blend_secondary_name=sec.datasource_name,
                blend_linking_fields=linking,
                blend_secondary_metadata=sec,
            )

    def _merged_meta(base: DataSourceMetadata) -> DataSourceMetadata:
        if not secondary_meta:
            return base
        seen = {f.name.lower() for f in base.fields}
        merged = list(base.fields) + [f for f in secondary_meta.fields if f.name.lower() not in seen]
        return DataSourceMetadata(datasource_name=base.datasource_name,
                                  fields=merged, luid=base.luid)

    def _sheet_names(path: str) -> list[str]:
        return [w.get("name", "") for w in etree.parse(path).getroot().findall(".//worksheets/worksheet")]

    workbook_luid: Optional[str] = None
    last_sheet_name: str = case.turns[0].viz.title
    try:
        cu = content_urls[primary.luid]
        for i, turn in enumerate(case.turns):
            # Per-turn datasource: a case may accumulate sheets from DIFFERENT
            # datasources (multi-DS workbook), so each turn resolves its own
            # schema + content URL from its viz.datasource_luid (single-DS cases
            # are unaffected — every turn resolves to the same `primary`).
            t_ds = ds_by_luid.get(turn.viz.datasource_luid) or primary
            t_cu = content_urls[t_ds.luid]
            # Blend kwargs apply only to turns whose intent declares a secondary
            # datasource (mirror main.py, which builds them per turn) — a case may
            # mix plain single-DS turns with blended ones (FIX-061 merge-path case).
            t_blend = blend_kwargs if turn.viz.secondary_datasource_luid else {}
            meta = _merged_meta(t_ds) if t_blend else t_ds
            # Mirror production: correct filter VALUES against real datasource members
            # (FIX-054) before generating. Toggle off with --no-value-correction to
            # reproduce the pre-fix "mis-cased value → blank chart" failure.
            if value_correction:
                try:
                    from main import _correct_filter_values
                    turn.viz, _w = await _correct_filter_values(
                        turn.viz, list(ds_by_luid.values()), turn.question)
                except Exception:
                    pass
            before = set(_sheet_names(wb_path)) if i > 0 else set()
            if i == 0:
                _fn, path = generate_twb(turn.viz, meta,
                                         server_ds_content_url=t_cu,
                                         server_ds_name=t_ds.datasource_name,
                                         **t_blend)
                shutil.copy2(str(path), wb_path)
                try:
                    os.remove(str(path))
                except OSError:
                    pass
                last_sheet_name = turn.viz.title
            elif turn.modify_of:
                # Production `action="modify"` path: replace ONLY the sheet titled
                # modify_of, keeping every other sheet (FIX-012 in-place tweak).
                modify_sheet_in_existing(wb_path, turn.viz, meta,
                                         old_title=turn.modify_of,
                                         server_ds_content_url=t_cu,
                                         server_ds_name=t_ds.datasource_name,
                                         **t_blend)
                added = [n for n in _sheet_names(wb_path) if n not in before]
                last_sheet_name = added[-1] if added else turn.viz.title
            else:
                add_sheet_to_existing(wb_path, turn.viz, meta,
                                      server_ds_content_url=t_cu,
                                      server_ds_name=t_ds.datasource_name,
                                      **t_blend)
                # add_sheet_to_existing dedups the title; the real sheet name is the
                # one that appeared in this turn (so verification reads the RIGHT view).
                added = [n for n in _sheet_names(wb_path) if n not in before]
                last_sheet_name = added[-1] if added else turn.viz.title

            # Tier 1 — structural
            reason = _structural_check(wb_path, [t.viz for t in case.turns[:i + 1]])
            n_sheets = len(_sheet_names(wb_path))
            if reason is None and n_sheets != turn.expect_sheets:
                reason = f"expected {turn.expect_sheets} worksheet(s), found {n_sheets}"
            if reason:
                res.structural = "FAIL"
                res.reason = f"turn {i + 1}: {reason}"
                return res
        res.structural = "PASS"

        # Tier 2 — publish (overwrite=True is idempotent for re-runs)
        if do_publish:
            try:
                workbook_luid = await publish_workbook(
                    wb_path, settings.tableau_default_project_id, overwrite=True)
                res.publish = "PASS"
            except Exception as exc:
                res.publish = "FAIL"
                res.reason = f"publish: {type(exc).__name__}: {str(exc)[:240]}"
                return res
        else:
            res.publish = "SKIP"

        # Tier 3 — data verification on the LAST turn's view.
        # Run whenever there's a verifier/baseline OR the case is filtered/blended
        # (so the non-empty guard catches a chart-blanking filter even with no verifier).
        last = case.turns[-1]
        _checkable = last.verify or last.baseline_compare or case.group in ("filter", "multiturn", "blend")
        if do_publish and do_verify and workbook_luid and _checkable:
            try:
                await asyncio.sleep(2)
                csv_text = await asyncio.to_thread(_read_view_csv, workbook_luid, last_sheet_name)
                rows = _parse_csv(csv_text)
                # Blind-spot guard (FIX-054): a filter that restricts to ZERO rows
                # used to pass vacuously ("all rows satisfy the predicate" is true on
                # an empty set). A blanked chart is a FAIL unless emptiness is expected.
                vreason = None
                if not rows and not last.allow_empty:
                    vreason = "filter emptied the chart (0 rows / 0 marks)"
                if vreason is None:
                    vreason = last.verify(rows) if last.verify else None
                if vreason is None and last.baseline_compare:
                    vreason = await _baseline_differs(last, primary, cu, rows)
                if vreason:
                    res.data = "CONTRADICTED"
                    res.reason = f"data: {vreason}"
                else:
                    res.data = "VERIFIED"
            except Exception as exc:
                res.data = "UNVERIFIED"
                res.reason = f"verify-skip: {type(exc).__name__}: {str(exc)[:160]}"
        elif do_publish and do_verify:
            res.data = "UNVERIFIED"  # no verifier (e.g. plain viz / not_null on null-free data)
        else:
            res.data = "SKIP"

    except Exception as exc:
        res.structural = res.structural if res.structural != "?" else "FAIL"
        res.reason = f"{type(exc).__name__}: {str(exc)[:240]}\n{traceback.format_exc()[-400:]}"
    finally:
        if workbook_luid and not keep:
            try:
                await asyncio.to_thread(_delete_workbook, workbook_luid)
            except Exception:
                pass
    return res


def print_report(results: list[CaseResult]) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append(f"{'CASE':<34} {'GROUP':<10} {'STRUCT':<7} {'PUBLISH':<8} {'DATA':<13} REASON")
    lines.append("-" * 96)
    for r in results:
        lines.append(f"{r.name:<34} {r.group:<10} {r.structural:<7} {r.publish:<8} {r.data:<13} {r.reason}")
    lines.append("=" * 96)
    n = len(results)
    fails = [r for r in results if not r.ok]
    lines.append(f"  {n - len(fails)}/{n} cases OK   ({len(fails)} FAIL)")
    if fails:
        lines.append("  FAILING (work list):")
        for r in fails:
            lines.append(f"    - {r.name}: {r.reason}")
    lines.append("=" * 96)
    out = "\n".join(lines)
    print(out)
    return out


async def main_async(args) -> int:
    if not _TSC_AVAILABLE:
        print("ERROR: tableau_server could not be imported (check .env credentials).")
        return 2

    await signin()
    schemas = await get_all_datasource_schemas()
    by_name = {d.datasource_name: d for d in schemas}
    if "trips" not in by_name:
        print(f"ERROR: 'trips' datasource not found. Available: {list(by_name)}")
        return 2

    trips = by_name["trips"]
    ds_by_luid = {d.luid: d for d in schemas}
    content_urls: dict[str, str] = {}
    for d in schemas:
        try:
            content_urls[d.luid] = await get_datasource_content_url(d.luid)
        except Exception:
            pass

    cases = build_matrix(trips.luid, args.matrix)
    if args.matrix in ("all", "multiturn") and "vehicles" in by_name:
        mdc = build_multids_modify_case(trips, by_name["vehicles"])
        if mdc:
            cases.append(mdc)
    if args.matrix in ("all", "blend") and "vehicles" in by_name:
        bc = build_blend_case(trips, by_name["vehicles"])
        if bc:
            cases.append(bc)
        bmc = build_blend_merge_case(trips, by_name["vehicles"])
        if bmc:
            cases.append(bmc)
        bavg = build_blend_measure_case(trips, by_name["vehicles"])
        if bavg:
            cases.append(bavg)
        btot = build_blend_total_case(trips, by_name["vehicles"])
        if btot:
            cases.append(btot)
    if args.matrix in ("all", "filter") and "ventes" in by_name:
        c5 = build_class5_case(by_name["ventes"])
        if c5:
            cases.append(c5)
        c5b = build_class5b_case(by_name["ventes"])
        if c5b:
            cases.append(c5b)
        c5c = build_class5c_case(by_name["ventes"])
        if c5c:
            cases.append(c5c)
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"No case named {args.case!r}")
            return 2

    print(f"Running {len(cases)} case(s)  publish={not args.no_publish}  "
          f"verify={args.verify_data and not args.no_publish}  rag={args.rag}")
    results = []
    for c in cases:
        t0 = time.time()
        r = await run_case(c, ds_by_luid, content_urls,
                           do_publish=not args.no_publish,
                           do_verify=args.verify_data,
                           keep=args.keep,
                           value_correction=not args.no_value_correction)
        results.append(r)
        print(f"  [{time.time() - t0:5.1f}s] {r.name:<34} "
              f"struct={r.structural} pub={r.publish} data={r.data}"
              + (f"  <- {r.reason}" if not r.ok else ""))

    report = print_report(results)
    report_path = Path(settings.output_dir) / "eval_harness_report.txt"
    try:
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {report_path}")
    except Exception:
        pass

    return 0 if all(r.ok for r in results) else 1


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end eval harness for the Text-to-Viz agent")
    p.add_argument("--matrix", choices=["all", "viz", "filter", "multiturn", "blend"], default="all")
    p.add_argument("--case", default=None, help="run only the case with this name")
    p.add_argument("--no-publish", action="store_true", help="Tier 1 only (offline, fast)")
    p.add_argument("--verify-data", dest="verify_data", action="store_true", default=True,
                   help="Tier 3: read published view CSV and assert filter applied (default on)")
    p.add_argument("--no-verify-data", dest="verify_data", action="store_false")
    p.add_argument("--keep", action="store_true", help="do not delete published _bench_* workbooks")
    p.add_argument("--no-value-correction", action="store_true",
                   help="skip FIX-054 filter-value correction (reproduce the mis-cased-value blank-chart failure)")
    p.add_argument("--rag", choices=["on", "off"], default="on", help="(Phase 2) RAG toggle")
    args = p.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
