"""Phase-1 production-hardening regression tests (classes #1 + #3).

Offline structural guards for the publish-parse crash class and the filters
that didn't actually filter. They exercise the real sqlproxy generation path
(generate_twb / add_sheet_to_existing with a fake content_url — no network) and
assert the XML invariants whose violation crashed Tableau's native parser or
silently dropped the filter. Each test names the FIX it pins.

The end-to-end proof (publish to the real Server + read the view CSV) lives in
eval_harness.py; these are the CI-runnable, network-free regressions.
"""
import os
import lxml.etree as ET
import pytest

from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
import twb_generator as G
from twb_generator import (
    generate_twb, add_sheet_to_existing, _subtract_months,
    VIZ_TYPE_TO_MARK,
)
from datetime import date

CU = "trips"  # fake content_url — wires sqlproxy XML attrs, no network

TRIPS = DataSourceMetadata(
    datasource_name="trips",
    fields=[
        FieldInfo(name="Trip Date", type=FieldType.DATE, role="dimension", local_name="trip_date"),
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension", local_name="status"),
        FieldInfo(name="Destination City", type=FieldType.STRING, role="dimension", local_name="destination_city"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure", local_name="cost_eur"),
        FieldInfo(name="Fuel Consumed Liters", type=FieldType.FLOAT, role="measure", local_name="fuel_consumed_liters"),
    ],
    luid="trips-luid",
)


def _mk(vt, title, x, y="", color=None, filters=None, agg="SUM"):
    return VizIntent(viz_type=vt, title=title, x_field=x, y_field=y, color_field=color,
                     filters=filters or [], aggregation=agg, action="new", datasource_luid="trips-luid")


def _gen(viz):
    _fn, path = generate_twb(viz, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    return str(path)


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _instance_type_map(root):
    m = {}
    for ci in root.iter("column-instance"):
        m.setdefault(ci.get("name"), set()).add(ci.get("type"))
    return m


# --------------------------------------------------------------------------
# Class #1 — publish-parse crashes
# --------------------------------------------------------------------------

def test_gantt_mark_is_ganttbar():
    """FIX-045: gantt mark primitive must be 'GanttBar' (Tableau rejects 'Gantt'/'Gantt Bar')."""
    assert VIZ_TYPE_TO_MARK["gantt"] == "GanttBar"
    path = _gen(_mk("gantt", "G", "Status", "Cost Eur", color="Trip Date"))
    try:
        root = ET.parse(path).getroot()
        marks = {m.get("class") for m in root.iter("mark")}
        assert "GanttBar" in marks
        assert "Gantt Bar" not in marks and "Gantt" not in marks
    finally:
        _cleanup(path)


def test_multiturn_combo_year_no_instance_type_conflict():
    """FIX-046 (the reported crash): a combo time-series sheet followed by a
    year-filtered combo must NOT leave two same-named column-instances with
    conflicting ordinal/quantitative types (→ 'worksheet could not be parsed')."""
    combo = _mk("combo", "Cout carburant par date", "Trip Date", "Cost Eur", color="Fuel Consumed Liters")
    path = _gen(combo)
    try:
        combo_filtered = _mk("combo", "Cout carburant par date", "Trip Date", "Cost Eur",
                             color="Fuel Consumed Liters",
                             filters=[FilterSpec(field="Trip Date", op="year", value=2025)])
        add_sheet_to_existing(path, combo_filtered, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
        root = ET.parse(path).getroot()

        # no instance name carries two different types
        conflicts = {n: t for n, t in _instance_type_map(root).items() if len(t) > 1}
        assert not conflicts, f"conflicting instance types: {conflicts}"

        # no duplicate worksheet names
        names = [w.get("name") for w in root.findall(".//worksheets/worksheet")]
        assert len(names) == len(set(names)) == 2

        # the unfiltered sheet keeps its ORDINAL continuous-date axis (:ok]), the
        # filtered sheet's quantitative date instance is :qk]
        types = _instance_type_map(root)
        assert types.get("[none:trip_date:ok]") == {"ordinal"}
        assert "quantitative" in types.get("[none:trip_date:qk]", set())
    finally:
        _cleanup(path)


def test_year_filter_has_matching_slices():
    """FIX (slices): every active filter must have a <slices><column> so it restricts data."""
    path = _gen(_mk("bar_chart", "Bar", "Status", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="year", value=2025)]))
    try:
        root = ET.parse(path).getroot()
        view = root.find(".//worksheet//view")
        filt_cols = {f.get("column") for f in view.findall("filter")}
        slice_cols = {c.text for c in view.findall("slices/column")}
        assert filt_cols and filt_cols.issubset(slice_cols)
    finally:
        _cleanup(path)


# --------------------------------------------------------------------------
# Class #3 — filters that actually filter
# --------------------------------------------------------------------------

def test_month_filter_is_unquoted_datepart():
    """FIX-048: month → discrete MONTH() date-part filter with an UNQUOTED numeric
    member (quoted member makes Tableau drop the filter)."""
    path = _gen(_mk("bar_chart", "Bar", "Trip Date", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="month", value=6)]))
    try:
        root = ET.parse(path).getroot()
        filt = root.find(".//worksheet//view/filter")
        assert "[mn:" in filt.get("column"), filt.get("column")
        members = [gf.get("member") for gf in filt.iter("groupfilter") if gf.get("member")]
        assert members == ["6"], members  # unquoted, not '"6"'
    finally:
        _cleanup(path)


def test_quarter_filter_is_unquoted_datepart():
    """FIX-048: quarter → discrete QUARTER() date-part filter, unquoted member."""
    path = _gen(_mk("bar_chart", "Bar", "Trip Date", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="quarter", value=1)]))
    try:
        root = ET.parse(path).getroot()
        filt = root.find(".//worksheet//view/filter")
        assert "[qr:" in filt.get("column")
        members = [gf.get("member") for gf in filt.iter("groupfilter") if gf.get("member")]
        assert members == ["1"], members
    finally:
        _cleanup(path)


def test_month_quarter_no_longer_hardcode_2024():
    """FIX-048: a bare month/quarter filter must not silently target year 2024."""
    path = _gen(_mk("bar_chart", "Bar", "Trip Date", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="month", value=6)]))
    try:
        txt = open(path, encoding="utf-8").read()
        assert "2024" not in txt, "month filter must not hardcode 2024"
    finally:
        _cleanup(path)


def test_last_n_days_is_live_relative_filter_not_noop():
    """FIX-049 → FIX-059: last_n_days must emit a REAL restricting filter — since
    FIX-059 that's a live relative-date window (re-evaluated at view time), never
    the pre-FIX-049 no-op empty categorical."""
    path = _gen(_mk("bar_chart", "Bar", "Trip Date", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="last_n_days", value=30)]))
    try:
        root = ET.parse(path).getroot()
        filt = root.find(".//worksheet//view/filter")
        assert filt.get("class") == "relative-date"
        assert filt.get("first-period") == "-30" and filt.get("period-type") == "day"
    finally:
        _cleanup(path)


def test_last_n_months_is_live_relative_filter():
    """FIX-049 → FIX-059: last_n_months emits a live relative-date month window."""
    path = _gen(_mk("bar_chart", "Bar", "Trip Date", "Cost Eur",
                    filters=[FilterSpec(field="Trip Date", op="last_n_months", value=12)]))
    try:
        root = ET.parse(path).getroot()
        filt = root.find(".//worksheet//view/filter")
        assert filt.get("class") == "relative-date"
        assert filt.get("first-period") == "-12" and filt.get("period-type") == "month"
    finally:
        _cleanup(path)


def test_not_null_emits_exclude_null_filter():
    """FIX-050: not_null is rewritten to except(all-members, %null%) — not a no-op."""
    path = _gen(_mk("bar_chart", "Bar", "Status", "Cost Eur",
                    filters=[FilterSpec(field="Status", op="not_null")]))
    try:
        root = ET.parse(path).getroot()
        filt = root.find(".//worksheet//view/filter")
        funcs = [gf.get("function") for gf in filt.iter("groupfilter")]
        assert "except" in funcs
        members = [gf.get("member") for gf in filt.iter("groupfilter") if gf.get("member")]
        assert "%null%" in members
    finally:
        _cleanup(path)


def test_top_n_ranks_by_aggregated_measure():
    """FIX-047: top_n must rank by SUM([measure]), not NONE([measure]) (which makes
    Tableau error on query / return everything unfiltered)."""
    path = _gen(_mk("bar_chart", "Bar", "Destination City", "Cost Eur",
                    filters=[FilterSpec(field="Destination City", op="top_n", value=5, by="Cost Eur")]))
    try:
        txt = open(path, encoding="utf-8").read()
        assert "NONE([cost_eur])" not in txt and "NONE(" not in txt
        assert "SUM([cost_eur])" in txt
    finally:
        _cleanup(path)


def test_bottom_n_ranks_by_aggregated_measure():
    """FIX-047: bottom_n also ranks by SUM([measure]); ASC ordering + end=top picks the lowest N."""
    path = _gen(_mk("bar_chart", "Bar", "Destination City", "Cost Eur",
                    filters=[FilterSpec(field="Destination City", op="bottom_n", value=5, by="Cost Eur")]))
    try:
        root = ET.parse(path).getroot()
        gf_orders = [gf for gf in root.iter("groupfilter") if gf.get("function") == "order"]
        assert gf_orders and gf_orders[0].get("expression", "").startswith("SUM(")
        assert gf_orders[0].get("direction") == "ASC"
    finally:
        _cleanup(path)


def test_subtract_months_calendar_math():
    """_subtract_months does real calendar-month subtraction with day clamping."""
    assert _subtract_months(date(2026, 6, 29), 12) == date(2025, 6, 29)
    assert _subtract_months(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert _subtract_months(date(2026, 1, 15), 13) == date(2024, 12, 15)
