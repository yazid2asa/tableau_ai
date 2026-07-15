"""Regression tests for FIX-055 (neq/not_in exclusion filters) and FIX-056
(date_range explicit date-span filter).

Offline structural guards in the test_publish_hardening.py style: they exercise
the real sqlproxy generation path (generate_twb with a fake content_url — no
network) and assert the exclude groupfilter shape / quantitative date range that
the real Server accepts. The end-to-end proof (publish + read the view CSV)
lives in eval_harness.py (`flt_neq`, `flt_not_in`, `flt_date_range`).
"""
import os
import lxml.etree as ET

from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
from twb_generator import generate_twb

CU = "trips"  # fake content_url — wires sqlproxy XML attrs, no network

TRIPS = DataSourceMetadata(
    datasource_name="trips",
    fields=[
        FieldInfo(name="Trip Date", type=FieldType.DATE, role="dimension", local_name="trip_date"),
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension", local_name="status"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure", local_name="cost_eur"),
    ],
    luid="trips-luid",
)


def _mk(title, filters, x="Status", y="Cost Eur"):
    return VizIntent(viz_type="bar_chart", title=title, x_field=x, y_field=y,
                     filters=filters, aggregation="SUM", action="new",
                     datasource_luid="trips-luid")


def _gen(viz):
    _fn, path = generate_twb(viz, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    return str(path)


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _find_filter(root, col_token):
    for f in root.iter("filter"):
        if col_token in (f.get("column") or ""):
            return f
    return None


# --------------------------------------------------------------------------
# FIX-055 — neq / not_in become a real except(all-members, …) exclude filter
# --------------------------------------------------------------------------

def test_neq_emits_except_groupfilter():
    """FIX-055: op=neq must produce except(level-members, member) — not a no-op."""
    path = _gen(_mk("Excl Neq", [FilterSpec(field="Status", op="neq", value="Cancelled")]))
    try:
        root = ET.parse(path).getroot()
        filt = _find_filter(root, ":status:")
        assert filt is not None, "a categorical filter on status must exist"
        gf = filt.find("groupfilter")
        assert gf is not None and gf.get("function") == "except"
        children = gf.findall("groupfilter")
        funcs = [c.get("function") for c in children]
        assert funcs[0] == "level-members", "first child must be all-members"
        assert funcs[1] == "member"
        assert children[1].get("member") == '"Cancelled"'
    finally:
        _cleanup(path)


def test_not_in_emits_except_union():
    """FIX-055: op=not_in with several values → except(level-members, union(member…))."""
    path = _gen(_mk("Excl NotIn", [FilterSpec(field="Status", op="not_in",
                                              values=["Cancelled", "Delayed"])]))
    try:
        root = ET.parse(path).getroot()
        filt = _find_filter(root, ":status:")
        assert filt is not None
        gf = filt.find("groupfilter")
        assert gf is not None and gf.get("function") == "except"
        children = gf.findall("groupfilter")
        assert children[0].get("function") == "level-members"
        union = children[1]
        assert union.get("function") == "union"
        members = {m.get("member") for m in union.findall("groupfilter")}
        assert members == {'"Cancelled"', '"Delayed"'}
    finally:
        _cleanup(path)


def test_exclude_filter_has_slices_entry():
    """FIX-051 must also cover the exclude filter: without <slices> the predicate
    is never pushed and the exclusion silently does nothing."""
    path = _gen(_mk("Excl Slices", [FilterSpec(field="Status", op="neq", value="Cancelled")]))
    try:
        root = ET.parse(path).getroot()
        view = root.find(".//worksheet//view")
        filt = _find_filter(root, ":status:")
        slice_cols = {c.text for c in view.findall("slices/column") if c.text}
        assert filt.get("column") in slice_cols
    finally:
        _cleanup(path)


def test_numeric_excluded_member_not_quoted():
    """FIX-055: a pure-numeric excluded member must stay unquoted (same lesson as
    FIX-048 — Tableau drops the filter when a numeric member is quoted)."""
    path = _gen(_mk("Excl Numeric", [FilterSpec(field="Status", op="neq", value=2024)]))
    try:
        root = ET.parse(path).getroot()
        filt = _find_filter(root, ":status:")
        member_gf = [g for g in filt.iter("groupfilter") if g.get("function") == "member"]
        assert member_gf and member_gf[0].get("member") == "2024"
    finally:
        _cleanup(path)


def test_eq_filter_untouched_by_exclude_pass():
    """The exclude rewrite must not touch a normal include (eq) filter."""
    path = _gen(_mk("Incl Eq", [FilterSpec(field="Status", op="eq", value="Completed")]))
    try:
        root = ET.parse(path).getroot()
        filt = _find_filter(root, ":status:")
        assert filt is not None
        gf = filt.find("groupfilter")
        assert gf is not None and gf.get("function") != "except"
    finally:
        _cleanup(path)


# --------------------------------------------------------------------------
# FIX-056 — date_range becomes a quantitative date range
# --------------------------------------------------------------------------

def test_date_range_emits_quantitative_range():
    """FIX-056: op=date_range with ISO bounds → quantitative filter #min#..#max#
    on the raw date field (the exact shape the year op already uses)."""
    path = _gen(_mk("DR Both", [FilterSpec(field="Trip Date", op="date_range",
                                           date_min="2025-03-01", date_max="2025-06-30")],
                    x="Trip Date"))
    try:
        content = open(path, encoding="utf-8").read()
        assert "#2025-03-01#" in content
        assert "#2025-06-30#" in content
        root = ET.parse(path).getroot()
        filt = _find_filter(root, ":trip_date:")
        assert filt is not None and filt.get("class") == "quantitative"
    finally:
        _cleanup(path)


def test_date_range_open_ended_min_only():
    """FIX-056: date_min alone gives an open-ended 'since' range (no crash, no no-op)."""
    path = _gen(_mk("DR Min", [FilterSpec(field="Trip Date", op="date_range",
                                          date_min="2025-01-01")],
                    x="Trip Date"))
    try:
        content = open(path, encoding="utf-8").read()
        assert "#2025-01-01#" in content
    finally:
        _cleanup(path)


def test_date_range_without_bounds_is_skipped():
    """FIX-056: a date_range with no bounds would be a silent no-op filter —
    it must be skipped entirely (no phantom filter card)."""
    path = _gen(_mk("DR None", [FilterSpec(field="Trip Date", op="date_range")],
                    x="Trip Date"))
    try:
        root = ET.parse(path).getroot()
        assert _find_filter(root, ":trip_date:") is None
    finally:
        _cleanup(path)
