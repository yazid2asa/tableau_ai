"""FIX-063 — TOTAL(<agg>) wrap for non-additive aggregations in a blend.

A blend computes at the linking-field granularity (FIX-062, measured), so an
"AVG by <cross-DS dimension>" bar is really a STACK of per-linking-value
segments whose averages ADD UP — a wrong total (the reported 'Coût moyen des
voyages par marque' screenshot, 100 marks). The fix, verified end-to-end on
the real Server (16/16 brands equal to the exact VDS oracle + clean PNG):

* the measure pill becomes a calc field ``TOTAL(<agg>([measure]))`` —
  every per-linking-value segment carries the TRUE category value;
* its ``<column-instance derivation='User'>`` gets
  ``<table-calc ordering-field='[..].[none:<link>:nk]' ordering-type='Field'/>``
  (compute using the linking field → partition per category);
* every pane's ``<view><breakdown value='off'/>`` (mark stacking OFF —
  the pane-view ``breakdown`` element per Tableau's official
  ``twb_2026.2.0.xsd``; a style-rule 'stack-marks' is silently ignored),
  so the identical segments overlap into ONE visible mark per category.

Additive aggregations (SUM/COUNT) are untouched — their segments stack to the
correct total. A secondary-owned measure is untouched too (a calc formula
cannot reference another datasource's field, FIX-044).
"""

from lxml import etree

import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent


TRIPS = DataSourceMetadata(
    datasource_name="trips",
    luid="trips-luid",
    fields=[
        FieldInfo(name="Vehicle Id", local_name="vehicle_id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Cost Eur", local_name="cost_eur", type=FieldType.FLOAT, role="measure"),
    ],
)

VEHICLES = DataSourceMetadata(
    datasource_name="vehicles",
    luid="vehicles-luid",
    fields=[
        FieldInfo(name="Vehicle Id", local_name="vehicle_id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Brand", local_name="brand", type=FieldType.STRING, role="dimension"),
    ],
)


def _union(primary, secondary):
    seen = {f.name.lower() for f in primary.fields}
    return primary.model_copy(update={
        "fields": primary.fields + [f for f in secondary.fields if f.name.lower() not in seen]})


MERGED = _union(TRIPS, VEHICLES)

BLEND_KWARGS = dict(
    server_ds_content_url="trips",
    server_ds_name="trips",
    blend_secondary_content_url="vehicles",
    blend_secondary_name="vehicles",
    blend_linking_fields=["Vehicle Id"],
    blend_secondary_metadata=VEHICLES,
)


def _gen(viz):
    return twb_generator.generate_twb(viz, MERGED, **BLEND_KWARGS)


def _viz(agg="AVG", **over):
    base = dict(viz_type="bar_chart", title=f"{agg} Cost by Brand",
                x_field="Brand", y_field="Cost Eur", aggregation=agg, action="new")
    base.update(over)
    return VizIntent(**base)


def test_avg_blend_gets_total_wrap_with_table_calc_and_breakdown_off():
    _fn, out = _gen(_viz("AVG"))
    try:
        root = etree.parse(str(out)).getroot()
        ws = root.findall(".//worksheet")[-1]

        # 1. TOTAL(AVG(...)) calc field exists (physical-name formula, FIX-004)
        calc = next((c for c in root.findall(".//datasources/datasource/column")
                     if c.get("caption") == "Cost Eur (AVG)"
                     and c.find("calculation") is not None), None)
        assert calc is not None
        assert calc.find("calculation").get("formula") == "TOTAL(AVG([cost_eur]))"

        # 2. The calc pill drives the value slot
        rows_txt = ws.find(".//rows").text or ""
        assert f"[usr:{calc.get('name').strip('[]')}:qk]" in rows_txt

        # 3. compute-using the linking field on the calc instance
        tc = next((tc for ci in ws.findall(".//datasource-dependencies/column-instance")
                   if ci.get("column") == calc.get("name")
                   for tc in ci.findall("table-calc")), None)
        assert tc is not None
        assert tc.get("ordering-type") == "Field"
        assert ":vehicle_id:" in tc.get("ordering-field", "")

        # 4. mark stacking OFF in every pane
        breakdowns = [b.get("value") for b in ws.findall(".//panes/pane/view/breakdown")]
        assert breakdowns and all(v == "off" for v in breakdowns), breakdowns
    finally:
        out.unlink(missing_ok=True)


def test_sum_blend_is_not_wrapped():
    """Additive aggregation: segments stack to the correct total — no wrap,
    stacking untouched (the VERIFIED blend_cargo_by_vehicle_type look)."""
    _fn, out = _gen(_viz("SUM"))
    try:
        root = etree.parse(str(out)).getroot()
        assert not any(c.get("caption") == "Cost Eur (SUM)"
                       for c in root.findall(".//datasources/datasource/column"))
        ws = root.findall(".//worksheet")[-1]
        breakdowns = [b.get("value") for b in ws.findall(".//panes/pane/view/breakdown")]
        assert all(v != "off" for v in breakdowns), breakdowns
    finally:
        out.unlink(missing_ok=True)


def test_secondary_owned_measure_is_not_wrapped():
    """vehicles primary, AVG measure owned by the SECONDARY (trips) — a calc
    formula can't reference another datasource's field (FIX-044): no wrap,
    honest per-linking-value rendering kept."""
    viz = _viz("AVG")
    _fn, out = twb_generator.generate_twb(
        viz, _union(VEHICLES, TRIPS),
        server_ds_content_url="vehicles", server_ds_name="vehicles",
        blend_secondary_content_url="trips", blend_secondary_name="trips",
        blend_linking_fields=["Vehicle Id"],
        blend_secondary_metadata=TRIPS,
    )
    try:
        root = etree.parse(str(out)).getroot()
        assert not any((c.get("caption") or "").endswith("(AVG)")
                       for c in root.findall(".//datasources/datasource/column"))
    finally:
        out.unlink(missing_ok=True)


def test_wrap_survives_add_sheet_merge():
    """The production flow: the AVG blend arrives as a LATER sheet. The merge
    dedups the primary datasource element, so Step 4b must copy the calc
    <column> def across or the sheet references an undefined [Calculation_x]."""
    first = VizIntent(viz_type="bar_chart", title="Cost by Vehicle",
                      x_field="Vehicle Id", y_field="Cost Eur",
                      aggregation="SUM", action="new")
    _fn, out = twb_generator.generate_twb(
        first, TRIPS, server_ds_content_url="trips", server_ds_name="trips")
    try:
        twb_generator.add_sheet_to_existing(str(out), _viz("AVG"), MERGED, **BLEND_KWARGS)
        root = etree.parse(str(out)).getroot()
        ws = root.findall(".//worksheet")[-1]

        calc = next((c for c in root.findall(".//datasources/datasource/column")
                     if c.get("caption") == "Cost Eur (AVG)"
                     and c.find("calculation") is not None), None)
        assert calc is not None, "calc column def lost in the merge (Step 4b)"

        # the calc def must live in the SAME datasource element the sheet references
        rows_txt = ws.find(".//rows").text or ""
        ds_of_pill = rows_txt.split("].")[0].strip("[")
        owner = next(d for d in root.findall(".//datasources/datasource")
                     if any(c is calc for c in d.findall("column")))
        assert owner.get("name") == ds_of_pill

        tc = next((tc for ci in ws.findall(".//datasource-dependencies/column-instance")
                   if ci.get("column") == calc.get("name")
                   for tc in ci.findall("table-calc")), None)
        assert tc is not None
        breakdowns = [b.get("value") for b in ws.findall(".//panes/pane/view/breakdown")]
        assert breakdowns and all(v == "off" for v in breakdowns)
    finally:
        out.unlink(missing_ok=True)
