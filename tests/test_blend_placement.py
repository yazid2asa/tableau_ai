"""FIX-062 — placement of the blend linking field (MEASURED on the real Server).

Reported with the "Coût moyen des voyages par marque de véhicule" screenshot
(100 marks): the hope was that when the secondary contributes ONLY a measure,
the linking field could stay out of the view and Tableau would re-aggregate
the blended measure per category (one clean mark per brand).

**Measured 2026-07-08: that variant is WRONG.** Without the linking field in
the view the blend never joins — every brand silently shows the same GLOBAL
average (harness `blend_secondary_measure_avg`: 'single repeated value
241.95 — measure not joined per brand'). A chart that looks clean but shows
one wrong number everywhere is the worst failure mode, so the policy is:

* the worksheet uses ANY secondary field (dimension, filter, or measure) →
  the linking field is injected on Detail (invisible; per-linking-value mark
  granularity is a Tableau blending constraint, not a placement bug);
* the LLM already placed the linking field on a visible shelf (the user asked
  a per-link breakdown) → no injection (pre-existing skip).

These tests pin the LOD injection for all three secondary-usage shapes so the
"cleaner" no-LOD variant is never re-introduced.
"""

from lxml import etree

import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, FilterSpec, VizIntent


TRIPS = DataSourceMetadata(
    datasource_name="trips",
    luid="trips-luid",
    fields=[
        FieldInfo(name="Vehicle Id", local_name="vehicle_id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Trip Id", local_name="trip_id", type=FieldType.STRING, role="dimension"),
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


def _union(primary: DataSourceMetadata, secondary: DataSourceMetadata) -> DataSourceMetadata:
    seen = {f.name.lower() for f in primary.fields}
    return primary.model_copy(update={
        "fields": primary.fields + [f for f in secondary.fields if f.name.lower() not in seen],
    })


def _worksheet(out) -> etree._Element:
    return etree.parse(str(out)).getroot().findall(".//worksheet")[-1]


def _lods(ws) -> list[str]:
    return [el.get("column", "") for el in ws.findall(".//panes/pane/encodings/lod")]


def test_measure_only_secondary_keeps_detail_lod():
    """vehicles primary (Brand on x), AVG(Cost Eur) blended from trips — the
    secondary contributes only a measure. The link must STILL be in the view:
    the no-LOD variant was CONTRADICTED on the real Server (global average
    repeated for every brand)."""
    viz = VizIntent(
        viz_type="bar_chart", title="Avg Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="AVG", action="new",
    )
    _fn, out = twb_generator.generate_twb(
        viz, _union(VEHICLES, TRIPS),
        server_ds_content_url="vehicles", server_ds_name="vehicles",
        blend_secondary_content_url="trips", blend_secondary_name="trips",
        blend_linking_fields=["Vehicle Id"],
        blend_secondary_metadata=TRIPS,
    )
    try:
        ws = _worksheet(out)
        lods = _lods(ws)
        assert any(":vehicle_id:" in ref for ref in lods), f"Detail LOD required, got {lods}"
        # The measure still routes to the secondary block
        rows = (ws.find(".//rows").text or "")
        cols = (ws.find(".//cols").text or "")
        assert ":cost_eur:" in rows and ":brand:" in cols
        assert rows.split("].")[0] != cols.split("].")[0], "measure must reference the SECONDARY datasource"
    finally:
        out.unlink(missing_ok=True)


def test_secondary_dimension_keeps_detail_lod():
    """FIX-043 regression: a secondary-owned dimension on a shelf requires the
    link in the view → Detail LOD injected."""
    viz = VizIntent(
        viz_type="bar_chart", title="Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="SUM", action="new",
    )
    _fn, out = twb_generator.generate_twb(
        viz, _union(TRIPS, VEHICLES),
        server_ds_content_url="trips", server_ds_name="trips",
        blend_secondary_content_url="vehicles", blend_secondary_name="vehicles",
        blend_linking_fields=["Vehicle Id"],
        blend_secondary_metadata=VEHICLES,
    )
    try:
        lods = _lods(_worksheet(out))
        assert any(":vehicle_id:" in ref for ref in lods), f"Detail LOD expected, got {lods}"
    finally:
        out.unlink(missing_ok=True)


def test_secondary_filter_keeps_detail_lod():
    """The Ford-KPI case: the only secondary field is a FILTER dimension — the
    link must be in the view for the filter to slice per linking value."""
    viz = VizIntent(
        viz_type="kpi", title="Voyages Ford",
        x_field="Trip Id", y_field="", aggregation="COUNT", action="new",
        filters=[FilterSpec(field="Brand", op="eq", value="Ford")],
    )
    _fn, out = twb_generator.generate_twb(
        viz, _union(TRIPS, VEHICLES),
        server_ds_content_url="trips", server_ds_name="trips",
        blend_secondary_content_url="vehicles", blend_secondary_name="vehicles",
        blend_linking_fields=["Vehicle Id"],
        blend_secondary_metadata=VEHICLES,
    )
    try:
        lods = _lods(_worksheet(out))
        assert any(":vehicle_id:" in ref for ref in lods), f"Detail LOD expected, got {lods}"
    finally:
        out.unlink(missing_ok=True)


def test_llm_placed_linking_field_skips_lod():
    """The user asked a per-vehicle breakdown (linking field on a visible
    shelf) — no duplicate Detail injection on top."""
    viz = VizIntent(
        viz_type="bar_chart", title="Cost per Vehicle by Brand",
        x_field="Vehicle Id", y_field="Cost Eur", color_field="Brand",
        aggregation="SUM", action="new",
    )
    _fn, out = twb_generator.generate_twb(
        viz, _union(TRIPS, VEHICLES),
        server_ds_content_url="trips", server_ds_name="trips",
        blend_secondary_content_url="vehicles", blend_secondary_name="vehicles",
        blend_linking_fields=["Vehicle Id"],
        blend_secondary_metadata=VEHICLES,
    )
    try:
        lods = _lods(_worksheet(out))
        assert not any(":vehicle_id:" in ref for ref in lods), (
            f"linking field already on cols — no LOD duplicate expected, got {lods}")
    finally:
        out.unlink(missing_ok=True)
