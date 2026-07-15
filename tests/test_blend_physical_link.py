"""FIX-061 — the blend's <datasource-relationships> block must bind the linking
field by its PHYSICAL name (local_name), never the GraphQL caption.

Reported bug (screenshot "Nombre total de voyages ... véhicules de marque Ford"):
both published DSes caption the linking field "Vehicle Id" (physical
``vehicle_id``). The worksheet binds physical (FIX-002/FIX-043:
``[none:vehicle_id:nk]`` on Detail), but the relationships block declared
caption-form ``[Vehicle Id]`` columns and mapped ``[none:Vehicle Id:nk]`` —
phantoms that exist in neither datasource. The data pane showed a duplicate
red-``!`` "Vehicle Id", the blend link never engaged, and the secondary-DS
filter (Brand=Ford) silently selected everything (100 marks / 1000 trips).

First-turn workbooks were incidentally repaired by the FIX-004 file-wide
caption→physical replace; the add-sheet merge path rebuilds the block AFTER
every post-save fix pass (Step 7 of ``_merge_new_sheet_into_workbook``) so it
shipped broken — exactly the accumulated-session flow in the screenshot.
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

# main.py passes the blend-merged (primary + secondary) schema as `metadata`
MERGED = TRIPS.model_copy(update={"fields": TRIPS.fields + VEHICLES.fields})

BLEND_KWARGS = dict(
    server_ds_content_url="trips",
    server_ds_name="trips",
    blend_secondary_content_url="vehicles",
    blend_secondary_name="vehicles",
    blend_linking_fields=["Vehicle Id"],  # captions — what _detect_linking_fields returns
    blend_secondary_metadata=VEHICLES,
)


def _assert_physical_link(root):
    rels = root.find("datasource-relationships")
    assert rels is not None, "blend workbook must declare <datasource-relationships>"

    dep_cols = [c.get("name") for dep in rels.findall("datasource-dependencies")
                for c in dep.findall("column")]
    dep_insts = [ci.get("name") for dep in rels.findall("datasource-dependencies")
                 for ci in dep.findall("column-instance")]
    maps = [(m.get("key", ""), m.get("value", ""))
            for rel in rels.findall("datasource-relationship")
            for m in rel.findall("./column-mapping/map")]

    assert dep_cols and all(name == "[vehicle_id]" for name in dep_cols), dep_cols
    assert dep_insts and all(name == "[none:vehicle_id:nk]" for name in dep_insts), dep_insts
    assert maps, "blend must declare at least one column-mapping/map"
    for key, value in maps:
        assert ":vehicle_id:" in key and ":vehicle_id:" in value, (key, value)

    # The caption form is a phantom column (exists in neither published DS) —
    # it must not survive anywhere in the workbook XML.
    content = etree.tostring(root).decode()
    assert "[Vehicle Id]" not in content
    assert ":Vehicle Id:" not in content


def test_first_turn_blend_links_by_physical_name():
    viz = VizIntent(
        viz_type="bar_chart", title="Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="SUM", action="new",
    )
    _fn, out = twb_generator.generate_twb(viz, MERGED, **BLEND_KWARGS)
    try:
        _assert_physical_link(etree.parse(str(out)).getroot())
    finally:
        out.unlink(missing_ok=True)


def test_added_blend_sheet_links_by_physical_name():
    """The reported path: single-DS first sheet, then a blended KPI with a
    secondary-DS filter added on a later turn (merge Step 7)."""
    first = VizIntent(
        viz_type="bar_chart", title="Cost by Vehicle",
        x_field="Vehicle Id", y_field="Cost Eur", aggregation="SUM", action="new",
    )
    kpi = VizIntent(
        viz_type="kpi", title="Voyages Ford",
        x_field="Trip Id", y_field="", aggregation="COUNT", action="new",
        filters=[FilterSpec(field="Brand", op="eq", value="Ford")],
    )
    _fn, out = twb_generator.generate_twb(
        first, TRIPS, server_ds_content_url="trips", server_ds_name="trips",
    )
    try:
        twb_generator.add_sheet_to_existing(str(out), kpi, MERGED, **BLEND_KWARGS)
        _assert_physical_link(etree.parse(str(out)).getroot())
    finally:
        out.unlink(missing_ok=True)


def test_stale_caption_link_is_healed_on_next_turn():
    """A session workbook published BEFORE the fix carries caption-form phantom
    declarations. The next blend turn must purge them (self-healing), so the
    user's broken accumulated workbook recovers on its next republish."""
    first = VizIntent(
        viz_type="bar_chart", title="Cost by Vehicle",
        x_field="Vehicle Id", y_field="Cost Eur", aggregation="SUM", action="new",
    )
    blended = VizIntent(
        viz_type="bar_chart", title="Cost by Brand",
        x_field="Brand", y_field="Cost Eur", aggregation="SUM", action="new",
    )
    _fn, out = twb_generator.generate_twb(
        first, TRIPS, server_ds_content_url="trips", server_ds_name="trips",
    )
    try:
        twb_generator.add_sheet_to_existing(str(out), blended, MERGED, **BLEND_KWARGS)

        # Corrupt the workbook the way pre-FIX-061 merges did: caption-form
        # phantom columns in both dep blocks + a caption-instance mapping.
        tree = etree.parse(str(out))
        rels = tree.getroot().find("datasource-relationships")
        assert rels is not None
        for dep in rels.findall("datasource-dependencies"):
            etree.SubElement(dep, "column", attrib={
                "caption": "Vehicle Id", "datatype": "string",
                "name": "[Vehicle Id]", "role": "dimension", "type": "nominal",
            })
            etree.SubElement(dep, "column-instance", attrib={
                "column": "[Vehicle Id]", "derivation": "None",
                "name": "[none:Vehicle Id:nk]", "pivot": "key", "type": "nominal",
            })
        rel = rels.find("datasource-relationship")
        mapping = rel.find("column-mapping")
        etree.SubElement(mapping, "map", attrib={
            "key": f"[{rel.get('source')}].[none:Vehicle Id:nk]",
            "value": f"[{rel.get('target')}].[none:Vehicle Id:nk]",
        })
        tree.write(str(out), xml_declaration=True, encoding="UTF-8")

        # Next blend turn — the injection must heal the stale caption entries.
        another = VizIntent(
            viz_type="bar_chart", title="Cost by Brand v2",
            x_field="Brand", y_field="Cost Eur", aggregation="SUM", action="new",
        )
        twb_generator.add_sheet_to_existing(str(out), another, MERGED, **BLEND_KWARGS)
        _assert_physical_link(etree.parse(str(out)).getroot())
    finally:
        out.unlink(missing_ok=True)
