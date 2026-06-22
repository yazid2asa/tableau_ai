from lxml import etree

import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent


TRIPS = DataSourceMetadata(
    datasource_name="trips",
    luid="trips-luid",
    fields=[
        FieldInfo(name="vehicle_id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="distance_km", type=FieldType.INTEGER, role="measure"),
        FieldInfo(name="cost_eur", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="fuel_consumed_liters", type=FieldType.FLOAT, role="measure"),
    ],
)

VEHICLES = DataSourceMetadata(
    datasource_name="vehicles",
    luid="vehicles-luid",
    fields=[
        FieldInfo(name="vehicle_id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="vehicle_type", type=FieldType.STRING, role="dimension"),
    ],
)


def test_adding_blended_sheet_preserves_datasource_relationships():
    first = VizIntent(
        viz_type="scatter",
        title="Cost vs Distance",
        x_field="distance_km",
        y_field="cost_eur",
        aggregation="SUM",
        action="new",
    )
    blended = VizIntent(
        viz_type="bar_chart",
        title="Fuel by Vehicle Type",
        x_field="vehicle_type",
        y_field="fuel_consumed_liters",
        aggregation="SUM",
        action="new",
    )
    merged_meta = TRIPS.model_copy(update={"fields": TRIPS.fields + VEHICLES.fields})

    _filename, out = twb_generator.generate_twb(
        first,
        TRIPS,
        server_ds_content_url="trips",
        server_ds_name="trips",
    )
    try:
        twb_generator.add_sheet_to_existing(
            str(out),
            blended,
            merged_meta,
            server_ds_content_url="trips",
            server_ds_name="trips",
            blend_secondary_content_url="vehicles",
            blend_secondary_name="vehicles",
            blend_linking_fields=["vehicle_id"],
            blend_secondary_metadata=VEHICLES,
        )

        root = etree.parse(str(out)).getroot()
        rels = root.find("datasource-relationships")
        assert rels is not None

        relationship = rels.find("datasource-relationship")
        assert relationship is not None
        assert relationship.get("source", "").startswith("sqlproxy.")
        assert relationship.get("target", "").startswith("sqlproxy.")

        mappings = relationship.findall("./column-mapping/map")
        assert mappings
        assert "vehicle_id" in mappings[0].get("key", "")
        assert "vehicle_id" in mappings[0].get("value", "")
    finally:
        out.unlink(missing_ok=True)
