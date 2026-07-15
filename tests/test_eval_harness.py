"""Wire the eval_harness Tier-1 structural checks into pytest (offline, no network).

The full harness (eval_harness.py) publishes to a real Tableau Server and reads
the view CSV back; that's the end-to-end proof and is run manually. Here we run
its structural checker (`_structural_check`) against freshly generated workbooks
for the representative crash/filter cases, so the safety net's core logic and the
generation invariants stay green in CI.
"""
import os
import pytest

from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
from twb_generator import generate_twb, add_sheet_to_existing
import eval_harness as H

CU = "trips"
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


def _viz(vt, title, x, y="", color=None, filters=None):
    return VizIntent(viz_type=vt, title=title, x_field=x, y_field=y, color_field=color,
                     filters=filters or [], action="new", datasource_luid="trips-luid")


def _cleanup(*p):
    for x in p:
        try:
            os.remove(x)
        except OSError:
            pass


def test_harness_structural_check_passes_single_sheet_filters():
    """Every single-sheet filter case the harness builds is structurally clean."""
    cases = [
        _viz("bar_chart", "eq", "Status", "Cost Eur", filters=[FilterSpec(field="Status", op="eq", value="Completed")]),
        _viz("bar_chart", "year", "Trip Date", "Cost Eur", filters=[FilterSpec(field="Trip Date", op="year", value=2025)]),
        _viz("bar_chart", "month", "Trip Date", "Cost Eur", filters=[FilterSpec(field="Trip Date", op="month", value=6)]),
        _viz("combo", "combo", "Trip Date", "Cost Eur", color="Fuel Consumed Liters"),
    ]
    for v in cases:
        _fn, path = generate_twb(v, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
        try:
            reason = H._structural_check(str(path), [v])
            assert reason is None, f"{v.title}: {reason}"
        finally:
            _cleanup(str(path))


def test_harness_detects_multiturn_combo_filter_clean():
    """The reported crash path (combo → year-filtered combo) passes the harness
    structural checker after the fix (FIX-046)."""
    combo = _viz("combo", "Cout carburant par date", "Trip Date", "Cost Eur", color="Fuel Consumed Liters")
    _fn, path = generate_twb(combo, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    path = str(path)
    try:
        filtered = _viz("combo", "Cout carburant par date", "Trip Date", "Cost Eur",
                        color="Fuel Consumed Liters",
                        filters=[FilterSpec(field="Trip Date", op="year", value=2025)])
        add_sheet_to_existing(path, filtered, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
        reason = H._structural_check(path, [combo, filtered])
        assert reason is None, reason
    finally:
        _cleanup(path)


def test_harness_structural_check_catches_duplicate_worksheet_names():
    """Sanity: the checker actually fails on a malformed workbook (two same-named sheets)."""
    import lxml.etree as ET
    combo = _viz("bar_chart", "Dup", "Status", "Cost Eur")
    _fn, path = generate_twb(combo, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    path = str(path)
    try:
        root = ET.parse(path).getroot()
        wss = root.find(".//worksheets")
        import copy
        wss.append(copy.deepcopy(wss.find("worksheet")))  # duplicate the sheet
        ET.ElementTree(root).write(path, xml_declaration=True, encoding="UTF-8")
        reason = H._structural_check(path, [combo])
        assert reason and "duplicate worksheet" in reason
    finally:
        _cleanup(path)
