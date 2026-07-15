"""FIX-059 — live relative-date filters + C4 — dashboard assembly.

Offline structural guards; the real-Server proof is the harness `flt_last_n_days`
/ `flt_last_n_months` cases (relative window verified on the published view CSV)
and a publish check of the dashboard workbook.
"""
import uuid
from unittest.mock import AsyncMock, patch

import lxml.etree as ET
import pytest

from conftest import make_tool_response
from config import settings
from llm import LLMResponse, ToolCall
from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
import twb_generator
from twb_generator import generate_twb, add_dashboard_to_workbook, list_worksheet_titles

CU = "trips"
TRIPS = DataSourceMetadata(
    datasource_name="trips",
    fields=[
        FieldInfo(name="Trip Date", type=FieldType.DATE, role="dimension", local_name="trip_date"),
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension", local_name="status"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure", local_name="cost_eur"),
    ],
    luid="trips-luid",
)


def _gen(filters, x="Trip Date"):
    viz = VizIntent(viz_type="bar_chart", title="RD", x_field=x, y_field="Cost Eur",
                    filters=filters, aggregation="SUM", action="new",
                    datasource_luid="trips-luid")
    _fn, path = generate_twb(viz, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    return str(path)


def _cleanup(*paths):
    import os
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# --- FIX-059: relative-date -------------------------------------------------

def test_last_n_days_becomes_live_relative_date():
    path = _gen([FilterSpec(field="Trip Date", op="last_n_days", value=30)])
    try:
        root = ET.parse(path).getroot()
        rel = [f for f in root.iter("filter") if f.get("class") == "relative-date"]
        assert rel, "last_n_days must emit a live relative-date filter"
        f = rel[0]
        assert f.get("first-period") == "-30"
        assert f.get("last-period") == "0"
        assert f.get("period-type") == "day"
        assert f.get("include-future") == "false"
        assert f.get("min") is None and f.get("max") is None
        # FIX-051 slices entry must still reference the filter column
        view = root.find(".//worksheet//view")
        slice_cols = {c.text for c in view.findall("slices/column") if c.text}
        assert f.get("column") in slice_cols
    finally:
        _cleanup(path)


def test_last_n_months_period_type_month():
    path = _gen([FilterSpec(field="Trip Date", op="last_n_months", value=12)])
    try:
        root = ET.parse(path).getroot()
        rel = [f for f in root.iter("filter") if f.get("class") == "relative-date"]
        assert rel and rel[0].get("period-type") == "month"
        assert rel[0].get("first-period") == "-12"
    finally:
        _cleanup(path)


def test_year_filter_untouched_by_relative_pass():
    """A plain year filter stays a quantitative range — only last_n_* upgrade."""
    path = _gen([FilterSpec(field="Trip Date", op="year", value=2025)])
    try:
        root = ET.parse(path).getroot()
        assert not [f for f in root.iter("filter") if f.get("class") == "relative-date"]
        assert [f for f in root.iter("filter") if f.get("class") == "quantitative"]
    finally:
        _cleanup(path)


# --- C4: dashboard -----------------------------------------------------------

def _wb_two_sheets():
    viz1 = VizIntent(viz_type="bar_chart", title="A", x_field="Status", y_field="Cost Eur",
                     aggregation="SUM", action="new", datasource_luid="trips-luid")
    _fn, out = generate_twb(viz1, TRIPS, server_ds_content_url=CU, server_ds_name="trips")
    viz2 = VizIntent(viz_type="bar_chart", title="B", x_field="Trip Date", y_field="Cost Eur",
                     aggregation="SUM", action="new", datasource_luid="trips-luid")
    twb_generator.add_sheet_to_existing(str(out), viz2, TRIPS,
                                        server_ds_content_url=CU, server_ds_name="trips")
    return out


def test_dashboard_block_and_window_injected():
    out = _wb_two_sheets()
    try:
        name = add_dashboard_to_workbook(str(out), "Mon Dashboard", ["A", "B"])
        assert name == "Mon Dashboard"
        root = ET.parse(str(out)).getroot()
        dash = root.find(".//dashboards/dashboard")
        assert dash is not None and dash.get("name") == "Mon Dashboard"
        zone_names = {z.get("name") for z in dash.iter("zone") if z.get("name")}
        assert zone_names == {"A", "B"}
        wins = [w for w in root.findall(".//windows/window")
                if w.get("class") == "dashboard"]
        assert len(wins) == 1 and wins[0].get("name") == "Mon Dashboard"
        # re-running with the same name replaces, never duplicates
        add_dashboard_to_workbook(str(out), "Mon Dashboard", ["A"])
        root = ET.parse(str(out)).getroot()
        assert len(root.findall(".//dashboards/dashboard")) == 1
        assert len([w for w in root.findall(".//windows/window")
                    if w.get("class") == "dashboard"]) == 1
    finally:
        out.unlink(missing_ok=True)


# --- pipeline ----------------------------------------------------------------

META_DICT = TRIPS.model_dump()
BAR_A = {"viz_type": "bar_chart", "title": "Sheet One", "x_field": "Status",
         "y_field": "Cost Eur", "color_field": None, "filters": [],
         "calculated_fields": [], "clarification_needed": None, "sort": None,
         "aggregation": "SUM", "color_scheme": "tableau10", "action": "new"}
BAR_B = {**BAR_A, "title": "Sheet Two", "x_field": "Trip Date"}


@pytest.mark.asyncio
async def test_pipeline_create_dashboard_from_all_sheets(client):
    session_id = str(uuid.uuid4())
    for intent in (BAR_A, BAR_B):
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_tool_response(intent)
            r = await client.post("/chat", json={
                "question": intent["title"], "session_id": session_id,
                "metadata": META_DICT, "conversation_history": [],
            })
        assert r.status_code == 200
    wb = settings.output_dir / f"Analyse_{session_id[:8]}.twb"
    try:
        dash_call = LLMResponse(tool_calls=[ToolCall(
            name="create_dashboard", arguments={"title": "Vue d'ensemble"})])
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = dash_call
            r = await client.post("/chat", json={
                "question": "mets tout dans un dashboard", "session_id": session_id,
                "metadata": META_DICT, "conversation_history": [],
            })
        assert r.status_code == 200
        assert "Vue d'ensemble" in r.json()["message"]
        root = ET.parse(str(wb)).getroot()
        dash = root.find(".//dashboards/dashboard")
        assert dash is not None
        assert {z.get("name") for z in dash.iter("zone") if z.get("name")} == \
               {"Sheet One", "Sheet Two"}
    finally:
        wb.unlink(missing_ok=True)
