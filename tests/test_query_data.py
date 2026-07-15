"""C1 — `query_data` tool: direct factual answers from the VizQL Data Service
(no chart generated) + C5 — user-named blend linking field.

Offline: the VDS call is mocked; these pin datasource selection, field-name
correction, filter-value snapping (FIX-054 machinery reused), answer formatting,
the pipeline dispatch (tool name query_data → conversation-mode answer, no .twb),
and the blend-linking-field resolution helper.
"""
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from llm import LLMResponse, ToolCall
from schemas import DataSourceMetadata, FieldInfo, FieldType
import main

TRIPS = DataSourceMetadata(
    datasource_name="trips", luid="trips-luid",
    fields=[
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Trip Date", type=FieldType.DATE, role="dimension"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure"),
    ],
)
VEHICLES = DataSourceMetadata(
    datasource_name="vehicles", luid="vehicles-luid",
    fields=[
        FieldInfo(name="Vehicle Type", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Capacity Kg", type=FieldType.FLOAT, role="measure"),
    ],
)


# --- formatting -------------------------------------------------------------

def test_format_single_value():
    out = main._format_data_answer([{"SUM(Cost Eur)": 12345.5}], "Cost Eur", "SUM", None, None)
    assert "12 345.50" in out and "Total" in out


def test_format_grouped_sorted_desc():
    rows = [{"Status": "Delayed", "SUM(Cost Eur)": 10.0},
            {"Status": "Completed", "SUM(Cost Eur)": 99.0}]
    out = main._format_data_answer(rows, "Cost Eur", "SUM", "Status", None)
    assert out.index("Completed") < out.index("Delayed"), "groups must be sorted by value desc"


def test_format_mentions_filters():
    out = main._format_data_answer([{"SUM(Cost Eur)": 5}], "Cost Eur", "SUM", None,
                                   [{"field": "Status", "values": ["Completed"]}])
    assert "Status" in out and "Completed" in out


# --- _answer_data_question ---------------------------------------------------

@pytest.mark.asyncio
async def test_measure_name_fuzzy_corrected_and_ds_autoselected():
    """"cost eur" (bad casing) resolves; the DS holding the measure is chosen."""
    vds = AsyncMock(return_value=[{"SUM(Cost Eur)": 777.0}])
    with patch.object(main, "query_datasource_aggregate", new=vds):
        out = await main._answer_data_question(
            {"measure": "cost_eur", "aggregation": "SUM"}, [VEHICLES, TRIPS], "total cost?")
    assert "777" in out
    vds.assert_awaited_once()
    assert vds.await_args.args[0] == "trips-luid"
    assert vds.await_args.args[1] == "Cost Eur"


@pytest.mark.asyncio
async def test_unknown_measure_returns_field_list():
    out = await main._answer_data_question(
        {"measure": "Chiffre Affaires Introuvable Xyz"}, [TRIPS], "total?")
    assert "Je ne trouve pas" in out and "Cost Eur" in out


@pytest.mark.asyncio
async def test_filter_value_snapped_to_real_member():
    """FIX-054 machinery applies to query_data filters too: "completed" → "Completed"."""
    main._member_cache.clear()
    vds = AsyncMock(return_value=[{"SUM(Cost Eur)": 1.0}])
    with patch.object(main, "query_datasource_aggregate", new=vds), \
         patch.object(main, "get_dimension_members",
                      new=AsyncMock(return_value=["Completed", "Cancelled"])):
        await main._answer_data_question(
            {"measure": "Cost Eur",
             "filters": [{"field": "status", "values": ["completed"]}]},
            [TRIPS], "total for completed?")
    sent_filters = vds.await_args.kwargs["filters"]
    assert sent_filters == [{"field": "Status", "values": ["Completed"], "exclude": False}]


@pytest.mark.asyncio
async def test_empty_vds_result_degrades_gracefully():
    with patch.object(main, "query_datasource_aggregate", new=AsyncMock(return_value=[])):
        out = await main._answer_data_question({"measure": "Cost Eur"}, [TRIPS], "total?")
    assert "KPI" in out  # offers the chart fallback instead of erroring


# --- pipeline dispatch -------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_query_data_returns_conversation_answer(client):
    """A query_data tool call → conversation-mode answer with the real number,
    NO .twb generated, NO publish."""
    session_id = str(uuid.uuid4())
    meta = {"datasource_name": "trips",
            "fields": [{"name": "Cost Eur", "type": "float", "role": "measure"}]}
    tool_resp = LLMResponse(tool_calls=[ToolCall(
        name="query_data",
        arguments={"measure": "Cost Eur", "aggregation": "SUM"},
    )])
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("main.get_all_datasource_schemas", new_callable=AsyncMock, return_value=[TRIPS]), \
         patch.object(main, "query_datasource_aggregate",
                      new=AsyncMock(return_value=[{"SUM(Cost Eur)": 424242.0}])):
        mock_llm.return_value = tool_resp
        r = await client.post("/chat", json={
            "question": "combien de coût total ?", "session_id": session_id,
            "metadata": meta, "conversation_history": [],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "conversation"
    assert "424 242" in body["message"]
    assert body["twb_filename"] == ""


# --- C5: blend linking field -------------------------------------------------

def test_blend_linking_field_resolved_normalized():
    """"vehicle_id" (underscore) resolves to caption "Vehicle Id" in both schemas."""
    trips_with_vid = TRIPS.model_copy(update={"fields": TRIPS.fields + [
        FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension")]})
    out = main._resolve_blend_linking_field("vehicle_id", trips_with_vid, VEHICLES)
    assert out == ["Vehicle Id"]


def test_blend_linking_field_missing_in_one_ds_falls_back():
    """A field present in only ONE datasource cannot link — return None (auto-detect)."""
    assert main._resolve_blend_linking_field("Status", TRIPS, VEHICLES) is None
    assert main._resolve_blend_linking_field(None, TRIPS, VEHICLES) is None
