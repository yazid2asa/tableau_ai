"""FIX-054c reliability pass + FIX-058 text-JSON intent rescue.

The LLM member resolver (`_resolve_member_via_llm`) is non-deterministic: one bad
answer used to be CACHED and blanked the chart for the whole session (the exact
regression seen in output/eval_harness_report.txt on class5c). These tests pin:
  - retry-once on a failed/invalid attempt, no retry on a deliberate NONE;
  - failures are never cached (successes are);
  - blend charts: a filter on a SECONDARY-owned dimension fetches members with
    the secondary luid (it was silently unprotected before);
  - transparency: a semantic correction ("West"→"Ouest") is surfaced to the user,
    a mere case fix ("OUest") is not;
  - FIX-058: an intent emitted as plain TEXT (OpenRouter fallback tool-calling
    flakiness) is rescued into a chart instead of a JSON conversation bubble.
"""
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_text_response
from llm import LLMResponse
from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
import main

VENTES = DataSourceMetadata(
    datasource_name="ventes", luid="ventes-luid",
    fields=[
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sub Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
    ],
)
TRIPS = DataSourceMetadata(
    datasource_name="trips", luid="trips-luid",
    fields=[
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure"),
    ],
)
VEHICLES = DataSourceMetadata(
    datasource_name="vehicles", luid="vehicles-luid",
    fields=[
        FieldInfo(name="Vehicle Type", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
    ],
)
REGION_MEMBERS = ["Centre", "Est", "Sud", "Ouest", "Nord"]


def _fresh_caches():
    main._member_cache.clear()
    main._value_resolution_cache.clear()


# --- _resolve_member_via_llm: retry + cache policy -------------------------

@pytest.mark.asyncio
async def test_resolver_retries_once_after_transient_failure():
    """Attempt 1 raises (rate limit) → attempt 2 succeeds → member resolved."""
    _fresh_caches()
    responses = [ValueError("[google] HTTP 429: rate limit"),
                 LLMResponse(content="Ouest")]
    with patch.object(main, "call_llm", new=AsyncMock(side_effect=responses)) as m:
        out = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
    assert out == "Ouest"
    assert m.await_count == 2


@pytest.mark.asyncio
async def test_resolver_retries_once_after_invalid_answer():
    """Attempt 1 hallucinates a non-member → attempt 2 gives a real member."""
    _fresh_caches()
    responses = [LLMResponse(content="Western Territories"),
                 LLMResponse(content="Ouest")]
    with patch.object(main, "call_llm", new=AsyncMock(side_effect=responses)) as m:
        out = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
    assert out == "Ouest"
    assert m.await_count == 2


@pytest.mark.asyncio
async def test_resolver_does_not_retry_deliberate_none():
    """A deliberate NONE is an answer, not a flake — exactly one LLM call."""
    _fresh_caches()
    with patch.object(main, "call_llm",
                      new=AsyncMock(return_value=LLMResponse(content="NONE"))) as m:
        out = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "gibberish", "Atlantis", "ventes-luid")
    assert out is None
    assert m.await_count == 1


@pytest.mark.asyncio
async def test_resolver_failure_is_not_cached():
    """Both attempts fail → None; a LATER call (LLM recovered) must succeed —
    the old behavior cached the None and made the miss permanent."""
    _fresh_caches()
    with patch.object(main, "call_llm",
                      new=AsyncMock(side_effect=ValueError("HTTP 429"))):
        first = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
    assert first is None
    with patch.object(main, "call_llm",
                      new=AsyncMock(return_value=LLMResponse(content="Ouest"))):
        second = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
    assert second == "Ouest", "a cached failure must not survive the session"


@pytest.mark.asyncio
async def test_resolver_success_is_cached():
    _fresh_caches()
    with patch.object(main, "call_llm",
                      new=AsyncMock(return_value=LLMResponse(content="Ouest"))) as m:
        a = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
        b = await main._resolve_member_via_llm(
            "Region", REGION_MEMBERS, "West region", "West", "ventes-luid")
    assert a == b == "Ouest"
    assert m.await_count == 1, "second call must come from the cache"


# --- _correct_filter_values: secondary datasource dims (blend) --------------

@pytest.mark.asyncio
async def test_secondary_dimension_members_fetched_with_secondary_luid():
    """Blend chart filtering on a secondary-owned dim: members must be fetched
    from the SECONDARY datasource (the primary knows nothing about it)."""
    _fresh_caches()
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Vehicle Type",
                    y_field="Cost Eur",
                    filters=[FilterSpec(field="Vehicle Type", op="eq", value="TRuck")],
                    action="new", datasource_luid="trips-luid",
                    secondary_datasource_luid="vehicles-luid")
    fetch = AsyncMock(return_value=["Truck", "Van", "Car"])
    with patch.object(main, "get_dimension_members", new=fetch):
        corrected, warning = await main._correct_filter_values(
            viz, [TRIPS, VEHICLES], "cost by vehicle type for TRuck")
    fetch.assert_awaited_once_with("vehicles-luid", "Vehicle Type")
    assert corrected.filters[0].value == "Truck"
    assert warning is None  # case fix only — no user-facing note


@pytest.mark.asyncio
async def test_primary_dimension_still_uses_primary_luid():
    _fresh_caches()
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Status", y_field="Cost Eur",
                    filters=[FilterSpec(field="Status", op="eq", value="completed")],
                    action="new", datasource_luid="trips-luid",
                    secondary_datasource_luid="vehicles-luid")
    fetch = AsyncMock(return_value=["Completed", "Cancelled"])
    with patch.object(main, "get_dimension_members", new=fetch):
        corrected, _ = await main._correct_filter_values(viz, [TRIPS, VEHICLES])
    fetch.assert_awaited_once_with("trips-luid", "Status")
    assert corrected.filters[0].value == "Completed"


# --- transparency: semantic corrections are surfaced ------------------------

@pytest.mark.asyncio
async def test_semantic_correction_is_surfaced_to_user():
    """"West"→"Ouest" (different words) → the user is told the interpretation."""
    _fresh_caches()
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="West")],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members",
                      new=AsyncMock(return_value=REGION_MEMBERS)), \
         patch.object(main, "call_llm",
                      new=AsyncMock(return_value=LLMResponse(content="Ouest"))):
        corrected, note = await main._correct_filter_values(
            viz, [VENTES], "Show sales for the West region")
    assert corrected.filters[0].value == "Ouest"
    assert note is not None and "interprété" in note and "Ouest" in note


@pytest.mark.asyncio
async def test_case_fix_is_silent():
    """"OUest"→"Ouest" (same word, case fix) → no user-facing note."""
    _fresh_caches()
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="OUest")],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members",
                      new=AsyncMock(return_value=REGION_MEMBERS)):
        corrected, note = await main._correct_filter_values(viz, [VENTES])
    assert corrected.filters[0].value == "Ouest"
    assert note is None


# --- FIX-058: intent rescued from a plain-text response ---------------------

INTENT_JSON = {
    "viz_type": "bar_chart", "title": "Sales by Category",
    "x_field": "Category", "y_field": "Sales", "action": "new",
    "filters": [], "calculated_fields": [], "aggregation": "SUM",
}


def test_extract_intent_from_plain_json():
    assert main._extract_intent_from_text(json.dumps(INTENT_JSON)) is not None


def test_extract_intent_from_fenced_json_with_preamble():
    text = "Here is the chart spec:\n```json\n" + json.dumps(INTENT_JSON) + "\n```"
    data = main._extract_intent_from_text(text)
    assert data is not None and data["viz_type"] == "bar_chart"


def test_extract_intent_rejects_conversation_text():
    assert main._extract_intent_from_text("Un treemap sert à visualiser {des proportions}.") is None
    assert main._extract_intent_from_text("viz_type is a field of the intent schema") is None
    assert main._extract_intent_from_text("") is None


def test_extract_intent_rejects_json_without_required_keys():
    assert main._extract_intent_from_text(json.dumps({"viz_type": "bar_chart"})) is None
    assert main._extract_intent_from_text(json.dumps({"x_field": "Category"})) is None


@pytest.mark.asyncio
async def test_pipeline_rescues_text_intent_into_chart(client):
    """End-to-end through /chat: a text-only LLM response carrying the intent JSON
    must produce a CHART (not a conversation bubble with raw JSON)."""
    session_id = str(uuid.uuid4())
    meta = {
        "datasource_name": "superstore",
        "fields": [
            {"name": "Category", "type": "string", "role": "dimension"},
            {"name": "Sales", "type": "float", "role": "measure"},
        ],
    }
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_text_response(
            "Voici le graphique demandé :\n" + json.dumps(INTENT_JSON))
        r = await client.post("/chat", json={
            "question": "sales by category", "session_id": session_id,
            "metadata": meta, "conversation_history": [],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] != "conversation", "the intent must be rescued into a chart"
    assert body["viz_intent"]["viz_type"] == "bar_chart"
    assert body["twb_filename"]
