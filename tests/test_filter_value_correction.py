"""Phase-2 Class #5 regression tests — filter VALUE correction (FIX-054).

A filter value the LLM mis-cased / mis-accented / invented ("OUest" vs the real
member "Ouest") publishes fine but, because Tableau string filters are
case-sensitive, selects 0 of N members → a blank chart. These offline tests pin
the matcher and the async correction pass (with member fetch stubbed — no network).
The end-to-end proof (publish + assert the view is non-empty) is in eval_harness.py
(`class5_miscased_region_ouest`, FAIL with --no-value-correction, PASS without).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest

from schemas import VizIntent, FilterSpec, DataSourceMetadata, FieldInfo, FieldType
import main

VENTES = DataSourceMetadata(
    datasource_name="ventes", luid="ventes-luid",
    fields=[
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sub Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
    ],
)
REGION_MEMBERS = ["Centre", "Est", "Sud", "Ouest", "Nord"]


# --- _match_member (pure) -------------------------------------------------

def test_match_member_case_insensitive():
    assert main._match_member("OUest", REGION_MEMBERS) == "Ouest"
    assert main._match_member("ouest", REGION_MEMBERS) == "Ouest"
    assert main._match_member("OUEST", REGION_MEMBERS) == "Ouest"


def test_match_member_exact():
    assert main._match_member("Ouest", REGION_MEMBERS) == "Ouest"


def test_match_member_accent_insensitive():
    assert main._match_member("OUÉST", REGION_MEMBERS) == "Ouest"


def test_match_member_high_confidence_typo():
    assert main._match_member("Oest", REGION_MEMBERS) == "Ouest"


def test_match_member_rejects_ambiguous_translation():
    # "West" (English) must NOT silently map to "Est"; better to return None and warn.
    assert main._match_member("West", REGION_MEMBERS) is None
    assert main._match_member("XYZ", REGION_MEMBERS) is None


# --- _recover_member_from_question (pure) ---------------------------------

def test_recover_member_from_question_finds_user_word():
    # The LLM translated "OUest"→"West"; the user's own word is still in the question.
    q = "Show sales by sub-category only for the OUest region"
    assert main._recover_member_from_question(q, REGION_MEMBERS) == "Ouest"


def test_recover_member_skips_stopword_member():
    # "Est" must not be recovered from the French verb "est" ("quel est ...").
    q = "Quel est le total des ventes par sous-catégorie"
    assert main._recover_member_from_question(q, REGION_MEMBERS) is None


def test_recover_member_returns_none_when_ambiguous():
    # Two real members named in the question → never guess between them.
    q = "Compare Ouest and Nord regions"
    assert main._recover_member_from_question(q, REGION_MEMBERS) is None


def test_recover_member_none_when_user_word_absent():
    # User wrote the English "West", which is in neither the data nor recoverable.
    q = "Show sales for the West region"
    assert main._recover_member_from_question(q, REGION_MEMBERS) is None


# --- _correct_filter_values (async, member fetch stubbed) -----------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
async def test_correct_eq_filter_value_snapped_to_real_member():
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="OUest")],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)):
        main._member_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES])
    assert corrected.filters[0].value == "Ouest"
    assert warning is None


@pytest.mark.asyncio
async def test_translated_value_recovered_from_question():
    # The reproduced production bug: user typed "OUest", the LLM emitted the English
    # "West" (not a real member). _match_member rightly refuses to guess "Est"; the
    # question-recovery fallback restores the member the user actually wrote.
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="West")],
                    action="new", datasource_luid="ventes-luid")
    q = "Show sales by sub-category only for the OUest region"
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)):
        main._member_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES], q)
    assert corrected.filters[0].value == "Ouest"
    # D4.3 transparency: a semantic change ("West"→"Ouest") is surfaced as an
    # interpretation note — never silent, never an unresolved-value warning.
    assert warning is not None and "interprété" in warning and "Ouest" in warning
    assert "n'existe pas" not in warning


@pytest.mark.asyncio
async def test_english_value_resolved_to_real_member_via_llm():
    # The user's scenario: they ask for the English "West region" not knowing the
    # data stores "Ouest". Deterministic matching can't bridge that, so the LLM is
    # shown the real members and maps "West" → "Ouest" (intent, not literal echo).
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="West")],
                    action="new", datasource_luid="ventes-luid")
    q = "Show sales by sub-category only for the West region"
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)), \
         patch.object(main, "call_llm", new=AsyncMock(return_value=SimpleNamespace(content="Ouest"))):
        main._member_cache.clear()
        main._value_resolution_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES], q)
    assert corrected.filters[0].value == "Ouest"
    # D4.3 transparency: the LLM-resolved interpretation is surfaced to the user.
    assert warning is not None and "interprété" in warning and "Ouest" in warning
    assert "n'existe pas" not in warning


@pytest.mark.asyncio
async def test_llm_resolution_rejects_non_member_answer():
    # Defensive: even if the LLM returns something not in the domain, we never ship
    # it — leave the value intact + warn (no hallucinated member reaches the chart).
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="Atlantis")],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)), \
         patch.object(main, "call_llm", new=AsyncMock(return_value=SimpleNamespace(content="NONE"))):
        main._member_cache.clear()
        main._value_resolution_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES], "Atlantis region")
    assert corrected.filters[0].value == "Atlantis"  # unchanged
    assert warning and "Atlantis" in warning


@pytest.mark.asyncio
async def test_correct_in_filter_values_each_snapped():
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="in", values=["ouest", "NORD"])],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)):
        main._member_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES])
    assert corrected.filters[0].values == ["Ouest", "Nord"]
    assert warning is None


@pytest.mark.asyncio
async def test_unresolvable_value_produces_warning_and_is_left_intact():
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="Atlantis")],
                    action="new", datasource_luid="ventes-luid")
    # LLM also can't map it → returns NONE; value stays intact and we warn.
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=REGION_MEMBERS)), \
         patch.object(main, "call_llm", new=AsyncMock(return_value=SimpleNamespace(content="NONE"))):
        main._member_cache.clear()
        main._value_resolution_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES])
    assert corrected.filters[0].value == "Atlantis"  # unchanged
    assert warning and "Atlantis" in warning and "Region" in warning


@pytest.mark.asyncio
async def test_degrades_gracefully_when_members_unavailable():
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Sub Category", y_field="Sales",
                    filters=[FilterSpec(field="Region", op="eq", value="OUest")],
                    action="new", datasource_luid="ventes-luid")
    with patch.object(main, "get_dimension_members", new=AsyncMock(return_value=[])):
        main._member_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES])
    # No members → no correction, no crash (value left as-is, no false warning).
    assert corrected.filters[0].value == "OUest"
    assert warning is None


@pytest.mark.asyncio
async def test_measure_and_nonmember_ops_are_not_touched():
    # gt on a measure must not trigger member fetch / correction.
    viz = VizIntent(viz_type="bar_chart", title="t", x_field="Region", y_field="Sales",
                    filters=[FilterSpec(field="Sales", op="gt", value=1000)],
                    action="new", datasource_luid="ventes-luid")
    fetch = AsyncMock(return_value=REGION_MEMBERS)
    with patch.object(main, "get_dimension_members", new=fetch):
        main._member_cache.clear()
        corrected, warning = await main._correct_filter_values(viz, [VENTES])
    fetch.assert_not_called()
    assert corrected.filters[0].value == 1000
    assert warning is None
