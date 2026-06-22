"""Tests for 'modify intelligent' accumulation — a follow-up must never overwrite
previously generated charts.

Behavior:
  - Pure tweak (filter/sort/chart-type; x/y/color unchanged) → modifies the CURRENT
    sheet in place, every other sheet preserved.
  - Structural change (x/y/color differs) → a NEW sheet is added; nothing overwritten.
"""
import uuid
from unittest.mock import AsyncMock, patch

from lxml import etree
import pytest

from conftest import make_tool_response
from config import settings
import twb_generator
from schemas import DataSourceMetadata, FieldInfo, FieldType, FilterSpec, VizIntent


META = DataSourceMetadata(
    datasource_name="superstore",
    fields=[
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Order Date", type=FieldType.DATE, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
    ],
)

SAMPLE_METADATA = META.model_dump()


def _worksheet_titles(path) -> list[str]:
    root = etree.parse(str(path)).getroot()
    return [ws.get("name") for ws in root.iter("worksheet")]


# --------------------------------------------------------------------------- #
# Unit: modify_sheet_in_existing replaces one sheet, preserves the rest
# --------------------------------------------------------------------------- #

def test_modify_sheet_preserves_other_sheets():
    viz_a = VizIntent(viz_type="bar_chart", title="A", x_field="Category",
                      y_field="Sales", aggregation="SUM", action="new")
    _fn, out = twb_generator.generate_twb(viz_a, META)
    try:
        viz_b = VizIntent(viz_type="bar_chart", title="B", x_field="Region",
                          y_field="Profit", aggregation="SUM", action="new")
        twb_generator.add_sheet_to_existing(str(out), viz_b, META)
        assert sorted(_worksheet_titles(out)) == ["A", "B"]

        # In-place modify of B (add a filter); old_title="B"
        viz_b_mod = VizIntent(
            viz_type="bar_chart", title="B", x_field="Region", y_field="Profit",
            filters=[FilterSpec(field="Category", op="eq", value="Furniture")],
            aggregation="SUM", action="modify",
        )
        twb_generator.modify_sheet_in_existing(str(out), viz_b_mod, META, old_title="B")

        titles = _worksheet_titles(out)
        assert titles.count("B") == 1, "B must be replaced, not duplicated"
        assert "A" in titles, "the other sheet must be preserved"
        assert len(titles) == 2
    finally:
        out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Handler: structural follow-up → new sheet; tweak follow-up → in place
# --------------------------------------------------------------------------- #

BAR_BY_CATEGORY = {
    "viz_type": "bar_chart", "title": "Sales by Category",
    "x_field": "Category", "y_field": "Sales", "color_field": None,
    "filters": [], "calculated_fields": [], "clarification_needed": None,
    "sort": None, "aggregation": "SUM", "color_scheme": "tableau10", "action": "new",
}


async def _first_chart(client, session_id):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(BAR_BY_CATEGORY)
        r = await client.post("/chat", json={
            "question": "sales by category", "session_id": session_id,
            "metadata": SAMPLE_METADATA, "conversation_history": [],
        })
    assert r.status_code == 200
    return settings.output_dir / f"Analyse_{session_id[:8]}.twb"


@pytest.mark.asyncio
async def test_structural_followup_adds_new_sheet(client):
    """LLM says 'modify' but x_field changed → treated as a NEW sheet; original kept."""
    session_id = str(uuid.uuid4())
    wb = await _first_chart(client, session_id)
    try:
        structural = {**BAR_BY_CATEGORY, "action": "modify",
                      "title": "Sales by Region", "x_field": "Region"}
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_tool_response(structural)
            r = await client.post("/chat", json={
                "question": "the same by region", "session_id": session_id,
                "metadata": SAMPLE_METADATA, "conversation_history": [],
            })
        assert r.status_code == 200
        titles = _worksheet_titles(wb)
        assert "Sales by Category" in titles, "original chart must NOT be overwritten"
        assert "Sales by Region" in titles
        assert len(titles) == 2
        cart = (await client.get(f"/session/{session_id}/charts")).json()
        assert len(cart["charts"]) == 2
    finally:
        wb.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_tweak_followup_modifies_in_place(client):
    """A pure tweak (add filter, same x/y) updates the current sheet, no new sheet."""
    session_id = str(uuid.uuid4())
    wb = await _first_chart(client, session_id)
    try:
        tweak = {**BAR_BY_CATEGORY, "action": "modify",
                 "filters": [{"field": "Order Date", "op": "year", "value": 2024}]}
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_tool_response(tweak)
            r = await client.post("/chat", json={
                "question": "filter on 2024", "session_id": session_id,
                "metadata": SAMPLE_METADATA, "conversation_history": [],
            })
        assert r.status_code == 200
        titles = _worksheet_titles(wb)
        assert titles == ["Sales by Category"], "tweak must stay one sheet, in place"
        cart = (await client.get(f"/session/{session_id}/charts")).json()
        assert len(cart["charts"]) == 1, "in-place modify must not duplicate the cart entry"
        assert cart["charts"][0]["filters"], "cart entry should reflect the added filter"
    finally:
        wb.unlink(missing_ok=True)
