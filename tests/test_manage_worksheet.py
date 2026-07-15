"""C2/C3 — manage_worksheet tool: delete / rename a sheet and undo, by natural
language. Offline: publish is mocked (conftest); the .twb surgery and the cart
bookkeeping are asserted for real on the session workbook.
"""
import uuid
from unittest.mock import AsyncMock, patch

from lxml import etree
import pytest

from conftest import make_tool_response
from config import settings
from llm import LLMResponse, ToolCall
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent
import twb_generator
from twb_generator import (
    delete_sheet_from_workbook, rename_sheet_in_workbook, list_worksheet_titles,
)
import main

META = DataSourceMetadata(
    datasource_name="superstore",
    fields=[
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Region", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
    ],
)
SAMPLE_METADATA = META.model_dump()


def _wb_with_sheets(titles):
    """Build a real multi-sheet workbook via the production path."""
    viz0 = VizIntent(viz_type="bar_chart", title=titles[0], x_field="Category",
                     y_field="Sales", aggregation="SUM", action="new")
    _fn, out = twb_generator.generate_twb(viz0, META)
    for t in titles[1:]:
        viz = VizIntent(viz_type="bar_chart", title=t, x_field="Region",
                        y_field="Profit", aggregation="SUM", action="new")
        twb_generator.add_sheet_to_existing(str(out), viz, META)
    return out


# --- twb primitives ---------------------------------------------------------

def test_delete_sheet_removes_sheet_and_window():
    out = _wb_with_sheets(["A", "B"])
    try:
        assert delete_sheet_from_workbook(str(out), "A") is True
        root = etree.parse(str(out)).getroot()
        assert list_worksheet_titles(str(out)) == ["B"]
        assert all(w.get("name") != "A" for w in root.findall(".//windows/window"))
    finally:
        out.unlink(missing_ok=True)


def test_delete_refuses_last_sheet():
    out = _wb_with_sheets(["Only"])
    try:
        assert delete_sheet_from_workbook(str(out), "Only") is False
        assert list_worksheet_titles(str(out)) == ["Only"]
    finally:
        out.unlink(missing_ok=True)


def test_rename_sheet_updates_sheet_and_window_with_dedup():
    out = _wb_with_sheets(["A", "B"])
    try:
        assert rename_sheet_in_workbook(str(out), "A", "Coûts 2025") == "Coûts 2025"
        assert sorted(list_worksheet_titles(str(out))) == ["B", "Coûts 2025"]
        # renaming B to an existing name gets deduped (FIX-052)
        assert rename_sheet_in_workbook(str(out), "B", "Coûts 2025") == "Coûts 2025 (2)"
        assert rename_sheet_in_workbook(str(out), "Missing", "X") is None
    finally:
        out.unlink(missing_ok=True)


# --- title matching ----------------------------------------------------------

def test_match_sheet_title_variants():
    titles = ["Ventes par Catégorie", "Coût par Région"]
    assert main._match_sheet_title("Ventes par Catégorie", titles) == "Ventes par Catégorie"
    assert main._match_sheet_title("ventes par categorie", titles) == "Ventes par Catégorie"
    assert main._match_sheet_title(None, titles) == "Coût par Région"  # default: last
    assert main._match_sheet_title("région", titles) == "Coût par Région"  # unique substring
    assert main._match_sheet_title("zzz introuvable", titles) is None


# --- pipeline ----------------------------------------------------------------

BAR_A = {"viz_type": "bar_chart", "title": "Sales by Category", "x_field": "Category",
         "y_field": "Sales", "color_field": None, "filters": [], "calculated_fields": [],
         "clarification_needed": None, "sort": None, "aggregation": "SUM",
         "color_scheme": "tableau10", "action": "new"}
BAR_B = {**BAR_A, "title": "Profit by Region", "x_field": "Region", "y_field": "Profit"}


async def _create(client, session_id, intent):
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = make_tool_response(intent)
        r = await client.post("/chat", json={
            "question": intent["title"], "session_id": session_id,
            "metadata": SAMPLE_METADATA, "conversation_history": [],
        })
    assert r.status_code == 200
    return settings.output_dir / f"Analyse_{session_id[:8]}.twb"


def _manage(operation, **kw):
    return LLMResponse(tool_calls=[ToolCall(name="manage_worksheet",
                                            arguments={"operation": operation, **kw})])


@pytest.mark.asyncio
async def test_pipeline_delete_sheet_by_natural_language(client):
    session_id = str(uuid.uuid4())
    await _create(client, session_id, BAR_A)
    wb = await _create(client, session_id, BAR_B)
    try:
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _manage("delete", sheet_title="sales by category")
            r = await client.post("/chat", json={
                "question": "supprime le chart des ventes", "session_id": session_id,
                "metadata": SAMPLE_METADATA, "conversation_history": [],
            })
        assert r.status_code == 200
        assert "supprimée" in r.json()["message"]
        assert list_worksheet_titles(str(wb)) == ["Profit by Region"]
        cart = (await client.get(f"/session/{session_id}/charts")).json()["charts"]
        assert [c["title"] for c in cart] == ["Profit by Region"]
    finally:
        wb.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_pipeline_rename_sheet(client):
    session_id = str(uuid.uuid4())
    await _create(client, session_id, BAR_A)
    wb = await _create(client, session_id, BAR_B)
    try:
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _manage("rename", sheet_title="profit by region",
                                            new_title="Bénéfices 2025")
            r = await client.post("/chat", json={
                "question": "renomme la feuille profit en Bénéfices 2025",
                "session_id": session_id, "metadata": SAMPLE_METADATA,
                "conversation_history": [],
            })
        assert r.status_code == 200
        assert "Bénéfices 2025" in r.json()["message"]
        assert sorted(list_worksheet_titles(str(wb))) == ["Bénéfices 2025", "Sales by Category"]
        cart = (await client.get(f"/session/{session_id}/charts")).json()["charts"]
        assert "Bénéfices 2025" in [c["title"] for c in cart]
    finally:
        wb.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_pipeline_undo_removes_last_added_sheet(client):
    session_id = str(uuid.uuid4())
    await _create(client, session_id, BAR_A)
    wb = await _create(client, session_id, BAR_B)
    try:
        with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _manage("undo")
            r = await client.post("/chat", json={
                "question": "annule le dernier chart", "session_id": session_id,
                "metadata": SAMPLE_METADATA, "conversation_history": [],
            })
        assert r.status_code == 200
        assert list_worksheet_titles(str(wb)) == ["Sales by Category"]
        cart = (await client.get(f"/session/{session_id}/charts")).json()["charts"]
        assert [c["title"] for c in cart] == ["Sales by Category"]
    finally:
        wb.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_pipeline_delete_without_workbook_is_friendly(client):
    session_id = str(uuid.uuid4())
    with patch("main.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _manage("delete", sheet_title="anything")
        r = await client.post("/chat", json={
            "question": "supprime le chart", "session_id": session_id,
            "metadata": SAMPLE_METADATA, "conversation_history": [],
        })
    assert r.status_code == 200
    assert "Aucun classeur" in r.json()["message"]
