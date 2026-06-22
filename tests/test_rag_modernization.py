"""Tests for M10 — RAG Modernization (agentic retrieval + datasource context)."""
import pytest
from main import _is_likely_modify
from prompts import _format_rag_context, build_intent_prompt


# ---------------------------------------------------------------------------
# Agentic RAG: _is_likely_modify
# ---------------------------------------------------------------------------


def test_modify_detected_with_keyword_and_previous():
    """Modification keywords + previous intent → skip RAG."""
    assert _is_likely_modify("ajoute un filtre par région", True) is True


def test_modify_detected_english_keywords():
    """English modification keywords also detected."""
    assert _is_likely_modify("add a filter for West region", True) is True


def test_no_modify_without_previous():
    """Even with keywords, no previous intent → don't skip."""
    assert _is_likely_modify("filter by region", False) is False


def test_no_modify_for_new_question():
    """New question without keywords → don't skip."""
    assert _is_likely_modify("show sales by category", True) is False


def test_no_modify_first_turn():
    """First turn (no previous) → never skip."""
    assert _is_likely_modify("show me the trend over time", False) is False


def test_modify_keywords_case_insensitive():
    """Keywords should match case-insensitively."""
    assert _is_likely_modify("CHANGE the chart type", True) is True


def test_modify_maintenant():
    """French 'maintenant' triggers modify detection."""
    assert _is_likely_modify("maintenant filtre par date", True) is True


# ---------------------------------------------------------------------------
# RAG disclaimer in prompt
# ---------------------------------------------------------------------------


def test_rag_context_has_field_disclaimer():
    """When RAG examples are injected, a disclaimer about field names should be present."""
    examples = [{"question": "sales by region", "viz_intent": {"viz_type": "bar_chart"}, "judge_score": 0.92}]
    text = _format_rag_context([], examples)
    # The disclaimer should warn the LLM not to copy field names from examples
    assert "field names" in text.lower() or "substitute" in text.lower() or "actual" in text.lower()


def test_rag_context_without_examples_no_disclaimer():
    """With no examples, no disclaimer needed."""
    text = _format_rag_context([], [])
    assert text == "" or "substitute" not in text.lower()


# ---------------------------------------------------------------------------
# RAG examples in prompt with datasource context
# ---------------------------------------------------------------------------


def test_rag_examples_injected_into_prompt():
    """RAG examples should appear in the prompt's user message."""
    examples = [{"question": "revenue by month", "viz_intent": {"viz_type": "line_chart"}, "judge_score": 0.95}]
    messages = build_intent_prompt("show trends", None, [], rag_examples=examples)
    user_msg = messages[-1]["content"]
    assert "revenue by month" in user_msg
    assert "0.95" in user_msg
