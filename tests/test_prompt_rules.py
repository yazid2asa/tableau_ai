"""Tests for M9 — Prompt decision tree and rule injection."""
import pytest
from prompts import build_intent_prompt
from schemas import DataSourceMetadata, FieldInfo, FieldType, VizIntent


# --- SYSTEM_PROMPT content tests ---

def test_system_prompt_contains_chart_decision_tree():
    """SYSTEM_PROMPT should include chart type guidance."""
    messages = build_intent_prompt("show sales", None, [])
    system = messages[0]["content"]
    assert "CHART TYPE" in system or "chart type" in system.lower()
    assert "line_chart" in system
    assert "bar_chart" in system


def test_system_prompt_contains_field_role_rules():
    """SYSTEM_PROMPT should include field role rules."""
    messages = build_intent_prompt("show sales", None, [])
    system = messages[0]["content"]
    assert "dimension" in system.lower()
    assert "measure" in system.lower()
    # x_field must be dimension rule
    assert "x_field" in system and "dimension" in system


def test_system_prompt_contains_aggregation_rules():
    """SYSTEM_PROMPT should include aggregation guidance."""
    messages = build_intent_prompt("show sales", None, [])
    system = messages[0]["content"]
    assert "SUM" in system
    assert "COUNT" in system


def test_system_prompt_contains_implicit_filter_rules():
    """SYSTEM_PROMPT should include implicit filter rules."""
    messages = build_intent_prompt("show sales", None, [])
    system = messages[0]["content"]
    assert "IMPLICIT FILTER" in system or "implicit" in system.lower()
    assert "top_n" in system or "top 10" in system.lower()


def test_system_prompt_contains_datasource_selection_rules():
    """SYSTEM_PROMPT should include datasource selection guidance."""
    messages = build_intent_prompt("show sales", None, [])
    system = messages[0]["content"]
    assert "datasource" in system.lower() or "DATASOURCE" in system
    assert "field match" in system.lower() or "field matching" in system.lower()


# --- build_intent_prompt context injection ---

def test_previous_intent_injected():
    """When previous_intent provided, it appears in the user message."""
    prev = VizIntent(
        viz_type="bar_chart", title="Sales by Region",
        x_field="Region", y_field="Sales", aggregation="SUM",
    )
    messages = build_intent_prompt("add a filter", None, [], previous_intent=prev)
    user_msg = messages[-1]["content"]
    assert "bar_chart" in user_msg
    assert "Region" in user_msg
    assert "modify" in user_msg.lower() or "previous" in user_msg.lower()


def test_datasources_injected_in_server_mode():
    """When available_datasources provided, all are listed in user message."""
    ds1 = DataSourceMetadata(
        datasource_name="Sales DB", luid="luid-1",
        fields=[FieldInfo(name="Revenue", type=FieldType.FLOAT, role="measure")],
    )
    ds2 = DataSourceMetadata(
        datasource_name="HR Data", luid="luid-2",
        fields=[FieldInfo(name="Employee", type=FieldType.STRING, role="dimension")],
    )
    messages = build_intent_prompt("show revenue", None, [], available_datasources=[ds1, ds2])
    user_msg = messages[-1]["content"]
    assert "Sales DB" in user_msg
    assert "HR Data" in user_msg
    assert "luid-1" in user_msg
    assert "Revenue" in user_msg


def test_rag_examples_injected():
    """When RAG examples provided, they appear in user message."""
    examples = [{"question": "sales by region", "viz_intent": {"viz_type": "bar_chart"}, "judge_score": 0.92}]
    messages = build_intent_prompt("show sales", None, [], rag_examples=examples)
    user_msg = messages[-1]["content"]
    assert "sales by region" in user_msg
    assert "0.92" in user_msg


def test_history_included():
    """Conversation history (up to 6 turns) is included."""
    history = [
        {"role": "user", "content": "show sales by category"},
        {"role": "assistant", "content": '{"viz_type":"bar_chart"}'},
    ]
    messages = build_intent_prompt("now filter by West", None, history)
    # History messages should be between system and final user message
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "show sales by category"
    assert messages[2]["role"] == "assistant"


def test_no_metadata_message():
    """When no metadata provided, user message says so."""
    messages = build_intent_prompt("show sales", None, [])
    user_msg = messages[-1]["content"]
    assert "no data source" in user_msg.lower() or "infer" in user_msg.lower()
