"""Regression tests for FIX-057 — calculated-field formula reference validation.

auto_correct_intent_fields fixes x/y/color and filter field names but never
looked INSIDE calculated-field formulas: a typo'd ref ([Sale] for [Sales])
published a broken pill ("The calculation contains errors"). These tests pin
_correct_calc_field_formulas: exact/normalized/fuzzy correction, other-calc-name
refs, the clarification path for an unresolvable ref on a shelf-used calc field,
and the silent drop of an unused broken calc field.
"""
from main import _correct_calc_field_formulas
from schemas import CalculatedField, DataSourceMetadata, FieldInfo, FieldType, VizIntent

DS = DataSourceMetadata(
    datasource_name="ventes",
    fields=[
        FieldInfo(name="Sales", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Profit", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Order ID", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Category", type=FieldType.STRING, role="dimension"),
    ],
    luid="ventes-luid",
)


def _viz(calc_fields, y_field="Marge"):
    return VizIntent(
        viz_type="bar_chart", title="T", x_field="Category", y_field=y_field,
        calculated_fields=calc_fields, aggregation="SUM", action="new",
        datasource_luid="ventes-luid",
    )


def test_typo_ref_is_fuzzy_corrected():
    """[Sale] → [Sales] (difflib), [Profits] → [Profit]."""
    viz = _viz([CalculatedField(name="Marge", formula="SUM([Profits])/SUM([Sale])")])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.calculated_fields[0].formula == "SUM([Profit])/SUM([Sales])"
    assert out.clarification_needed is None


def test_normalized_ref_is_corrected():
    """[order_id] matches "Order ID" by normalization (space vs underscore/case)."""
    viz = _viz([CalculatedField(name="Marge", formula="SUM([Sales])/COUNTD([order_id])")])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.calculated_fields[0].formula == "SUM([Sales])/COUNTD([Order ID])"


def test_valid_formula_untouched():
    viz = _viz([CalculatedField(name="Marge", formula="SUM([Profit])/SUM([Sales])")])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.calculated_fields[0].formula == "SUM([Profit])/SUM([Sales])"


def test_ref_to_other_calc_field_is_valid():
    """A formula may reference another calc field of the same intent by name."""
    viz = _viz([
        CalculatedField(name="Marge", formula="SUM([Profit])/SUM([Sales])"),
        CalculatedField(name="Double Marge", formula="[Marge]*2"),
    ], y_field="Double Marge")
    out = _correct_calc_field_formulas(viz, DS)
    assert out.clarification_needed is None
    assert out.calculated_fields[1].formula == "[Marge]*2"


def test_unresolvable_ref_on_shelf_used_calc_asks_clarification():
    """A broken ref in a calc field that IS on a shelf → clarification, never a
    knowingly broken published pill."""
    viz = _viz([CalculatedField(name="Marge", formula="SUM([Chiffre Affaires Net Net])/2")])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.clarification_needed is not None
    assert "Marge" in out.clarification_needed


def test_unresolvable_ref_on_unused_calc_is_dropped():
    """A broken calc field NOT used on any shelf is silently dropped (FIX-044 spirit)."""
    viz = _viz([
        CalculatedField(name="Marge", formula="SUM([Profit])/SUM([Sales])"),
        CalculatedField(name="Zombie", formula="[Champ Qui N Existe Absolument Pas]*3"),
    ])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.clarification_needed is None
    assert [cf.name for cf in out.calculated_fields] == ["Marge"]


def test_mixed_aggregate_row_level_does_not_crash():
    """SUM([A])/[B] (Tableau would reject) is flagged in the log but must not
    block generation nor mutate the formula."""
    viz = _viz([CalculatedField(name="Marge", formula="SUM([Profit])/[Sales]")])
    out = _correct_calc_field_formulas(viz, DS)
    assert out.calculated_fields[0].formula == "SUM([Profit])/[Sales]"
    assert out.clarification_needed is None


def test_no_calc_fields_is_noop():
    viz = _viz([], y_field="Sales")
    out = _correct_calc_field_formulas(viz, DS)
    assert out is viz
