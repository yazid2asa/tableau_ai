"""C1b — cross-datasource query_data: measure in one DS, group-by / filter
dimension in another, joined on the linking field via two VDS queries.

Live-verified on the real Server (2026-07-08): COUNTD(Trip Id) filtered
Brand=Ford → 30 (the broken blended KPI showed 1000); AVG(Cost Eur) by Brand
→ 16 exact trip-weighted averages (Ford 204.57 … Iveco 654.39), whose global
consistency (241.95) matches the measured blend fallback value.

Before C1b, foreign filter fields were silently DROPPED (the Ford question
answered the unfiltered total) and a foreign group_by answered ungrouped.
"""

import pytest

import main
from schemas import DataSourceMetadata, FieldInfo, FieldType


TRIPS = DataSourceMetadata(
    datasource_name="trips",
    luid="trips-luid",
    fields=[
        FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Trip Id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Cost Eur", type=FieldType.FLOAT, role="measure"),
        FieldInfo(name="Status", type=FieldType.STRING, role="dimension"),
    ],
)

VEHICLES = DataSourceMetadata(
    datasource_name="vehicles",
    luid="vehicles-luid",
    fields=[
        FieldInfo(name="Vehicle Id", type=FieldType.STRING, role="dimension"),
        FieldInfo(name="Brand", type=FieldType.STRING, role="dimension"),
    ],
)

ALL_DS = [TRIPS, VEHICLES]


@pytest.fixture(autouse=True)
def _clear_member_cache():
    main._member_cache.clear()
    yield
    main._member_cache.clear()


@pytest.fixture
def vds(monkeypatch):
    """Mock the three VDS primitives; record every call."""
    calls = {"aggregate": [], "pairs": [], "members": []}

    async def fake_members(luid, field, **kw):
        calls["members"].append((luid, field))
        return ["Ford", "Renault"] if field == "Brand" else []

    async def fake_pairs(luid, field_a, field_b=None, filters=None, limit=2000):
        calls["pairs"].append((luid, field_a, field_b, filters))
        assert luid == VEHICLES.luid
        if field_b:  # linking → group mapping
            return [{"Vehicle Id": "V1", "Brand": "Ford"},
                    {"Vehicle Id": "V2", "Brand": "Ford"},
                    {"Vehicle Id": "V3", "Brand": "Renault"}]
        # filtered linking SET (Brand=Ford)
        return [{"Vehicle Id": "V1"}, {"Vehicle Id": "V2"}]

    async def fake_aggregate(luid, measure, agg, group_by=None, filters=None, limit=50):
        calls["aggregate"].append((luid, measure, agg, group_by, filters))
        assert luid == TRIPS.luid
        link_filter = next((f for f in (filters or []) if f["field"] == "Vehicle Id"), None)
        links = set(link_filter["values"]) if link_filter else set()
        per_link_avg = {"V1": 100.0, "V2": 300.0, "V3": 50.0}
        per_link_trips = {"V1": 10, "V2": 10, "V3": 5}
        if agg == "AVG":
            tot = sum(per_link_avg[v] * per_link_trips[v] for v in links)
            n = sum(per_link_trips[v] for v in links)
            return [{f"AVG(Cost Eur)": tot / n}] if n else []
        if agg == "COUNTD":
            return [{"COUNTD(Trip Id)": sum(per_link_trips[v] for v in links)}]
        return []

    monkeypatch.setattr(main, "get_dimension_members", fake_members)
    monkeypatch.setattr(main, "query_dimension_pairs", fake_pairs)
    monkeypatch.setattr(main, "query_datasource_aggregate", fake_aggregate)
    return calls


async def test_cross_ds_filter_only(vds):
    """'Combien de voyages des véhicules Ford ?' — the linking SET from the
    dimension DS filters ONE exact aggregate on the measure DS."""
    answer = await main._answer_data_question(
        {"measure": "Trip Id", "aggregation": "COUNTD",
         "filters": [{"field": "Brand", "values": ["Ford"]}]},
        ALL_DS, "combien de voyages des véhicules Ford")
    assert "20" in answer                     # V1 (10) + V2 (10)
    assert "Ford" in answer                   # the foreign filter is displayed
    # one pairs call on vehicles, one aggregate on trips with the link IN set
    assert len(vds["pairs"]) == 1 and vds["pairs"][0][2] is None
    (_l, _m, _a, group, filters) = vds["aggregate"][0]
    assert group is None
    assert sorted(next(f["values"] for f in filters if f["field"] == "Vehicle Id")) == ["V1", "V2"]


async def test_cross_ds_group_by(vds):
    """'Coût moyen par marque' — per-group aggregates on the measure DS,
    weighted correctly (NOT an average of per-vehicle averages)."""
    answer = await main._answer_data_question(
        {"measure": "Cost Eur", "aggregation": "AVG", "group_by": "Brand"},
        ALL_DS, "coût moyen des voyages par marque")
    # Ford = (100*10 + 300*10) / 20 = 200 ; Renault = 50
    assert "Ford" in answer and "200" in answer
    assert "Renault" in answer and "50" in answer
    assert len(vds["aggregate"]) == 2         # one exact query per group


async def test_single_ds_path_untouched(vds):
    """No foreign field → the classic single-DS path, no pairs call."""
    async def fake_single(luid, measure, agg, group_by=None, filters=None, limit=50):
        return [{"Status": "Completed", "AVG(Cost Eur)": 250.0}]
    main.query_datasource_aggregate = fake_single  # type: ignore
    answer = await main._answer_data_question(
        {"measure": "Cost Eur", "aggregation": "AVG", "group_by": "Status"},
        ALL_DS, "coût moyen par statut")
    assert "Completed" in answer
    assert not vds["pairs"]


async def test_no_partner_datasource_message(vds):
    """A field no datasource owns → explicit message, never a silently
    unfiltered answer."""
    answer = await main._answer_data_question(
        {"measure": "Cost Eur", "aggregation": "AVG",
         "filters": [{"field": "Nonexistent Thing", "values": ["X"]}]},
        [TRIPS], "coût moyen filtré")
    assert "impossible de croiser" in answer
    assert not vds["aggregate"]


async def test_too_many_groups_message(vds, monkeypatch):
    async def many_pairs(luid, field_a, field_b=None, filters=None, limit=2000):
        return [{"Vehicle Id": f"V{i}", "Brand": f"B{i}"} for i in range(30)]
    monkeypatch.setattr(main, "query_dimension_pairs", many_pairs)
    answer = await main._answer_data_question(
        {"measure": "Cost Eur", "aggregation": "AVG", "group_by": "Brand"},
        ALL_DS, "coût moyen par marque")
    assert "trop" in answer.lower()
    assert not vds["aggregate"]
