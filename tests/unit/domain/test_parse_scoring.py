"""Unit tests for the query-parser comparison scoring — pure, offline."""

import pytest

from qhld_engine.domain.evaluation import parse_scoring as ps

pytestmark = pytest.mark.unit


def test_date_matches_within_tolerance():
    assert ps.date_matches({"gte": 20250704, "lte": 20260703}, {"gte": 20250703, "lte": 20260703})


def test_date_matches_rejects_out_of_tolerance():
    assert not ps.date_matches({"gte": 20250801}, {"gte": 20250703})


def test_date_matches_rejects_missing_bound():
    # gold has a closed range; a point prediction (gte only) is wrong.
    assert not ps.date_matches({"gte": 20240703}, {"gte": 20240101, "lte": 20241231})


def test_value_matches_any_of():
    assert ps.value_matches("GS", ["GS", "GMx"], "group")
    assert not ps.value_matches("GP", ["GS", "GMx"], "group")


def test_slot_counts_true_positive():
    assert ps.slot_counts({"group": "GS"}, {"group": "GS"}, "group") == (1, 0, 0)


def test_slot_counts_wrong_value_is_fp_and_fn():
    assert ps.slot_counts({"group": "GP"}, {"group": "GS"}, "group") == (0, 1, 1)


def test_slot_counts_missing_is_fn_only():
    assert ps.slot_counts({}, {"group": "GS"}, "group") == (0, 0, 1)


def test_slot_counts_hallucinated_is_fp_only():
    assert ps.slot_counts({"group": "GS"}, {}, "group") == (0, 1, 0)


def test_slot_counts_true_negative_is_zero():
    assert ps.slot_counts({}, {}, "group") == (0, 0, 0)


def test_topic_f1_perfect_and_disjoint():
    assert ps.topic_f1("financiación autonómica", "financiación autonómica") == 1.0
    assert ps.topic_f1("vivienda", "pesca") == 0.0


def test_score_aggregates_slots_and_exact_match():
    rows = [
        {"pred_filters": {"speaker": "Montero Cuadrado, María Jesús"},
         "gold": {"speaker": "Montero Cuadrado, María Jesús"},
         "pred_topic": "financiación", "gold_topic": "financiación", "latency": 0.5},
        {"pred_filters": {"group": "GP"},          # wrong: gold is GS
         "gold": {"group": "GS"},
         "pred_topic": "vivienda", "gold_topic": "vivienda", "latency": 0.7},
    ]
    report = ps.score(rows)
    assert report["n"] == 2
    assert report["slots"]["speaker"]["f1"] == 1.0
    assert report["slots"]["group"]["tp"] == 0 and report["slots"]["group"]["fp"] == 1
    assert report["exact_match"] == 0.5          # only the first query fully correct
    assert report["mean_latency"] == 0.6
    assert report["topic_f1"] == 1.0
