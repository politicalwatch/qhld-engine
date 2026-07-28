"""Unit tests for the pure mention-eval scoring."""

import pytest

from qhld_engine.domain.evaluation import mentions_scoring

pytestmark = pytest.mark.unit


def test_perfect_prediction_scores_one():
    rows = [{"pred_deputies": ["A", "B"], "gold_deputies": ["A", "B"]}]
    report = mentions_scoring.score(rows)
    assert report["micro"]["precision"] == 1.0
    assert report["micro"]["recall"] == 1.0
    assert report["micro"]["f1"] == 1.0
    assert report["exact_match"] == 1.0


def test_counts_tp_fp_fn_across_speeches():
    rows = [
        {"pred_deputies": ["A", "X"], "gold_deputies": ["A", "B"]},  # tp A, fp X, fn B
        {"pred_deputies": ["C"], "gold_deputies": ["C"]},            # tp C
    ]
    micro = mentions_scoring.score(rows)["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (2, 1, 1)


def test_exact_match_is_per_speech_set_equality():
    rows = [
        {"pred_deputies": ["A", "B"], "gold_deputies": ["B", "A"]},  # exact (order-free)
        {"pred_deputies": ["A"], "gold_deputies": ["A", "B"]},       # not exact
    ]
    assert mentions_scoring.score(rows)["exact_match"] == 0.5


def test_empty_gold_and_pred_is_exact_match():
    rows = [{"pred_deputies": [], "gold_deputies": []}]
    report = mentions_scoring.score(rows)
    assert report["exact_match"] == 1.0
    assert report["micro"]["f1"] == 0.0  # no tp/fp/fn → 0


def test_mean_latency_ignores_missing():
    rows = [
        {"pred_deputies": [], "gold_deputies": [], "latency": 0.2},
        {"pred_deputies": [], "gold_deputies": []},
    ]
    assert mentions_scoring.score(rows)["mean_latency"] == 0.2


def test_counts_ignore_speeches_and_people_without_gold_counts():
    rows = [
        {"pred_counts": {"A": 3}},                                # no gold map at all
        {"pred_counts": {"A": 3, "B": 9}, "expected_counts": {"A": 3}},  # B unannotated
    ]
    report = mentions_scoring.score_counts(rows)
    assert (report["speeches"], report["pairs"], report["exact"]) == (1, 1, 1)
    assert report["over"] == 0  # B's 9 occurrences are out of scope, not an error


def test_counts_split_under_and_over_instead_of_netting_them():
    rows = [{"pred_counts": {"A": 1, "B": 5}, "expected_counts": {"A": 4, "B": 2}}]
    report = mentions_scoring.score_counts(rows)
    assert (report["under"], report["over"]) == (3, 3)
    assert (report["gold_occurrences"], report["pred_occurrences"]) == (6, 6)
    assert report["exact_rate"] == 0.0


def test_a_person_not_predicted_at_all_counts_as_zero_and_is_flagged_missing():
    rows = [{"pred_counts": {}, "expected_counts": {"A": 2}}]
    report = mentions_scoring.score_counts(rows)
    assert (report["under"], report["missing"], report["pred_occurrences"]) == (2, 1, 0)


def test_count_mismatches_list_only_the_differing_pairs():
    rows = [{
        "id": "M61", "reference": "210/000112",
        "pred_counts": {"A": 28, "B": 2},
        "expected_counts": {"A": 30, "B": 2},
    }]
    assert mentions_scoring.count_mismatches(rows) == [
        {"id": "M61", "reference": "210/000112", "name": "A", "gold": 30, "pred": 28}]
