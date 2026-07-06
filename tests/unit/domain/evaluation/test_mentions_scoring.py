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
