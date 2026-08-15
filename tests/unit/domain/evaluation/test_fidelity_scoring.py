"""Offline unit tests for the retrieval-layer fidelity metrics."""

import pytest

from qhld_engine.domain.evaluation import fidelity_scoring as fs

pytestmark = pytest.mark.unit


def test_recall_is_set_agreement_not_order():
    assert fs.recall_at_k(["c", "b", "a"], ["a", "b", "c"]) == 1.0


def test_recall_counts_the_fraction_of_truth_that_was_found():
    assert fs.recall_at_k(["a", "b", "z"], ["a", "b", "c", "d"]) == 0.5


def test_recall_truncates_both_sides_at_k():
    assert fs.recall_at_k(["a", "b", "z"], ["a", "b", "c"], k=2) == 1.0


def test_recall_is_none_when_there_was_nothing_to_find():
    # A filter matching no points carries no evidence: averaging it in as 0.0
    # would report a fidelity failure where no search happened.
    assert fs.recall_at_k(["a"], []) is None


def test_aggregate_drops_queries_with_no_ground_truth():
    result = fs.aggregate([1.0, None, 0.5])
    assert result["n"] == 2
    assert result["mean"] == 0.75


def test_aggregate_counts_only_exact_agreement_as_perfect():
    result = fs.aggregate([1.0, 0.99, 1.0])
    assert result["perfect"] == 2
    assert result["worst"] == 0.99


def test_aggregate_of_nothing_reports_no_mean_rather_than_zero():
    assert fs.aggregate([None, None]) == {
        "n": 0, "mean": None, "perfect": 0, "worst": None}


def test_paired_bootstrap_finds_a_consistent_gain_significant():
    baseline = [0.90] * 40
    arm = [0.95] * 40
    result = fs.paired_bootstrap(baseline, arm, resamples=2000)
    assert result["significant"]
    assert result["delta"] == pytest.approx(0.05)


def test_paired_bootstrap_calls_noise_not_significant():
    baseline = [0.9, 0.5, 0.7, 0.3] * 10
    arm = [0.5, 0.9, 0.3, 0.7] * 10
    assert not fs.paired_bootstrap(baseline, arm, resamples=2000)["significant"]


def test_paired_bootstrap_is_seeded_so_a_rerun_is_identical():
    baseline = [0.1, 0.9, 0.4, 0.6, 0.5] * 8
    arm = [0.2, 0.8, 0.5, 0.7, 0.4] * 8
    first = fs.paired_bootstrap(baseline, arm, resamples=2000)
    assert first == fs.paired_bootstrap(baseline, arm, resamples=2000)


def test_paired_bootstrap_drops_a_query_missing_from_either_arm():
    result = fs.paired_bootstrap([1.0, None, 0.5], [1.0, 0.5, None], resamples=500)
    assert result["n"] == 1


def test_paired_bootstrap_of_nothing_comparable_is_none():
    assert fs.paired_bootstrap([None], [None]) is None


def test_first_miss_reports_where_the_ranking_broke():
    # Truth ranks a,b,c,d; the arm missed c, the third.
    assert fs.reciprocal_rank_agreement(["a", "b", "d"], ["a", "b", "c", "d"]) == 1 / 3


def test_first_miss_is_zero_when_nothing_was_missed():
    assert fs.reciprocal_rank_agreement(["a", "b"], ["a", "b"]) == 0.0


def test_summarise_misses_reports_the_shallowest_as_the_damaging_one():
    # 1/2 is a shallower miss than 1/10 and is the one that can change results.
    assert fs.summarise_misses([1 / 10, 1 / 2, 0.0])["shallowest"] == 2


def test_cell_label_names_every_knob_that_is_set():
    label = fs.cell_label({"hnsw_ef": 1024, "rescore": True, "oversampling": 2.0})
    assert label == "ef=1024 rescore=on oversampling=2.0"


def test_cell_label_says_server_default_rather_than_none():
    assert fs.cell_label({"hnsw_ef": None}) == "ef=server-default"


def test_cell_label_distinguishes_rescore_off_from_rescore_unset():
    assert "rescore=off" in fs.cell_label({"hnsw_ef": 8, "rescore": False})
    assert "rescore" not in fs.cell_label({"hnsw_ef": 8})


def test_fingerprint_ignores_the_order_ids_came_back_in():
    assert fs.ids_fingerprint(["b", "a", "c"]) == fs.ids_fingerprint(["a", "c", "b"])


def test_fingerprint_changes_when_the_sample_changes():
    assert fs.ids_fingerprint(["a", "b"]) != fs.ids_fingerprint(["a", "b", "c"])


def test_fingerprint_reads_ids_as_text_so_int_and_str_agree():
    assert fs.ids_fingerprint([1, 2]) == fs.ids_fingerprint(["1", "2"])


def test_format_recall_keeps_a_missing_value_visible():
    assert fs.format_recall(None).strip() == "—"
    assert fs.format_recall(0.5) == "0.5000"
