"""Offline unit tests for the intent-gate scoring metrics."""

import pytest

from qhld_engine.domain.evaluation import gate_scoring
from qhld_engine.domain.evaluation.gate_scoring import (
    PASS, REFUSED_EMPTY, REFUSED_FLAG)

pytestmark = pytest.mark.unit


def _row(id_, outcome=PASS, cls="shape", route="search"):
    return {"id": id_, "class": cls, "query": f"q-{id_}", "outcome": outcome,
            "route": route if outcome == PASS else None}


def test_score_run_counts_every_refusal_whatever_the_site():
    rows = [_row("L1"), _row("L2", REFUSED_FLAG), _row("L3", REFUSED_EMPTY)]
    report = gate_scoring.score_run(rows)
    assert report["refused"] == 2
    assert report["rate"] == round(2 / 3, 4)


def test_score_run_keeps_the_two_sites_apart():
    rows = [_row("L1", REFUSED_FLAG), _row("L2", REFUSED_FLAG),
            _row("L3", REFUSED_EMPTY), _row("L4")]
    report = gate_scoring.score_run(rows)
    assert report["by_site"] == {
        REFUSED_FLAG: 2, REFUSED_EMPTY: 1, gate_scoring.REFUSED_LANGUAGE: 0}
    assert report["by_site_rate"][REFUSED_FLAG] == 0.5


def test_score_run_reports_a_rate_per_hazard_class():
    rows = [_row("L1", REFUSED_FLAG, cls="language"), _row("L2", PASS, cls="language"),
            _row("L3", PASS, cls="route")]
    report = gate_scoring.score_run(rows)
    assert report["by_class"]["language"] == {"n": 2, "refused": 1, "rate": 0.5}
    assert report["by_class"]["route"]["rate"] == 0.0


def test_band_spans_the_passes_rather_than_averaging_them():
    runs = [
        [_row("L1"), _row("L2")],                       # 0.0
        [_row("L1", REFUSED_FLAG), _row("L2")],         # 0.5
        [_row("L1", REFUSED_FLAG), _row("L2", REFUSED_FLAG)],  # 1.0
    ]
    report = gate_scoring.score_arm(runs)
    assert report["band"] == {"min": 0.0, "median": 0.5, "max": 1.0}


def test_a_query_refused_on_some_passes_only_is_flagged_unstable():
    runs = [[_row("L1")], [_row("L1", REFUSED_FLAG)], [_row("L1")]]
    report = gate_scoring.score_arm(runs)
    unstable = report["unstable"]
    assert [e["id"] for e in unstable] == ["L1"]
    assert unstable[0]["frequency"] == round(1 / 3, 4)


def test_a_query_refused_every_pass_is_not_unstable():
    runs = [[_row("L1", REFUSED_FLAG)], [_row("L1", REFUSED_FLAG)]]
    report = gate_scoring.score_arm(runs)
    assert report["unstable"] == []
    assert report["always"] == 1


def test_ever_refused_exceeds_the_median_rate_when_refusals_are_intermittent():
    """The point of reporting both: three queries each refused once out of three
    passes read as a 1/3 rate on any pass, but every one of them is broken for
    some user."""
    runs = [
        [_row("L1", REFUSED_FLAG), _row("L2"), _row("L3")],
        [_row("L1"), _row("L2", REFUSED_FLAG), _row("L3")],
        [_row("L1"), _row("L2"), _row("L3", REFUSED_FLAG)],
    ]
    report = gate_scoring.score_arm(runs)
    assert report["band"]["median"] == round(1 / 3, 4)
    assert report["ever"] == 3 and report["ever_rate"] == 1.0
    assert report["always"] == 0


def test_a_query_can_be_refused_under_both_sites_across_passes():
    runs = [[_row("L1", REFUSED_FLAG)], [_row("L1", REFUSED_EMPTY)]]
    entry = gate_scoring.score_arm(runs)["queries"][0]
    assert entry["sites"] == [REFUSED_EMPTY, REFUSED_FLAG]


def test_routes_record_that_a_passing_query_only_browsed():
    runs = [[_row("L1", PASS, route="browse")], [_row("L1", PASS, route="browse")]]
    assert gate_scoring.score_arm(runs)["queries"][0]["routes"] == ["browse"]


def test_an_arm_with_no_refusals_reports_a_ceiling_not_a_measured_zero():
    runs = [[_row(f"L{i}") for i in range(1, 25)] for _ in range(5)]
    report = gate_scoring.score_arm(runs)
    assert report["band"]["median"] == 0.0
    # 3/24 — the rule of three over QUERIES, not over query-passes: repeating a
    # probe measures the model's stability, not the product's coverage.
    assert report["ceiling_95"] == 0.125


def test_the_ceiling_is_absent_once_anything_was_refused():
    runs = [[_row("L1", REFUSED_FLAG), _row("L2")], [_row("L1"), _row("L2")]]
    assert gate_scoring.score_arm(runs)["ceiling_95"] is None


def test_score_arm_rejects_an_empty_run_list():
    with pytest.raises(ValueError):
        gate_scoring.score_arm([])


def test_confusion_reads_a_legitimate_refusal_as_a_false_positive():
    legitimate = gate_scoring.score_arm([[_row(f"L{i}") for i in range(1, 5)]])
    junk = gate_scoring.score_arm(
        [[_row(f"J{i}", REFUSED_FLAG) for i in range(1, 5)]])
    matrix = gate_scoring.confusion(legitimate, junk)
    assert matrix == {
        "tp": 4, "fp": 0, "tn": 4, "fn": 0,
        "false_positive_rate": 0.0, "suppression": 1.0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


def test_confusion_precision_falls_when_the_gate_refuses_real_searches():
    legitimate = gate_scoring.score_arm(
        [[_row("L1", REFUSED_FLAG), _row("L2"), _row("L3"), _row("L4")]])
    junk = gate_scoring.score_arm(
        [[_row(f"J{i}", REFUSED_FLAG) for i in range(1, 5)]])
    matrix = gate_scoring.confusion(legitimate, junk)
    assert (matrix["fp"], matrix["tn"]) == (1, 3)
    assert matrix["false_positive_rate"] == 0.25
    assert matrix["precision"] == 0.8


def test_is_refusal_treats_only_pass_as_a_pass():
    assert not gate_scoring.is_refusal(PASS)
    assert gate_scoring.is_refusal(REFUSED_FLAG)
    assert gate_scoring.is_refusal(REFUSED_EMPTY)
    assert gate_scoring.is_refusal(gate_scoring.REFUSED_LANGUAGE)


def test_a_refusal_for_the_wrong_reason_is_counted_separately():
    """Refusing a French query as 'not a speech search' is a correct verdict with a
    useless explanation. Refused/passed counts cannot see it."""
    rows = [
        {**_row("U1", gate_scoring.REFUSED_FLAG), "expected_reason": "unsupported_language"},
        {**_row("U2", gate_scoring.REFUSED_LANGUAGE), "expected_reason": "unsupported_language"},
    ]
    report = gate_scoring.score_run(rows)
    assert report["refused"] == 2          # both refused...
    assert report["wrong_reason"] == 1     # ...but one with the wrong message
    assert report["wrong_reason_ids"] == ["U1"]


def test_either_intent_site_satisfies_a_not_a_speech_search_expectation():
    rows = [
        {**_row("N1", REFUSED_FLAG), "expected_reason": "not_a_speech_search"},
        {**_row("N2", REFUSED_EMPTY), "expected_reason": "not_a_speech_search"},
    ]
    assert gate_scoring.score_run(rows)["wrong_reason"] == 0


def test_rows_without_an_expected_reason_are_never_judged_on_reason():
    """The legitimate arm states no reason — nothing there should be refused at
    all, and a refusal is already counted as a false positive."""
    rows = [_row("L1", REFUSED_FLAG)]
    assert gate_scoring.score_run(rows)["wrong_reason"] == 0
