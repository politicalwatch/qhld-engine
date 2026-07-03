"""Offline unit tests for the A/B benchmark scoring metrics."""

import pytest

from qhld_engine.domain.evaluation import scoring
from qhld_engine.domain.ports.vector_store import SearchHit

pytestmark = pytest.mark.unit


def _hit(reference, score=0.5):
    return SearchHit(id="x", score=score, payload={"reference": reference})


def test_first_rank_returns_position_of_first_expected():
    hits = [_hit("A/1"), _hit("B/2"), _hit("C/3")]
    assert scoring.first_rank(hits, ["C/3", "B/2"]) == 2  # B/2 is the earlier match


def test_first_rank_none_when_no_match():
    assert scoring.first_rank([_hit("A/1")], ["Z/9"]) is None


def test_first_rank_ignores_missing_reference_payload():
    hits = [SearchHit(id="x", score=0.9, payload={}), _hit("A/1")]
    assert scoring.first_rank(hits, ["A/1"]) == 2


def test_distinct_refs_dedupes_preserving_order():
    hits = [_hit("A/1"), _hit("A/1"), _hit("B/2"), SearchHit(id="x", score=0.1, payload={})]
    assert scoring.distinct_refs(hits) == ["A/1", "B/2"]


def test_reciprocal_rank():
    assert scoring.reciprocal_rank(1) == 1.0
    assert scoring.reciprocal_rank(4) == 0.25
    assert scoring.reciprocal_rank(None) == 0.0


def test_hit_at_k():
    assert scoring.hit_at_k(5, 5) is True
    assert scoring.hit_at_k(6, 5) is False
    assert scoring.hit_at_k(None, 5) is False


def test_recall_at_k_counts_distinct_expected_found():
    ranked = ["A/1", "B/2", "C/3", "D/4"]
    assert scoring.recall_at_k(ranked, ["A/1", "C/3"], k=3) == 1.0
    assert scoring.recall_at_k(ranked, ["A/1", "D/4"], k=3) == 0.5  # D/4 beyond k
    assert scoring.recall_at_k(ranked, [], k=3) == 0.0


def test_average_precision():
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    ranked = ["A/1", "B/2", "C/3"]
    assert scoring.average_precision(ranked, ["A/1", "C/3"]) == pytest.approx((1.0 + 2 / 3) / 2)
    # one relevant never retrieved -> normalised by 2 relevant, only rank-1 found
    assert scoring.average_precision(ranked, ["A/1", "Z/9"]) == pytest.approx(1.0 / 2)
    assert scoring.average_precision(ranked, []) == 0.0


def test_aggregate_per_dimension_and_overall():
    rows = [
        {"dimension": "topical", "rank": 1, "ranked_refs": ["A/1", "B/2"], "expected_refs": ["A/1"]},
        {"dimension": "topical", "rank": None, "ranked_refs": ["X/9"], "expected_refs": ["Z/8"]},
        {"dimension": "crosslingual", "rank": 2, "ranked_refs": ["P/1", "Q/2"], "expected_refs": ["Q/2"]},
    ]
    agg = scoring.aggregate(rows, k=5)

    assert agg["topical"]["n"] == 2
    assert agg["topical"]["mrr"] == 0.5          # (1/1 + 0) / 2
    assert agg["topical"]["hit_at_5"] == 0.5
    assert agg["topical"]["recall_at_5"] == 0.5  # one of two queries fully recalled
    assert agg["topical"]["map"] == 0.5          # (1.0 + 0.0) / 2
    assert agg["crosslingual"]["mrr"] == 0.5     # 1/2
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["mrr"] == round((1.0 + 0.0 + 0.5) / 3, 4)
