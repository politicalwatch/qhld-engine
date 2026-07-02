"""Offline unit tests for the A/B benchmark scoring metrics."""

import pytest

from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.evaluation import scoring

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


def test_reciprocal_rank():
    assert scoring.reciprocal_rank(1) == 1.0
    assert scoring.reciprocal_rank(4) == 0.25
    assert scoring.reciprocal_rank(None) == 0.0


def test_hit_at_k():
    assert scoring.hit_at_k(5, 5) is True
    assert scoring.hit_at_k(6, 5) is False
    assert scoring.hit_at_k(None, 5) is False


def test_aggregate_per_dimension_and_overall():
    rows = [
        {"dimension": "topical", "rank": 1},
        {"dimension": "topical", "rank": None},
        {"dimension": "crosslingual", "rank": 2},
    ]
    agg = scoring.aggregate(rows, k=5)

    assert agg["topical"]["n"] == 2
    assert agg["topical"]["mrr"] == 0.5          # (1/1 + 0) / 2
    assert agg["topical"]["hit_at_5"] == 0.5     # one of two within top-5
    assert agg["crosslingual"]["mrr"] == 0.5     # 1/2
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["mrr"] == round((1.0 + 0.0 + 0.5) / 3, 4)
