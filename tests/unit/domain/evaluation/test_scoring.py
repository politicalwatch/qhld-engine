"""Offline unit tests for the A/B benchmark scoring metrics."""

import pytest

from qhld_engine.domain.evaluation import scoring
from qhld_ai.domain.ports.vector_store import SearchHit

pytestmark = pytest.mark.unit


def _hit(*references, score=0.5):
    return SearchHit(id="x", score=score, payload={"references": list(references)})


def test_first_rank_returns_position_of_first_expected():
    hits = [_hit("A/1"), _hit("B/2"), _hit("C/3")]
    assert scoring.first_rank(hits, ["C/3", "B/2"]) == 2  # B/2 is the earlier match


def test_first_rank_none_when_no_match():
    assert scoring.first_rank([_hit("A/1")], ["Z/9"]) is None


def test_first_rank_ignores_missing_references_payload():
    hits = [SearchHit(id="x", score=0.9, payload={}), _hit("A/1")]
    assert scoring.first_rank(hits, ["A/1"]) == 2


def test_first_rank_matches_any_reference_of_an_accumulated_debate():
    hits = [_hit("A/1"), _hit("B/2", "C/3")]
    assert scoring.first_rank(hits, ["C/3"]) == 2


def test_distinct_refs_dedupes_preserving_order():
    hits = [_hit("A/1"), _hit("A/1"), _hit("B/2"), SearchHit(id="x", score=0.1, payload={})]
    assert scoring.distinct_refs(hits) == ["A/1", "B/2"]


def test_distinct_refs_counts_every_reference_of_an_accumulated_debate():
    # one deduplicated speech addressing two initiatives ranks both at once
    hits = [_hit("A/1", "B/2"), _hit("B/2"), _hit("C/3")]
    assert scoring.distinct_refs(hits) == ["A/1", "B/2", "C/3"]


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


def _row(query_id, hits, expected=(), rejected=()):
    return {
        "id": query_id,
        "hits": hits,
        "expected_refs": list(expected),
        "rejected_refs": list(rejected),
    }


def test_pool_candidates_excludes_judged_refs():
    rows = [_row("T1", [_hit("A/1"), _hit("B/2"), _hit("C/3")],
                 expected=["A/1"], rejected=["B/2"])]
    pooled = scoring.pool_candidates({"cell": rows})
    assert [c["ref"] for c in pooled["T1"]] == ["C/3"]
    assert pooled["T1"][0]["rank"] == 3


def test_pool_candidates_keeps_best_rank_across_cells():
    cell_a = [_row("T1", [_hit("X/9"), _hit("A/1", score=0.4)])]
    cell_b = [_row("T1", [_hit("A/1", score=0.9), _hit("Y/8")])]
    pooled = scoring.pool_candidates({"a": cell_a, "b": cell_b})
    best = next(c for c in pooled["T1"] if c["ref"] == "A/1")
    assert (best["cell"], best["rank"], best["score"]) == ("b", 1, 0.9)


def test_pool_candidates_covers_every_reference_of_an_accumulated_debate():
    rows = [_row("T1", [_hit("A/1", "B/2")], expected=["A/1"])]
    pooled = scoring.pool_candidates({"cell": rows})
    assert [c["ref"] for c in pooled["T1"]] == ["B/2"]


def test_pool_candidates_orders_by_rank_and_reports_empty_queries():
    rows = [
        _row("T1", [_hit("B/2"), _hit("A/1")]),
        _row("T2", [_hit("Z/9")], expected=["Z/9"]),
    ]
    pooled = scoring.pool_candidates({"cell": rows})
    assert [c["ref"] for c in pooled["T1"]] == ["B/2", "A/1"]
    assert pooled["T2"] == []


def test_pool_candidates_carries_hit_payload_evidence():
    hit = SearchHit(id="x", score=0.71234, payload={
        "references": ["A/1"], "speaker": "Montero", "lang": "es", "text": "snippet"})
    pooled = scoring.pool_candidates({"cell": [_row("T1", [hit])]})
    candidate = pooled["T1"][0]
    assert candidate["speaker"] == "Montero"
    assert candidate["lang"] == "es"
    assert candidate["text"] == "snippet"
    assert candidate["score"] == 0.7123


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


# --- Relevance-floor sweep and off-domain suppression -------------------------


def _floor_row(query_id, hits, expected=(), dimension="topical"):
    return {
        "id": query_id,
        "dimension": dimension,
        "hits": hits,
        "rank": scoring.first_rank(hits, list(expected)),
        "ranked_refs": scoring.distinct_refs(hits),
        "expected_refs": list(expected),
    }


def test_apply_floor_drops_hits_and_recomputes_rank():
    # Expected ref sits at rank 2 behind a junk hit; the floor removes the junk
    # hit (0.1) so the ref climbs to rank 1 — and a sub-floor expected hit at
    # the tail (0.05) drops out of ranked_refs entirely.
    hits = [_hit("X/9", score=0.1), _hit("A/1", score=0.9), _hit("B/2", score=0.05)]
    rows = [_floor_row("T1", hits, expected=["A/1", "B/2"])]

    floored = scoring.apply_floor(rows, 0.15)

    assert [h.score for h in floored[0]["hits"]] == [0.9]
    assert floored[0]["rank"] == 1
    assert floored[0]["ranked_refs"] == ["A/1"]
    assert floored[0]["score"] == 0.9


def test_apply_floor_zero_is_identity():
    rows = [_floor_row("T1", [_hit("A/1", score=0.05)], expected=["A/1"])]
    assert scoring.apply_floor(rows, 0)[0]["rank"] == 1
    assert scoring.apply_floor(rows, 0.0)[0]["hits"] == rows[0]["hits"]


def test_aggregate_excludes_offdomain_probes():
    rows = [
        {"dimension": "topical", "rank": 1, "ranked_refs": ["A/1"], "expected_refs": ["A/1"]},
        {"dimension": "offdomain", "rank": None, "ranked_refs": ["X/9"], "expected_refs": []},
    ]
    agg = scoring.aggregate(rows, k=5)
    assert "offdomain" not in agg
    assert agg["overall"]["n"] == 1              # junk row doesn't drag the averages
    assert agg["overall"]["mrr"] == 1.0


def test_suppression_counts_leaks_and_margin():
    rows = [
        _floor_row("J1", [], dimension="offdomain"),
        _floor_row("J2", [_hit("X/9", score=0.04), _hit("Y/8", score=0.12)],
                   dimension="offdomain"),
        _floor_row("T1", [_hit("A/1", score=0.9)], expected=["A/1"]),  # not junk
    ]
    stats = scoring.suppression(rows)
    assert (stats["n"], stats["suppressed"], stats["rate"]) == (2, 1, 0.5)
    assert stats["max_leak"] == 0.12


def test_suppression_after_floor_reports_full_suppression():
    rows = [_floor_row("J1", [_hit("X/9", score=0.04)], dimension="offdomain")]
    stats = scoring.suppression(scoring.apply_floor(rows, 0.15))
    assert (stats["suppressed"], stats["max_leak"]) == (1, None)


def test_suppression_none_without_offdomain_rows():
    assert scoring.suppression([_floor_row("T1", [_hit("A/1")], expected=["A/1"])]) is None


def test_pool_candidates_skips_offdomain_probes():
    rows = [
        _row("T1", [_hit("A/1")]),
        {**_row("J1", [_hit("X/9")]), "dimension": "offdomain"},
    ]
    pooled = scoring.pool_candidates({"cell": rows})
    assert [c["ref"] for c in pooled["T1"]] == ["A/1"]   # unjudged query still pools
    assert "J1" not in pooled                             # junk never does
