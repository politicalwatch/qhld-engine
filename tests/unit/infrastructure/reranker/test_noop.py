"""Unit test for the NoOpReranker placeholder."""

import pytest

from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.infrastructure.reranker.noop import NoOpReranker

pytestmark = pytest.mark.unit


def test_noop_reranker_returns_topk_unchanged():
    hits = [
        SearchHit(id="a", score=0.9, payload={}),
        SearchHit(id="b", score=0.5, payload={}),
    ]
    assert NoOpReranker().rerank("q", hits, 1) == [hits[0]]


def test_noop_reranker_returns_all_when_k_exceeds_hits():
    hits = [SearchHit(id="a", score=0.9, payload={})]
    assert NoOpReranker().rerank("q", hits, 10) == hits
