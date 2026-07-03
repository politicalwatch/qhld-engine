"""Offline tests for the CrossEncoderReranker — the encoder is stubbed so no
model (or torch) is loaded."""

import pytest

from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.infrastructure.reranker.cross_encoder import CrossEncoderReranker

pytestmark = pytest.mark.unit


class _FakeEncoder:
    """Scores each (query, passage) pair by the trailing int in the passage text,
    so the reranker's ordering is deterministic and independent of input order."""

    def predict(self, pairs):
        return [float(passage.split("t")[-1]) for _query, passage in pairs]


def _hit(id_, score, text):
    return SearchHit(id=id_, score=score, payload={"text": text})


def test_rerank_reorders_and_rescores_by_cross_encoder():
    reranker = CrossEncoderReranker("dummy-model")
    reranker._encoder = _FakeEncoder()
    hits = [_hit("a", 0.9, "t1"), _hit("b", 0.8, "t3"), _hit("c", 0.7, "t2")]

    out = reranker.rerank("q", hits, k=2)

    assert [h.id for h in out] == ["b", "c"]   # cross-encoder scores 3 > 2 > 1
    assert out[0].score == 3.0                 # score replaced by cross-encoder score
    assert out[0].payload["text"] == "t3"      # payload preserved


def test_rerank_empty_hits_returns_empty():
    reranker = CrossEncoderReranker("dummy-model")
    reranker._encoder = _FakeEncoder()
    assert reranker.rerank("q", [], k=5) == []
