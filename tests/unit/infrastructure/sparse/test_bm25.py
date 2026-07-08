"""Offline tests for the BM25 sparse embedder (fastembed is never loaded)."""

import numpy as np
import pytest

from qhld_engine.domain.ports.vector_store import SparseVector
from qhld_engine.infrastructure.sparse.bm25 import Bm25SparseEmbedder

pytestmark = pytest.mark.unit


class _FakeEmbedding:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class _FakeModel:
    def embed(self, texts):
        return [
            _FakeEmbedding(np.array([1, 7]), np.array([0.5, 1.5])) for _ in texts
        ]

    def query_embed(self, text):
        yield _FakeEmbedding(np.array([7]), np.array([1.0]))


def _embedder_with_fake_model():
    embedder = Bm25SparseEmbedder(model="Qdrant/bm25", language="spanish")
    embedder._model = _FakeModel()
    return embedder


def test_embed_documents_converts_to_plain_sparse_vectors():
    vectors = _embedder_with_fake_model().embed_documents(["uno", "dos"])
    assert vectors == [
        SparseVector(indices=[1, 7], values=[0.5, 1.5]),
        SparseVector(indices=[1, 7], values=[0.5, 1.5]),
    ]
    assert all(isinstance(i, int) for i in vectors[0].indices)
    assert all(isinstance(v, float) for v in vectors[0].values)


def test_embed_query_converts_single_vector():
    vector = _embedder_with_fake_model().embed_query("uno")
    assert vector == SparseVector(indices=[7], values=[1.0])


def test_model_load_is_lazy():
    embedder = Bm25SparseEmbedder(model="Qdrant/bm25", language="spanish")
    assert embedder._model is None
