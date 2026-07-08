"""Offline tests for the sparse-embedder factory registry."""

import pytest

from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.sparse.bm25 import Bm25SparseEmbedder
from qhld_engine.infrastructure.sparse.factory import create_sparse_embedder_from_env

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_bm25_provider_builds_bm25_embedder():
    settings = _settings(
        sparse_provider="bm25",
        sparse_model="Qdrant/bm25",
        sparse_language="spanish",
    )
    embedder = create_sparse_embedder_from_env(settings)
    assert isinstance(embedder, Bm25SparseEmbedder)
    assert embedder._model_name == "Qdrant/bm25"
    assert embedder._language == "spanish"
    assert embedder._model is None     # constructing it does not load the model


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown sparse provider"):
        create_sparse_embedder_from_env(_settings(sparse_provider="bogus"))
