"""Tests for the per-model collection naming (pure string logic)."""

import pytest

from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.vectorstore.naming import collection_name

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="bge-m3:567m",
        **overrides,
    )


def test_derives_per_model_name():
    assert collection_name(_settings(), 1024) == "speeches__ollama__bge_m3_567m__1024"


def test_sparse_provider_adds_suffix():
    assert (
        collection_name(_settings(sparse_provider="bm25"), 1024)
        == "speeches__ollama__bge_m3_567m__1024__bm25"
    )


def test_sparse_none_keeps_dense_name():
    assert (
        collection_name(_settings(sparse_provider="none"), 1024)
        == "speeches__ollama__bge_m3_567m__1024"
    )


def test_explicit_collection_overrides_everything():
    settings = _settings(qdrant_collection="fixed", sparse_provider="bm25")
    assert collection_name(settings, 1024) == "fixed"
