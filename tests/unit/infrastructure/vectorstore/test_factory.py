"""Unit tests for the vector-store registry + factory — Qdrant runs in-memory."""

import pytest

import qhld_engine.infrastructure.vectorstore.factory as factory_module
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_engine.infrastructure.vectorstore.qdrant import QdrantAdapter

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_qdrant_provider_registered():
    assert "qdrant" in factory_module._PROVIDERS


def test_create_builds_qdrant_adapter():
    store = create_vector_store_from_env(_settings(qdrant_host=":memory:"))
    assert isinstance(store, QdrantAdapter)
