"""Unit test for the SearchSpeeches service — embedder and vector store stubbed."""

import pytest

from qhld_engine.application.search.search_speeches import SearchSpeeches
from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        **overrides,
    )


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]  # dim 3


class _FakeStore:
    def __init__(self):
        self.searched = None

    def search(self, name, vector, k, filters=None):
        self.searched = (name, vector, k, filters)
        return [SearchHit(id="p1", score=0.9, payload={"speaker": "X"})]


def test_search_embeds_query_and_uses_per_model_collection():
    store = _FakeStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    hits = service.search("financiación autonómica", k=5)

    name, vector, k, filters = store.searched
    assert name == "speeches__ollama__qwen3_embedding_0_6b__3"  # dim from query vector
    assert vector == [0.1, 0.2, 0.3]
    assert k == 5
    assert filters is None
    assert hits[0].id == "p1"


def test_none_filters_are_dropped():
    store = _FakeStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    service.search("hola", filters={"group": "GMx", "lang": None, "speaker": None})

    assert store.searched[3] == {"group": "GMx"}
