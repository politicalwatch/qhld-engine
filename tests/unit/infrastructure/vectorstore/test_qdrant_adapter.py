"""Unit tests for the Qdrant adapter against an in-process store — no Docker."""

from uuid import uuid4

import pytest

from qhld_engine.domain.ports.vector_store import VectorPoint
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.vectorstore.qdrant import QdrantAdapter

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter():
    return QdrantAdapter(Settings(_env_file=None, qdrant_host=":memory:"))


def _point(payload):
    return VectorPoint(id=str(uuid4()), vector=[0.1, 0.2, 0.3], payload=payload)


def test_ensure_collection_is_idempotent(adapter):
    adapter.ensure_collection("c", 3)
    adapter.ensure_collection("c", 3)  # no error on second call
    assert adapter.client.collection_exists("c")


def test_upsert_and_search_round_trip(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5)
    assert {h.payload["speech_id"] for h in hits} == {"a", "b"}
    assert all(isinstance(h.score, float) for h in hits)


def test_search_applies_payload_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5, filters={"lang": "es"})
    assert [h.payload["speech_id"] for h in hits] == ["a"]


def test_delete_by_removes_matching_points(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    adapter.delete_by("c", "speech_id", "a")
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5)
    assert [h.payload["speech_id"] for h in hits] == ["b"]


def test_upsert_empty_is_a_noop(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [])  # must not raise
    assert adapter.search("c", [0.1, 0.2, 0.3], k=5) == []
