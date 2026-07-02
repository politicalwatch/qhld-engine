"""Unit tests for the Qdrant adapter against an in-process store — no Docker."""

from uuid import uuid4

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from qhld_engine.domain.ports.vector_store import VectorPoint
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.vectorstore import qdrant as qdrant_mod
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


def test_distinct_values_returns_unique_payload_values(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "a", "lang": "gl"}),  # same speech, second block
        _point({"speech_id": "b", "lang": "es"}),
    ])
    assert adapter.distinct_values("c", "speech_id") == {"a", "b"}


def test_distinct_values_empty_collection(adapter):
    adapter.ensure_collection("c", 3)
    assert adapter.distinct_values("c", "speech_id") == set()


def test_search_grouped_returns_speeches_with_capped_highlights(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "B", "lang": "es"}),
    ])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=2)

    by_id = {g.speech_id: g for g in groups}
    assert set(by_id) == {"A", "B"}
    assert len(by_id["A"].highlights) == 2      # 3 passages, capped at group_size
    assert len(by_id["B"].highlights) == 1
    assert by_id["A"].score == by_id["A"].highlights[0].score


def test_search_grouped_limit_gives_stable_speech_count(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": s}) for s in ["A", "B", "C"]])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=3)
    assert len(groups) == 2  # number of speeches == limit, regardless of passages


def test_search_grouped_exclude_paginates(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": s}) for s in ["A", "B", "C"]])
    first = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=1)
    seen = {g.speech_id for g in first}
    nxt = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=1, exclude=seen)
    assert {g.speech_id for g in nxt}.isdisjoint(seen)  # load-more returns new speeches


def test_search_grouped_applies_exact_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "B", "lang": "gl"}),
    ])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=3,
        filters={"lang": "gl"})
    assert [g.speech_id for g in groups] == ["B"]


def test_retry_recovers_after_transient_disconnect(adapter, monkeypatch):
    monkeypatch.setattr(qdrant_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResponseHandlingException(Exception("Server disconnected"))
        return "ok"

    assert adapter._retry(flaky) == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_retry_raises_after_max_attempts(adapter, monkeypatch):
    monkeypatch.setattr(qdrant_mod.time, "sleep", lambda *_: None)

    def always_down():
        raise ResponseHandlingException(Exception("down"))

    with pytest.raises(ResponseHandlingException):
        adapter._retry(always_down)
