"""Unit test for the IndexSpeeches service — embedder, vector store and the
Speeches repo are stubbed, so this exercises chunk→point mapping, idempotent
delete-before-upsert, per-model collection naming and dim probing without any
network, Qdrant or Mongo.
"""

from types import SimpleNamespace
from uuid import NAMESPACE_DNS, uuid5

import pytest

from qhld_engine.application.speeches import index_speeches as mod
from qhld_engine.application.speeches.index_speeches import IndexSpeeches
from qhld_ai.domain.ports.vector_store import SparseVector
from qhld_ai.infrastructure.config.settings import Settings

from tipi_data.models.speech import Mention, Speech, SpeechText

pytestmark = pytest.mark.unit

_NS = uuid5(NAMESPACE_DNS, "speeches.qhld.politicalwatch.es")


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        **overrides,
    )


class _FakeEmbedder:
    """3-dim embedder: the doc vector encodes the text length so points are distinct."""

    def embed_query(self, text):
        return [0.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]


class _FakeStore:
    def __init__(self, indexed=()):
        self.calls = []
        self.upserts = []
        self.indexed = set(indexed)

    def ensure_collection(self, name, dim):
        self.calls.append(("ensure", name, dim))

    def delete_by(self, name, key, value):
        self.calls.append(("delete", name, key, value))

    def upsert(self, name, points):
        self.calls.append(("upsert", name, points))
        self.upserts.append((name, points))

    def distinct_values(self, name, key):
        self.calls.append(("distinct", name, key))
        return set(self.indexed)


@pytest.fixture(autouse=True)
def _deputy_catalog(monkeypatch):
    """The constituency join reads the deputy catalog; keep unit tests off Mongo."""
    monkeypatch.setattr(
        mod.Deputies, "get_all",
        lambda: [
            SimpleNamespace(name="Rego Candamil, Néstor", constituency="Coruña (A)"),
            SimpleNamespace(name="Sin Provincia, Ana", constituency=None),
        ],
        raising=False)


def _bilingual_speech():
    return Speech(
        id="sid1",
        references=["172/000001", "172/000005"],
        session_id="ses1",
        speaker="Rego Candamil, Néstor",
        speaker_surname="Rego",
        group="GMx",
        role="Diputado",
        order=1,
        legislature="15",
        date=20231213,
        session_name="Pleno",
        video_link="http://v/1.mp4",
        session_link="/s",
        speech=[
            SpeechText(lang="gl", text="Boas tardes a todas e todos.", original=True),
            SpeechText(lang="es", text="Buenas tardes a todas y todos.", original=False),
        ],
        original_language="gl",
        mentions=[
            Mention(person_id="d9", person_type="deputy",
                    name="Sánchez, Pedro", surface_forms=["Sánchez"], count=2),
            Mention(person_id="isabel-diaz-ayuso", person_type="regional_president",
                    name="Díaz Ayuso, Isabel", surface_forms=["Ayuso"], count=1),
            # unresolved (no person_id) → must not leak into the filterable payload
            Mention(person_id=None, name="Unknown", surface_forms=["Unknown"], count=1),
        ],
    )


def test_indexes_both_language_blocks_as_separate_points(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(
        mod.Speeches, "by_references", lambda refs: [_bilingual_speech()],
        raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute(["172/000001"])

    # dim probed from the embedder → per-model auto collection name
    assert service.dim == 3
    assert service.collection == "speeches__ollama__qwen3_embedding_0_6b__3"
    assert ("ensure", service.collection, 3) in store.calls

    # delete happens before upsert (idempotent re-index)
    kinds = [c[0] for c in store.calls if c[0] in ("delete", "upsert")]
    assert kinds == ["delete", "upsert"]
    assert ("delete", service.collection, "speech_id", "sid1") in store.calls

    _, points = store.upserts[0]
    assert len(points) == 2  # one point per language block (short text → 1 chunk each)

    by_lang = {p.payload["lang"]: p for p in points}
    assert by_lang["gl"].payload["original"] is True
    assert by_lang["gl"].payload["block_index"] == 0
    assert by_lang["es"].payload["original"] is False
    assert by_lang["es"].payload["block_index"] == 1
    # shared speech metadata rides on every point
    assert by_lang["gl"].payload["speaker_surname"] == "Rego"
    assert by_lang["gl"].payload["references"] == ["172/000001", "172/000005"]
    assert by_lang["gl"].payload["group"] == "GMx"
    # snippet text is stored for display
    assert by_lang["gl"].payload["text"] == "Boas tardes a todas e todos."
    # resolved mentions ride on every point — deputy AND non-deputy person ids, the
    # types map and the reserved counts; the unresolved mention (person_id=None) is excluded
    for point in points:
        assert point.payload["mentions"] == ["d9", "isabel-diaz-ayuso"]
        assert point.payload["mention_counts"] == {"d9": 2, "isabel-diaz-ayuso": 1}
        assert point.payload["mention_types"] == {
            "d9": "deputy", "isabel-diaz-ayuso": "regional_president"}
        # the speaker is a deputy → their province of election rides on every point
        assert point.payload["constituency"] == "Coruña (A)"

    # deterministic point ids
    assert by_lang["gl"].id == str(uuid5(_NS, "sid1:0:0"))
    assert by_lang["es"].id == str(uuid5(_NS, "sid1:1:0"))


def test_non_deputy_speaker_gets_no_constituency_key(monkeypatch):
    """A minister/witness (or a deputy without a recorded province) must not carry
    the key at all — absent, not null — so filters and facets never see them."""
    store = _FakeStore()
    speech = _bilingual_speech()
    speech.speaker = "García García, Ministra"
    monkeypatch.setattr(mod.Speeches, "by_references", lambda refs: [speech], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute(["172/000001"])

    _, points = store.upserts[0]
    assert all("constituency" not in p.payload for p in points)


def test_empty_speech_is_deleted_but_not_upserted(monkeypatch):
    store = _FakeStore()
    empty = _bilingual_speech()
    empty.speech = []
    monkeypatch.setattr(mod.Speeches, "all", lambda: [empty], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute()  # no references → all()

    assert ("delete", service.collection, "speech_id", "sid1") in store.calls
    assert store.upserts == []  # nothing to index


def test_collection_override_is_respected():
    store = _FakeStore()
    service = IndexSpeeches(
        settings=_settings(qdrant_collection="my_speeches"),
        embedder=_FakeEmbedder(),
        store=store,
    )
    assert service.collection == "my_speeches"


def _speech(sid):
    speech = _bilingual_speech()
    speech.id = sid
    return speech


def _upserted_speech_ids(store):
    return {p.payload["speech_id"] for _, points in store.upserts for p in points}


def test_incremental_default_skips_already_indexed(monkeypatch):
    store = _FakeStore(indexed={"sid1"})
    monkeypatch.setattr(
        mod.Speeches, "all", lambda: [_speech("sid1"), _speech("sid2")], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute()  # no references, incremental default

    # only the not-yet-indexed speech is embedded
    assert _upserted_speech_ids(store) == {"sid2"}
    # the already-indexed one is skipped entirely (not even deleted)
    assert ("delete", service.collection, "speech_id", "sid1") not in store.calls
    assert ("distinct", service.collection, "speech_id") in store.calls


def test_index_all_reindexes_everything(monkeypatch):
    store = _FakeStore(indexed={"sid1", "sid2"})
    monkeypatch.setattr(
        mod.Speeches, "all", lambda: [_speech("sid1"), _speech("sid2")], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute(incremental=False)

    # every speech is re-indexed despite already being present
    assert _upserted_speech_ids(store) == {"sid1", "sid2"}
    # the indexed set is not even consulted
    assert not any(c[0] == "distinct" for c in store.calls)


def test_targeted_reference_is_always_forced(monkeypatch):
    """`-r` re-indexes the named references even when they are already present."""
    store = _FakeStore(indexed={"sid1"})
    monkeypatch.setattr(
        mod.Speeches, "by_references", lambda refs: [_speech("sid1")], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute(["172/000001"])  # incremental default, but references given

    assert _upserted_speech_ids(store) == {"sid1"}
    assert not any(c[0] == "distinct" for c in store.calls)


# --- Hybrid (dense + sparse) indexing ----------------------------------------

class _FakeSparseEmbedder:
    """Encodes each text's length as its single term id, so vectors are distinct."""

    def embed_documents(self, texts):
        return [SparseVector(indices=[len(t)], values=[1.0]) for t in texts]


class _SparseAwareStore(_FakeStore):
    def ensure_collection(self, name, dim, sparse=False):
        self.calls.append(("ensure", name, dim, sparse))


def test_dense_only_default_leaves_points_without_sparse(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(
        mod.Speeches, "by_references", lambda refs: [_bilingual_speech()], raising=False)

    service = IndexSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)
    service.execute(["172/000001"])

    assert service.sparse_embedder is None
    _, points = store.upserts[0]
    assert all(p.sparse is None for p in points)


def test_hybrid_indexing_attaches_sparse_vectors(monkeypatch):
    store = _SparseAwareStore()
    monkeypatch.setattr(
        mod.Speeches, "by_references", lambda refs: [_bilingual_speech()], raising=False)

    service = IndexSpeeches(
        settings=_settings(sparse_provider="bm25"), embedder=_FakeEmbedder(),
        store=store, sparse_embedder=_FakeSparseEmbedder())
    service.execute(["172/000001"])

    # hybrid collections get their own suffixed name and a sparse-enabled schema
    assert service.collection == "speeches__ollama__qwen3_embedding_0_6b__3__bm25"
    assert ("ensure", service.collection, 3, True) in store.calls

    _, points = store.upserts[0]
    assert len(points) == 2
    for point in points:
        assert point.sparse == SparseVector(
            indices=[len(point.payload["text"])], values=[1.0])
