"""Unit test for the IndexSpeeches service — embedder, vector store and the
Speeches repo are stubbed, so this exercises chunk→point mapping, idempotent
delete-before-upsert, per-model collection naming and dim probing without any
network, Qdrant or Mongo.
"""

from uuid import NAMESPACE_DNS, uuid5

import pytest

from qhld_engine.application.speeches import index_speeches as mod
from qhld_engine.application.speeches.index_speeches import IndexSpeeches
from qhld_engine.infrastructure.config.settings import Settings

from tipi_data.models.speech import Speech, SpeechText

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
    def __init__(self):
        self.calls = []
        self.upserts = []

    def ensure_collection(self, name, dim):
        self.calls.append(("ensure", name, dim))

    def delete_by(self, name, key, value):
        self.calls.append(("delete", name, key, value))

    def upsert(self, name, points):
        self.calls.append(("upsert", name, points))
        self.upserts.append((name, points))


def _bilingual_speech():
    return Speech(
        id="sid1",
        reference="172/000001",
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
    assert by_lang["gl"].payload["reference"] == "172/000001"
    assert by_lang["gl"].payload["group"] == "GMx"
    # snippet text is stored for display
    assert by_lang["gl"].payload["text"] == "Boas tardes a todas e todos."

    # deterministic point ids
    assert by_lang["gl"].id == str(uuid5(_NS, "sid1:0:0"))
    assert by_lang["es"].id == str(uuid5(_NS, "sid1:1:0"))


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
