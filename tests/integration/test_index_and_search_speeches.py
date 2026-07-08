"""Integration test: index + search speeches through the REAL qhld-data Speeches
repository (throwaway Mongo) and a REAL in-memory Qdrant, stubbing only the
embedder (no Ollama/network).

This exercises the actual ``Speeches.all()`` / ``by_references()`` read methods
wired through ``IndexSpeeches``/``SearchSpeeches`` — the part the unit tests can
only stub. Guarded so it skips on a qhld-data pin that predates those methods
(i.e. before the qhld-data commit + engine repin lands).
"""

import pytest

from qhld_engine.domain.ports.vector_store import SparseVector
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.vectorstore.qdrant import QdrantAdapter

from tipi_data.models.speech import Speech, SpeechText
from tipi_data.repositories.speeches import Speeches

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not hasattr(Speeches, "all"),
        reason="qhld-data read methods (all/by_references) not yet installed/committed",
    ),
]


class _FakeEmbedder:
    """3-dim keyword embedder — no network. 'financ' and Galician cues get axes."""

    def _vec(self, text):
        text = text.lower()
        return [float("financ" in text), float("boas" in text or "gl" in text), 1.0]

    def embed_query(self, text):
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


def _settings():
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        qdrant_host=":memory:",
    )


def _bilingual_speech():
    return Speech(
        _id="x1", references=["172/000001"], session_id="s1", speaker="Rego",
        group="GMx", order=1, legislature="15", session_link="/s",
        speech=[
            SpeechText(lang="gl", text="Boas tardes, falamos de financiamento.", original=True),
            SpeechText(lang="es", text="Hablamos de financiación autonómica.", original=False),
        ],
        original_language="gl",
    )


def test_index_all_then_search_uses_real_speeches_repo(mongo_db):
    from qhld_engine.application.speeches.index_speeches import IndexSpeeches
    from qhld_engine.application.search.search_speeches import SearchSpeeches

    Speeches.save(_bilingual_speech())

    store = QdrantAdapter(_settings())  # shared in-memory store
    embedder = _FakeEmbedder()

    # execute() with no refs -> real Speeches.all()
    IndexSpeeches(settings=_settings(), embedder=embedder, store=store).execute()

    search = SearchSpeeches(settings=_settings(), embedder=embedder, store=store)
    hits = search.search("financiación autonómica", k=5)
    # both language blocks were indexed as separate points
    assert {h.payload["lang"] for h in hits} == {"gl", "es"}
    assert all(h.payload["speech_id"] == "x1" for h in hits)

    # payload filter narrows to the Spanish block
    es_only = search.search("financiación autonómica", k=5, filters={"lang": "es"})
    assert [h.payload["lang"] for h in es_only] == ["es"]


class _BlindDenseEmbedder:
    """Maps every text to the same vector — a dense embedder that cannot tell
    documents apart, so only the lexical branch can rank them."""

    def embed_query(self, text):
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]


class _FakeSparseEmbedder:
    """Token-hash sparse embedder — no network. Same tokenization for docs and
    queries, so a literal token match works like BM25's."""

    @staticmethod
    def _terms(text):
        tokens = {t for t in text.lower().replace(",", " ").replace(".", " ").split() if t}
        return sorted(hash(t) % (2**31) for t in tokens)

    def embed_documents(self, texts):
        return [
            SparseVector(indices=self._terms(t), values=[1.0] * len(self._terms(t)))
            for t in texts
        ]

    def embed_query(self, text):
        terms = self._terms(text)
        return SparseVector(indices=terms, values=[1.0] * len(terms))


def test_hybrid_index_and_search_surfaces_literal_token(mongo_db):
    from qhld_engine.application.speeches.index_speeches import IndexSpeeches
    from qhld_engine.application.search.search_speeches import SearchSpeeches

    ap9 = _bilingual_speech()
    ap9.id = "x2"
    ap9.references = ["172/000002"]
    ap9.speech = [SpeechText(lang="es", text="La autovía AP-9 en Galicia.", original=True)]
    Speeches.save(_bilingual_speech())
    Speeches.save(ap9)

    settings = _settings().model_copy(update={"sparse_provider": "bm25"})
    store = QdrantAdapter(settings)
    embedder = _BlindDenseEmbedder()
    sparse = _FakeSparseEmbedder()

    IndexSpeeches(
        settings=settings, embedder=embedder, store=store, sparse_embedder=sparse,
    ).execute()

    search = SearchSpeeches(
        settings=settings, embedder=embedder, store=store, sparse_embedder=sparse)
    # The dense embedder sees every document as identical — only the lexical
    # branch matches the literal token, so hybrid fusion must surface it first.
    hits = search.search("ap-9", k=3)
    assert hits and hits[0].payload["speech_id"] == "x2"


def test_index_by_reference_uses_real_repo(mongo_db):
    from qhld_engine.application.speeches.index_speeches import IndexSpeeches
    from qhld_engine.application.search.search_speeches import SearchSpeeches

    Speeches.save(_bilingual_speech())

    store = QdrantAdapter(_settings())
    embedder = _FakeEmbedder()

    # execute([ref]) -> real Speeches.by_references()
    IndexSpeeches(settings=_settings(), embedder=embedder, store=store).execute(["172/000001"])

    hits = SearchSpeeches(settings=_settings(), embedder=embedder, store=store).search(
        "financiación", k=5)
    assert {h.payload["lang"] for h in hits} == {"gl", "es"}
