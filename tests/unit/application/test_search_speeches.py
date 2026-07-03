"""Unit test for the SearchSpeeches service — embedder and vector store stubbed."""

import pytest

from qhld_engine.application.search.search_speeches import SearchSpeeches
from qhld_engine.domain.ports.vector_store import SearchHit, SpeechGroup
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


class _GroupStore:
    def __init__(self):
        self.called = None

    def search_grouped(self, name, vector, group_by, limit, group_size,
                       filters=None, exclude=None):
        self.called = dict(
            name=name, group_by=group_by, limit=limit, group_size=group_size,
            filters=filters, exclude=exclude)
        return [SpeechGroup(
            speech_id="A", score=0.9,
            highlights=[SearchHit(id="p1", score=0.9, payload={"speech_id": "A"})])]


def test_search_grouped_passes_params_and_drops_none_filters():
    store = _GroupStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    groups = service.search_grouped(
        "q", page_size=5, highlights=2,
        filters={"lang": "gl", "group": None}, exclude={"X"})

    call = store.called
    assert call["name"] == "speeches__ollama__qwen3_embedding_0_6b__3"
    assert call["group_by"] == "speech_id"
    assert call["limit"] == 5          # page_size → number of speeches
    assert call["group_size"] == 2     # highlights per speech
    assert call["filters"] == {"lang": "gl"}  # None dropped
    assert call["exclude"] == {"X"}
    assert groups[0].speech_id == "A"
    assert groups[0].highlights[0].id == "p1"


def test_search_grouped_no_filters_or_cursor_pass_none():
    store = _GroupStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    service.search_grouped("q")

    assert store.called["filters"] is None
    assert store.called["exclude"] is None


class _WideStore:
    """Returns a fixed pool regardless of k, recording the k it was asked for."""

    def __init__(self, pool):
        self.pool = pool
        self.k = None

    def search(self, name, vector, k, filters=None):
        self.k = k
        return self.pool


class _FakeReranker:
    def __init__(self):
        self.call = None

    def rerank(self, query, hits, k):
        self.call = (query, list(hits), k)
        # reverse the pool and rescore, then trim — a visible reordering
        rescored = [SearchHit(id=h.id, score=float(i), payload=h.payload)
                    for i, h in enumerate(reversed(hits))]
        return rescored[:k]


def test_search_overfetches_and_reranks_when_reranker_set():
    pool = [SearchHit(id=f"p{i}", score=1.0 - i / 10, payload={"text": f"t{i}"}) for i in range(6)]
    store = _WideStore(pool)
    reranker = _FakeReranker()
    service = SearchSpeeches(
        settings=_settings(), embedder=_FakeEmbedder(), store=store, reranker=reranker)

    hits = service.search("q", k=3)

    assert store.k == 50                      # over-fetched to reranker_top_n, not k
    assert reranker.call[0] == "q" and reranker.call[2] == 3
    assert [h.id for h in hits] == ["p5", "p4", "p3"]  # reranker's reversed top-3


def test_reranker_none_by_default_keeps_baseline():
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=_FakeStore())
    assert service.reranker is None           # noop provider => no reranking


def test_search_grouped_reranks_highlights_and_resorts():
    hi_a = [SearchHit(id="a1", score=0.5, payload={"text": "x"}),
            SearchHit(id="a2", score=0.4, payload={"text": "y"})]
    hi_b = [SearchHit(id="b1", score=0.9, payload={"text": "z"})]

    class _Store:
        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            # A ranked first by the bi-encoder, B second
            return [SpeechGroup(speech_id="A", score=0.5, highlights=hi_a),
                    SpeechGroup(speech_id="B", score=0.9, highlights=hi_b)]

    class _Promote:
        # Gives group B a higher reranked score so it should overtake A
        def rerank(self, query, hits, k):
            score = 9.0 if hits[0].id == "b1" else 1.0
            return [SearchHit(id=hits[0].id, score=score, payload=hits[0].payload)]

    service = SearchSpeeches(
        settings=_settings(), embedder=_FakeEmbedder(), store=_Store(), reranker=_Promote())

    groups = service.search_grouped("q", page_size=2, highlights=1)

    assert [g.speech_id for g in groups] == ["B", "A"]  # rerank promoted B
    assert groups[0].score == 9.0                        # group score = best highlight
