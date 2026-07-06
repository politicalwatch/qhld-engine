"""Cross-encoder reranker via sentence-transformers.

Scores each ``(query, passage)`` pair jointly and reorders — sharper than the
bi-encoder cone. Model is a settings/env value so it can be swapped via config;
default is ``BAAI/bge-reranker-v2-m3`` (Apache-2.0, XLM-R backbone with explicit
eu/ca/gl/es coverage). The ``CrossEncoder`` (and torch) is loaded lazily on first
use, so importing this module stays cheap for callers that don't rerank.
"""

from qhld_engine.domain.ports.reranker import RerankerPort
from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


class CrossEncoderReranker(RerankerPort):
    def __init__(self, model: str, top_n: int = 50):
        self._model = model
        self._top_n = top_n
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self._model)
        return self._encoder

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.payload.get("text") or "") for hit in hits]
        scores = self.encoder.predict(pairs)
        rescored = [
            SearchHit(id=hit.id, score=float(score), payload=hit.payload)
            for hit, score in zip(hits, scores)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("cross_encoder")
def create(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.reranker_model, settings.reranker_top_n)
