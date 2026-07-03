"""No-op reranker — returns the bi-encoder order unchanged (top-k).

Placeholder implementation of ``RerankerPort`` so the rerank plug-point in
``SearchSpeeches`` can be wired and tested before the real multilingual
cross-encoder adapter (and its model) are chosen.
"""

from qhld_engine.domain.ports.reranker import RerankerPort
from qhld_engine.domain.ports.vector_store import SearchHit
from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


class NoOpReranker(RerankerPort):
    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        return hits[:k]


@_register("noop")
def create(settings: Settings) -> NoOpReranker:
    return NoOpReranker()
