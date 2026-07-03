"""Port for a reranker (cross-encoder) that reorders candidate hits by relevance.

Backend-agnostic: an infrastructure adapter (a multilingual cross-encoder served
via sentence-transformers, or a rerank API) implements it. A bi-encoder vector
search returns a wide candidate pool; the reranker scores each ``(query, passage)``
pair jointly for a sharper final ordering. Ollama has no rerank endpoint, so this
is served separately from the embedder.
"""

from typing import Protocol

from qhld_engine.domain.ports.vector_store import SearchHit


class RerankerPort(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        """Return the ``k`` hits most relevant to ``query``, reordered. Each
        returned hit's ``score`` is replaced by the cross-encoder score, so
        downstream aggregation (e.g. dedup/grouping by max score) stays correct."""
        ...
