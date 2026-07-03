"""Application service: natural-language semantic search over indexed speeches.

Embeds the query with the same configured embedder used for indexing, runs a vector
search in Qdrant (optionally filtered by exact payload matches — group, legislature,
lang, speaker…), and returns the ranked hits. Each hit's payload carries the speech
metadata and the passage snippet, so callers can render results without a Mongo
round-trip; ``Speeches.get`` is available for full-text hydration when needed.
"""

from qhld_engine.domain.ports.vector_store import SearchHit, SpeechGroup
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_engine.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_engine.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_engine.infrastructure.vectorstore.naming import collection_name


class SearchSpeeches:
    def __init__(self, settings=None, embedder=None, store=None, reranker=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embedder_from_env(self.settings)
        self.store = store or create_vector_store_from_env(self.settings)
        self.reranker = reranker if reranker is not None else self._reranker_from_settings()

    def _reranker_from_settings(self):
        """Build the configured reranker, or ``None`` for the "noop"/unset default
        so the bi-encoder baseline path stays byte-identical."""
        provider = (self.settings.reranker_provider or "").lower()
        if not provider or provider == "noop":
            return None
        from qhld_engine.infrastructure.reranker.factory import create_reranker_from_env

        return create_reranker_from_env(self.settings)

    def search(self, query, k=10, filters=None) -> list[SearchHit]:
        vector = self.embedder.embed_query(query)
        # The query vector's length is the model dimension, which is part of the
        # per-model collection name — no separate probe needed.
        collection = collection_name(self.settings, len(vector))
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        if self.reranker is None:
            return self.store.search(collection, vector, k, clean or None)
        # Over-fetch a wide candidate pool for the cross-encoder to reorder.
        fetch = max(k, self.settings.reranker_top_n)
        hits = self.store.search(collection, vector, fetch, clean or None)
        return self.reranker.rerank(query, hits, k)

    def search_grouped(
        self, query, page_size=10, highlights=3, filters=None, exclude=None
    ) -> list[SpeechGroup]:
        """Speech-level results: ``page_size`` distinct speeches, each with up to
        ``highlights`` matching passages. Pagination is stateless — the caller
        accumulates the returned ``speech_id``s and passes them back as ``exclude``
        to fetch the next page ("load more")."""
        vector = self.embedder.embed_query(query)
        collection = collection_name(self.settings, len(vector))
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        if self.reranker is None:
            return self.store.search_grouped(
                collection,
                vector,
                group_by="speech_id",
                limit=page_size,
                group_size=highlights,
                filters=clean or None,
                exclude=exclude,
            )
        # Over-fetch groups (and highlights per group) so the reranker can promote
        # a speech the bi-encoder ranked lower; rerank each group's highlights,
        # recompute the group score from the best reranked highlight, re-sort, trim.
        groups = self.store.search_grouped(
            collection,
            vector,
            group_by="speech_id",
            limit=page_size * 2,
            group_size=max(highlights, 5),
            filters=clean or None,
            exclude=exclude,
        )
        reranked = []
        for group in groups:
            top = self.reranker.rerank(query, group.highlights, highlights)
            score = top[0].score if top else group.score
            reranked.append(SpeechGroup(speech_id=group.speech_id, score=score, highlights=top))
        reranked.sort(key=lambda group: group.score, reverse=True)
        return reranked[:page_size]
