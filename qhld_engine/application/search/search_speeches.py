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
    def __init__(self, settings=None, embedder=None, store=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embedder_from_env(self.settings)
        self.store = store or create_vector_store_from_env(self.settings)

    def search(self, query, k=10, filters=None) -> list[SearchHit]:
        vector = self.embedder.embed_query(query)
        # The query vector's length is the model dimension, which is part of the
        # per-model collection name — no separate probe needed.
        collection = collection_name(self.settings, len(vector))
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        return self.store.search(collection, vector, k, clean or None)

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
        return self.store.search_grouped(
            collection,
            vector,
            group_by="speech_id",
            limit=page_size,
            group_size=highlights,
            filters=clean or None,
            exclude=exclude,
        )
