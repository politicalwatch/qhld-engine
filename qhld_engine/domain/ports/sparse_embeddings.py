"""Port for a sparse (lexical) embedder — the term-weight counterpart of the
dense embedder, used for hybrid retrieval.

Document- and query-side encodings differ on purpose (document weights carry
term-frequency saturation; query weights are plain term presence), and both
sides must tokenize identically, so index and search must use the same adapter
with the same configuration.
"""

from typing import Protocol

from qhld_engine.domain.ports.vector_store import SparseVector


class SparseEmbedderPort(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode passages for indexing (term weights with frequency saturation)."""
        ...

    def embed_query(self, text: str) -> SparseVector:
        """Encode a search query (plain term-presence weights)."""
        ...
