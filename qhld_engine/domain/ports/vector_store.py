"""Port for a vector store (semantic search backend).

Kept minimal and backend-agnostic: the Qdrant adapter in
``infrastructure/vectorstore/`` implements it, but nothing in the domain or
application layers depends on qdrant-client. Points carry an opaque ``payload``
dict (speech metadata + snippet) so search can filter and render hits without a
round-trip to Mongo.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SparseVector:
    """A lexical (term-weight) vector: parallel lists of term ids and weights.
    Sparse counterpart of the dense embedding — used for hybrid retrieval, where
    exact-token matches (names, codes) complement the dense semantic ranking."""
    indices: list[int]
    values: list[float]


@dataclass
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)
    sparse: SparseVector | None = None


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class SpeechGroup:
    """A speech-level search result: the speech id, its best passage score, and
    its top matching passages as highlights (most-relevant first)."""
    speech_id: str
    score: float
    highlights: list[SearchHit] = field(default_factory=list)


class VectorStorePort(Protocol):
    def ensure_collection(self, name: str, dim: int, sparse: bool = False) -> None:
        """Create the collection with vector size ``dim`` if it does not exist.
        With ``sparse`` the collection also stores a lexical vector per point, so
        searches can fuse dense and lexical rankings (hybrid retrieval)."""
        ...

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        """Insert or overwrite ``points`` (by id) in the collection. A point's
        ``sparse`` vector is stored only in collections created with ``sparse``."""
        ...

    def delete_by(self, name: str, key: str, value) -> None:
        """Delete every point whose payload ``key`` equals ``value``."""
        ...

    def distinct_values(self, name: str, key: str) -> set:
        """The set of distinct values the payload ``key`` takes across the
        collection — used to skip already-indexed items on an incremental run."""
        ...

    def search(
        self,
        name: str,
        vector: list[float],
        k: int,
        filters: dict | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list[SearchHit]:
        """Return the ``k`` nearest points, optionally filtered by payload.

        ``filters`` is ``{key: value}``: a scalar value is an exact match; a dict
        value is a numeric range with ``gte``/``gt``/``lte``/``lt`` keys, e.g.
        ``{"date": {"gte": 20250403, "lte": 20250703}}`` (``date`` is a YYYYMMDD int).

        With ``sparse_vector`` (a lexical encoding of the same query) the store
        ranks by dense and lexical similarity separately and fuses the two
        rankings; scores are then fusion ranks, not cosine similarities."""
        ...

    def search_grouped(
        self,
        name: str,
        vector: list[float],
        group_by: str,
        limit: int,
        group_size: int,
        filters: dict | None = None,
        exclude: set | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list["SpeechGroup"]:
        """Return the ``limit`` best groups (by payload ``group_by``), each with up
        to ``group_size`` passages as highlights. ``filters`` follow the same
        scalar-or-range form as ``search``; ``exclude`` is a set of ``group_by``
        values to omit — a stateless pagination cursor ("load more" = re-query
        excluding seen ids); ``sparse_vector`` enables hybrid ranking as in
        ``search``."""
        ...
