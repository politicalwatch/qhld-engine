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
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)


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
    def ensure_collection(self, name: str, dim: int) -> None:
        """Create the collection with vector size ``dim`` if it does not exist."""
        ...

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        """Insert or overwrite ``points`` (by id) in the collection."""
        ...

    def delete_by(self, name: str, key: str, value) -> None:
        """Delete every point whose payload ``key`` equals ``value``."""
        ...

    def distinct_values(self, name: str, key: str) -> set:
        """The set of distinct values the payload ``key`` takes across the
        collection — used to skip already-indexed items on an incremental run."""
        ...

    def search(
        self, name: str, vector: list[float], k: int, filters: dict | None = None
    ) -> list[SearchHit]:
        """Return the ``k`` nearest points, optionally filtered by exact payload
        matches given as ``{key: value}``."""
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
    ) -> list["SpeechGroup"]:
        """Return the ``limit`` best groups (by payload ``group_by``), each with up
        to ``group_size`` passages as highlights. ``filters`` are exact payload
        matches; ``exclude`` is a set of ``group_by`` values to omit — used as a
        stateless pagination cursor ("load more" = re-query excluding seen ids)."""
        ...
