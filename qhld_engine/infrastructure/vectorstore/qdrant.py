"""Qdrant implementation of ``VectorStorePort`` over the raw ``qdrant-client``.

We use the low-level client (not ``langchain-qdrant``) to keep control of
deterministic point ids and the payload shape. ``qdrant_host == ":memory:"``
selects qdrant-client's in-process mode, which lets the tests run with no Docker.

Every client call goes through ``_retry``: a slow embedder can leave the HTTP
connection idle past Qdrant's keep-alive timeout, so the server closes it and the
next call lands on a dead socket (``ResponseHandlingException``). httpx discards
the dead connection, so retrying dials a fresh one — this keeps long full-corpus
index runs (especially with slower/larger embedding models) from dying mid-way.
"""

import time

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from qhld_engine.domain.ports.vector_store import (
    SearchHit,
    SparseVector,
    SpeechGroup,
    VectorPoint,
    VectorStorePort,
)
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.logger import get_logger
from .factory import _register

log = get_logger(__name__)

# Named vectors of a hybrid collection (dense-only collections keep the
# original unnamed vector, so they need no migration).
_DENSE = "dense"
_SPARSE = "sparse"


class QdrantAdapter(VectorStorePort):
    _MAX_ATTEMPTS = 4
    _BACKOFF_SECONDS = 0.5

    def __init__(self, settings: Settings):
        if settings.qdrant_host == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
        self._prefetch_limit = settings.hybrid_prefetch_limit
        self._fusion = models.Fusion(settings.hybrid_fusion.lower())

    def _retry(self, operation):
        """Run a Qdrant client call, retrying transient connection drops (stale
        keep-alive sockets closed by the server during long idle gaps)."""
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return operation()
            except ResponseHandlingException as exc:
                if attempt == self._MAX_ATTEMPTS:
                    raise
                log.warning(
                    f"Qdrant connection error (attempt {attempt}/{self._MAX_ATTEMPTS}), "
                    f"retrying: {exc}")
                time.sleep(self._BACKOFF_SECONDS * attempt)

    def ensure_collection(self, name: str, dim: int, sparse: bool = False) -> None:
        def _ensure():
            if self.client.collection_exists(name):
                return
            if sparse:
                # Hybrid collection: a named dense vector plus a named sparse
                # (lexical) vector. The IDF modifier makes Qdrant weight sparse
                # matches by term rarity server-side, so the client only sends
                # corpus-independent term weights.
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={
                        _DENSE: models.VectorParams(
                            size=dim, distance=models.Distance.COSINE),
                    },
                    sparse_vectors_config={
                        _SPARSE: models.SparseVectorParams(
                            modifier=models.Modifier.IDF),
                    },
                )
            else:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=dim, distance=models.Distance.COSINE),
                )
        self._retry(_ensure)

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        self._retry(lambda: self.client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=p.id, vector=self._vector(p), payload=p.payload)
                for p in points
            ],
        ))

    @staticmethod
    def _vector(point: VectorPoint):
        """A point with a sparse vector targets a hybrid collection's named
        vectors; without one, the original unnamed dense layout."""
        if point.sparse is None:
            return point.vector
        return {
            _DENSE: point.vector,
            _SPARSE: models.SparseVector(
                indices=point.sparse.indices, values=point.sparse.values),
        }

    def delete_by(self, name: str, key: str, value) -> None:
        self._retry(lambda: self.client.delete(
            collection_name=name,
            points_selector=models.Filter(must=self._conditions(key, value)),
        ))

    def distinct_values(self, name: str, key: str) -> set:
        values = set()
        offset = None
        while True:
            records, offset = self._retry(lambda: self.client.scroll(
                collection_name=name,
                with_payload=[key],
                with_vectors=False,
                limit=1000,
                offset=offset,
            ))
            for record in records:
                if record.payload and key in record.payload:
                    values.add(record.payload[key])
            if offset is None:
                break
        return values

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
    ) -> list[SpeechGroup]:
        must = self._build_conditions(filters)
        must_not = (
            [models.FieldCondition(key=group_by, match=models.MatchAny(any=list(exclude)))]
            if exclude
            else []
        )
        query_filter = (
            models.Filter(must=must or None, must_not=must_not or None)
            if (must or must_not)
            else None
        )
        if sparse_vector is None:
            response = self._retry(lambda: self.client.query_points_groups(
                collection_name=name,
                group_by=group_by,
                query=vector,
                limit=limit,
                group_size=group_size,
                query_filter=query_filter,
                with_payload=True,
            ))
        else:
            fetch = max(limit * group_size, self._prefetch_limit)
            response = self._retry(lambda: self.client.query_points_groups(
                collection_name=name,
                group_by=group_by,
                prefetch=self._hybrid_prefetch(vector, sparse_vector, fetch, query_filter),
                query=models.FusionQuery(fusion=self._fusion),
                limit=limit,
                group_size=group_size,
                with_payload=True,
            ))
        groups = []
        for group in response.groups:
            highlights = [
                SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
                for p in group.hits
            ]
            top_score = highlights[0].score if highlights else 0.0
            groups.append(
                SpeechGroup(
                    speech_id=str(group.id), score=top_score, highlights=highlights))
        return groups

    def search(
        self,
        name: str,
        vector: list[float],
        k: int,
        filters: dict | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list[SearchHit]:
        must = self._build_conditions(filters)
        query_filter = models.Filter(must=must) if must else None
        if sparse_vector is None:
            response = self._retry(lambda: self.client.query_points(
                collection_name=name,
                query=vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            ))
        else:
            response = self._retry(lambda: self.client.query_points(
                collection_name=name,
                prefetch=self._hybrid_prefetch(
                    vector, sparse_vector, max(k, self._prefetch_limit), query_filter),
                query=models.FusionQuery(fusion=self._fusion),
                limit=k,
                with_payload=True,
            ))
        return [
            SearchHit(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    def _hybrid_prefetch(
        self,
        vector: list[float],
        sparse_vector: SparseVector,
        fetch: int,
        query_filter: models.Filter | None,
    ) -> list[models.Prefetch]:
        """Dense and sparse candidate branches for a fusion query. The payload
        filter goes on each branch: a top-level filter is not applied to
        prefetched candidates under fusion, so it would be silently ignored."""
        return [
            models.Prefetch(
                query=vector, using=_DENSE, limit=fetch, filter=query_filter),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values),
                using=_SPARSE, limit=fetch, filter=query_filter),
        ]

    @classmethod
    def _build_conditions(cls, filters: dict | None) -> list[models.FieldCondition]:
        """Translate a ``{key: value}`` filter dict into Qdrant conditions. A scalar
        value is an exact ``MatchValue``; a list is a ``MatchAny`` (any-of); a dict
        is either ``{"all": [...]}`` — one condition per element, so a list payload
        must contain every one — or a numeric ``Range`` whose keys are
        ``gte``/``gt``/``lte``/``lt`` (used for the ``date`` YYYYMMDD int)."""
        return [
            condition
            for key, value in (filters or {}).items()
            for condition in cls._conditions(key, value)
        ]

    @staticmethod
    def _conditions(key: str, value) -> list[models.FieldCondition]:
        if isinstance(value, dict):
            if "all" in value:
                return [
                    models.FieldCondition(key=key, match=models.MatchValue(value=v))
                    for v in value["all"]
                ]
            allowed = {"gte", "gt", "lte", "lt"}
            bounds = {k: v for k, v in value.items() if k in allowed}
            return [models.FieldCondition(key=key, range=models.Range(**bounds))]
        if isinstance(value, (list, tuple, set)):
            return [models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))]
        return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]


@_register("qdrant")
def create(settings: Settings) -> QdrantAdapter:
    return QdrantAdapter(settings)
