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
    SpeechGroup,
    VectorPoint,
    VectorStorePort,
)
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.logger import get_logger
from .factory import _register

log = get_logger(__name__)


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

    def ensure_collection(self, name: str, dim: int) -> None:
        def _ensure():
            if not self.client.collection_exists(name):
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
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        ))

    def delete_by(self, name: str, key: str, value) -> None:
        self._retry(lambda: self.client.delete(
            collection_name=name,
            points_selector=models.Filter(must=[self._condition(key, value)]),
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
    ) -> list[SpeechGroup]:
        must = [self._condition(key, value) for key, value in (filters or {}).items()]
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
        response = self._retry(lambda: self.client.query_points_groups(
            collection_name=name,
            group_by=group_by,
            query=vector,
            limit=limit,
            group_size=group_size,
            query_filter=query_filter,
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
        self, name: str, vector: list[float], k: int, filters: dict | None = None
    ) -> list[SearchHit]:
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[self._condition(key, value) for key, value in filters.items()])
        response = self._retry(lambda: self.client.query_points(
            collection_name=name,
            query=vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        ))
        return [
            SearchHit(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    @staticmethod
    def _condition(key: str, value) -> models.FieldCondition:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))


@_register("qdrant")
def create(settings: Settings) -> QdrantAdapter:
    return QdrantAdapter(settings)
