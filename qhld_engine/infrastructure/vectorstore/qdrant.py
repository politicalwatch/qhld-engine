"""Qdrant implementation of ``VectorStorePort`` over the raw ``qdrant-client``.

We use the low-level client (not ``langchain-qdrant``) to keep control of
deterministic point ids and the payload shape. ``qdrant_host == ":memory:"``
selects qdrant-client's in-process mode, which lets the tests run with no Docker.
"""

from qdrant_client import QdrantClient, models

from qhld_engine.domain.ports.vector_store import SearchHit, VectorPoint, VectorStorePort
from qhld_engine.infrastructure.config.settings import Settings
from .factory import _register


class QdrantAdapter(VectorStorePort):
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

    def ensure_collection(self, name: str, dim: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )

    def delete_by(self, name: str, key: str, value) -> None:
        self.client.delete(
            collection_name=name,
            points_selector=models.Filter(must=[self._condition(key, value)]),
        )

    def search(
        self, name: str, vector: list[float], k: int, filters: dict | None = None
    ) -> list[SearchHit]:
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[self._condition(key, value) for key, value in filters.items()])
        response = self.client.query_points(
            collection_name=name,
            query=vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
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
