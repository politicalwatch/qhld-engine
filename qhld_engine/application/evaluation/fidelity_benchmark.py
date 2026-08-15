"""Retrieval-layer fidelity benchmark: approximate search against brute force.

Answers the question the labelled benchmark provably cannot. ``qhld eval
retrieval`` measures the end of the pipeline (MAP/MRR after fusion and the
reranker) on 45 human-labelled queries; run on a retrieval-layer change it came
back saturated at MRR 1.0, with n=4-6 per dimension and opposite signs across
cells. It cannot arbitrate a knob that changes what *enters* the pipeline,
because the reranker absorbs the difference downstream.

This one compares a cell against ``exact: true`` on the SAME collection. Brute
force is the ground truth, so no labels are needed, the query set is unbounded,
and the only difference between the two arms is the thing under test.

Two query arms, because neither is sufficient alone:

* ``corpus`` — vectors sampled from the collection itself. Deterministic, needs
  no embedder, and can be scaled to whatever n a question deserves. Its one
  distortion is that a corpus vector has its own point at distance 0, so it is
  always found: absolute recall reads slightly high.
* ``real`` — the 45 labelled queries plus the 49 parser queries, embedded once
  through the production embedder. User-shaped, and it is what catches a result
  that is an artifact of the arm above. Small (n=94) and it needs Ollama.

IMPORTANT, and the reason this file talks to Qdrant directly: controlling
``exact``/``rescore``/``oversampling`` per cell means issuing raw queries, so
this instrument does NOT exercise the production adapter's ``_search_params`` or
``_hybrid_prefetch``. That is precisely the gap in which a top-level ``params``
once became a silent no-op on the hybrid path — a harness issuing its own
requests could never have caught it. The adapter's spy-based unit tests remain
that check; this is not a substitute for them.
"""

import json
import os
import random

from qhld_engine.domain.evaluation import fidelity_scoring

DEFAULT_QUERYSET = os.path.join(os.path.dirname(__file__), "fidelity_queryset.json")
LABELLED_QUERYSET = os.path.join(os.path.dirname(__file__), "queryset.json")
PARSE_QUERYSET = os.path.join(os.path.dirname(__file__), "parse_queryset.json")

# Named dense vector of a hybrid collection. Dense-only collections keep the
# original unnamed vector, so they need no ``using``.
_DENSE = "dense"
_SPARSE = "sparse"


def load_asset(path=DEFAULT_QUERYSET):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class RunFidelityBenchmark:
    """Runs fidelity cells against one collection.

    ``collection`` skips embedder-based name resolution entirely, which is what
    lets the corpus arm run with no services but Qdrant.
    """

    def __init__(self, settings=None, collection=None, asset_path=DEFAULT_QUERYSET):
        from qhld_ai.infrastructure.config.settings import get_settings

        self.settings = settings or get_settings()
        self.asset = load_asset(asset_path)
        self._collection = collection
        self._client = None
        self._using = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                grpc_port=self.settings.qdrant_grpc_port,
                prefer_grpc=self.settings.qdrant_prefer_grpc,
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            from qhld_ai.infrastructure.embeddings.factory import (
                create_embedder_from_env,
            )
            from qhld_ai.infrastructure.vectorstore.naming import collection_name

            embedder = create_embedder_from_env(self.settings)
            dim = len(embedder.embed_query("probe"))
            self._collection = collection_name(self.settings, dim)
        return self._collection

    def using(self, collection=None):
        """``"dense"`` on a hybrid collection, ``None`` on a dense-only one.
        Read from the collection rather than assumed, so the float32 reference
        arm and the serving collection can both be addressed by name."""
        name = collection or self.collection
        config = self.client.get_collection(name).config.params.vectors
        return _DENSE if isinstance(config, dict) else None

    # --- query arms -------------------------------------------------------

    def sample_corpus_vectors(self, collection=None, size=None, seed=None):
        """Draw ``size`` point vectors at random from the WHOLE id space.

        Walking every id first costs one scroll pass and is the point: sampling
        from a single scroll page draws only the lowest ids, which are one end of
        an insertion-ordered corpus, not a sample of it.

        Returns ``(vectors, fingerprint)`` — see ``ids_fingerprint`` for why the
        sample is identified by a hash instead of being stored.
        """
        name = collection or self.collection
        size = size or self.asset["sample"]["size"]
        seed = seed if seed is not None else self.asset["sample"]["seed"]

        ids, offset = [], None
        while True:
            page, offset = self.client.scroll(
                collection_name=name, limit=10000, offset=offset,
                with_payload=False, with_vectors=False)
            ids.extend(point.id for point in page)
            if offset is None:
                break
        picked = random.Random(seed).sample(ids, min(size, len(ids)))
        records = self.client.retrieve(
            collection_name=name, ids=picked, with_vectors=True, with_payload=False)
        vectors = [self._dense_of(record.vector) for record in records]
        return vectors, fidelity_scoring.ids_fingerprint(picked)

    @staticmethod
    def _dense_of(vector):
        return vector[_DENSE] if isinstance(vector, dict) else vector

    def real_query_texts(self):
        """The labelled query set plus the parser query set: every real,
        human-written query the project owns. The parser set is included because
        it is the only place filter-bearing queries were ever written down — the
        labelled set has none at all."""
        texts = []
        for path in (LABELLED_QUERYSET, PARSE_QUERYSET):
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            # The two assets disagree in shape: the labelled set is a bare list,
            # the parser set wraps its rows under "queries" beside the "today"
            # the date slots are judged against.
            rows = loaded if isinstance(loaded, list) else loaded["queries"]
            texts.extend(row["query"] for row in rows if row.get("query"))
        return texts

    def real_query_vectors(self, cache_path=None):
        """Embed the real queries through the production embedder, caching to
        ``cache_path`` so a re-run neither depends on Ollama nor re-embeds. The
        cache is keyed by embedding model: reading one model's vectors against
        another model's collection would silently measure nonsense."""
        model = self.settings.embedding_model
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("model") == model:
                return cached["vectors"]
        from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env

        embedder = create_embedder_from_env(self.settings)
        vectors = [embedder.embed_query(text) for text in self.real_query_texts()]
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump({"model": model, "vectors": vectors}, handle)
        return vectors

    # --- searching --------------------------------------------------------

    def _search_params(self, cell):
        from qdrant_client import models

        if cell.get("exact"):
            return models.SearchParams(exact=True)
        quantization = None
        if cell.get("rescore") is not None:
            quantization = models.QuantizationSearchParams(
                rescore=cell["rescore"], oversampling=cell.get("oversampling"))
        return models.SearchParams(
            hnsw_ef=cell.get("hnsw_ef"), quantization=quantization)

    def dense_hits(self, vector, cell, k, collection=None, query_filter=None):
        response = self.client.query_points(
            collection_name=collection or self.collection,
            query=vector,
            using=self._using_for(collection),
            limit=k,
            query_filter=query_filter,
            search_params=self._search_params(cell),
            with_payload=False,
        )
        return [str(point.id) for point in response.points]

    def hybrid_hits(self, vector, sparse, cell, k, collection=None, query_filter=None):
        """Fused dense+sparse top-k, the shape the product actually serves.

        The filter and the search params ride on the prefetch branches, not the
        top level — under fusion a top-level one is ignored. Only the dense
        branch takes params: a sparse branch matches an inverted index exactly
        and has no approximation to tune, which is also why fusion DILUTES
        whatever the dense branch loses."""
        from qdrant_client import models

        name = collection or self.collection
        fetch = max(k, self.settings.hybrid_prefetch_limit)
        prefetch = [
            models.Prefetch(
                query=vector, using=_DENSE, limit=fetch, filter=query_filter,
                params=self._search_params(cell)),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=_SPARSE, limit=fetch, filter=query_filter),
        ]
        response = self.client.query_points(
            collection_name=name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion(self.settings.hybrid_fusion.lower())),
            limit=k,
            with_payload=False,
        )
        return [str(point.id) for point in response.points]

    def _using_for(self, collection=None):
        key = collection or self.collection
        if self._using is None:
            self._using = {}
        if key not in self._using:
            self._using[key] = self.using(key)
        return self._using[key]

    # --- the run ----------------------------------------------------------

    def run_cell(self, vectors, cell, k, collection=None, query_filter=None,
                 sparse_vectors=None):
        """Score one cell against exact search over the same vectors, filter and
        collection. Ground truth is recomputed per cell rather than cached once,
        because a filter or a collection change moves it — caching it across
        cells is how an instrument starts comparing arms to the wrong reference."""
        truth_cell = {"exact": True}
        recalls, misses = [], []
        for index, vector in enumerate(vectors):
            if sparse_vectors is None:
                approx = self.dense_hits(vector, cell, k, collection, query_filter)
                truth = self.dense_hits(vector, truth_cell, k, collection, query_filter)
            else:
                sparse = sparse_vectors[index]
                approx = self.hybrid_hits(
                    vector, sparse, cell, k, collection, query_filter)
                truth = self.hybrid_hits(
                    vector, sparse, truth_cell, k, collection, query_filter)
            recalls.append(fidelity_scoring.recall_at_k(approx, truth))
            misses.append(
                fidelity_scoring.reciprocal_rank_agreement(approx, truth)
                if truth else None)
        return {
            "cell": cell,
            "label": fidelity_scoring.cell_label(cell),
            "per_query": recalls,
            **fidelity_scoring.aggregate(recalls),
            **fidelity_scoring.summarise_misses(misses),
        }

    def filter_cells(self):
        """The payload filters to sweep, with the cardinality each was chosen to
        represent. Stratified by cardinality on purpose: a filter degrades an
        HNSW graph by disconnecting it, so the hypothesis is about how MUCH of
        the collection a filter leaves, not which key it names."""
        from qdrant_client import models

        cells = [{"label": "unfiltered", "filter": None}]
        for row in self.asset["filters"]:
            cells.append({
                "label": row["label"],
                "filter": models.Filter(must=[
                    models.FieldCondition(**self._condition(condition))
                    for condition in row["must"]
                ]),
                "expected_count": row.get("expected_count"),
            })
        return cells

    @staticmethod
    def _condition(condition):
        from qdrant_client import models

        if "range" in condition:
            return {
                "key": condition["key"],
                "range": models.Range(**condition["range"]),
            }
        return {
            "key": condition["key"],
            "match": models.MatchValue(value=condition["value"]),
        }

    def count(self, query_filter, collection=None):
        return self.client.count(
            collection_name=collection or self.collection,
            count_filter=query_filter, exact=True).count
