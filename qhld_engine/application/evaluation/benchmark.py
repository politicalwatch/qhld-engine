"""A/B benchmark service for semantic search over speeches.

Driving-adapter orchestration: runs the frozen query set through the
``SearchSpeeches`` application service for each (embedding model x reranker)
cell and returns scored rows — no printing (the CLI owns presentation). The
grid is what makes it reusable as retrieval changes: pass more models or
rerankers and each combination becomes a column.

Scores are passage-level (the clean embedder signal). Crosslingual entries run
twice — with the ``--lang`` filter (forces the original-language block) and
without — to expose the cross-language penalty. Absolute cosine is not
comparable across models; compare MRR / hit@k / recall@k / MAP.
"""

import json
import os

from qhld_engine.domain.evaluation import scoring

DEFAULT_QUERYSET = os.path.join(os.path.dirname(__file__), "queryset.json")

# Reranker CLI tokens that mean "leave the bi-encoder order untouched".
_NO_RERANKER = {None, "", "none", "noop"}

# Sparse CLI tokens that mean "dense-only retrieval, no hybrid fusion".
_NO_SPARSE = {None, "", "none"}


def load_queryset(path=DEFAULT_QUERYSET):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _score_at(hits, rank):
    return round(hits[rank - 1].score, 4) if rank else None


class RunBenchmark:
    """Runs the query set for one (model, reranker, sparse) cell at a time."""

    def __init__(self, queryset_path=DEFAULT_QUERYSET):
        self.queryset = load_queryset(queryset_path)

    def run(self, model, reranker="none", sparse="none", k=10):
        """Return a scored row per query for ``model`` reranked by ``reranker``
        ("none"/"noop" => bi-encoder order untouched) with sparse provider
        ``sparse`` ("none" => dense-only; e.g. "bm25" => hybrid fusion against
        that model's hybrid collection)."""
        service = self._service(model, reranker, sparse)
        return [self._run_entry(service, entry, k) for entry in self.queryset]

    def _service(self, model, reranker, sparse="none"):
        from qhld_ai.application.search.search_speeches import SearchSpeeches
        from qhld_ai.infrastructure.config.settings import get_settings

        rerank_off = reranker in _NO_RERANKER
        settings = get_settings().model_copy(update={
            "embedding_provider": "ollama",
            "embedding_model": model,
            "reranker_provider": "noop" if rerank_off else "cross_encoder",
            "reranker_model": "" if rerank_off else reranker,
            "sparse_provider": "none" if sparse in _NO_SPARSE else sparse,
        })
        return SearchSpeeches(settings=settings)

    def _run_entry(self, service, entry, k):
        base_filters = dict(entry.get("filters") or {})
        filters = {**base_filters, "lang": entry["lang"]} if entry.get("lang") else base_filters
        hits = service.search(entry["query"], k=k, filters=filters or None)
        rank = scoring.first_rank(hits, entry["expected_refs"])
        row = {
            **entry,
            "rank": rank,
            "ranked_refs": scoring.distinct_refs(hits),
            "score": _score_at(hits, rank),
            "hits": hits,
        }
        if entry.get("lang"):
            # Cross-language penalty baseline: same query, no lang filter.
            unfiltered = service.search(entry["query"], k=k, filters=base_filters or None)
            row["nolang_rank"] = scoring.first_rank(unfiltered, entry["expected_refs"])
            row["nolang_score"] = _score_at(unfiltered, row["nolang_rank"])
        return row
