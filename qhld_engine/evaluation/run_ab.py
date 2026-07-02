"""A/B benchmark runner for semantic search over speeches.

Runs the frozen query set through ``SearchSpeeches`` for each given embedding
model and reports rank / MRR / hit@k per dimension. Parametrizable over models so
new embedders slot in with a CLI arg — the harness is reused after every retrieval
lever (rerank, hybrid).

Run in-container (the repo is volume-mounted), where ``Settings`` already reaches
the Qdrant service + ollama — the same path ``index``/``search`` use:

    docker exec qhld-engine python -m qhld_engine.evaluation.run_ab \\
        --models qwen3-embedding:0.6b qwen3-embedding:4b

Scores are passage-level (the clean embedder signal). Crosslingual entries run
twice — with the ``--lang`` filter (forces the original-language block) and without
— to expose the cross-language penalty. Absolute cosine is not comparable across
models; compare MRR / hit@k.
"""

import argparse
import json
import os

from qhld_engine.evaluation import scoring

DEFAULT_QUERYSET = os.path.join(os.path.dirname(__file__), "queryset.json")


def _load_queryset(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _service_for(model):
    from qhld_engine.application.search.search_speeches import SearchSpeeches
    from qhld_engine.infrastructure.config.settings import get_settings

    settings = get_settings().model_copy(
        update={"embedding_provider": "ollama", "embedding_model": model})
    return SearchSpeeches(settings=settings)


def _score_at(hits, rank):
    return round(hits[rank - 1].score, 4) if rank else None


def _run_entry(service, entry, k):
    base_filters = dict(entry.get("filters") or {})
    filters = {**base_filters, "lang": entry["lang"]} if entry.get("lang") else base_filters
    hits = service.search(entry["query"], k=k, filters=filters or None)
    rank = scoring.first_rank(hits, entry["expected_refs"])
    row = {**entry, "rank": rank, "score": _score_at(hits, rank)}
    if entry.get("lang"):
        # Cross-language penalty baseline: same query, no lang filter.
        unfiltered = service.search(entry["query"], k=k, filters=base_filters or None)
        nolang_rank = scoring.first_rank(unfiltered, entry["expected_refs"])
        row["nolang_rank"] = nolang_rank
        row["nolang_score"] = _score_at(unfiltered, nolang_rank)
    return row, hits


def evaluate(model, queryset, k, verbose=False):
    service = _service_for(model)
    rows = []
    for entry in queryset:
        row, hits = _run_entry(service, entry, k)
        rows.append(row)
        if verbose:
            _dump(entry, hits)
    return rows


def _dump(entry, hits):
    print(f"  · {entry['id']} {entry['query']!r}")
    for position, hit in enumerate(hits, start=1):
        payload = hit.payload
        print(f"      {position:>2}. [{hit.score:.3f}] {payload.get('reference')} "
              f"· {payload.get('lang')} · {payload.get('speaker')}")


def _print_report(model, rows, k, hit_at):
    print(f"\n=== {model} ===")
    header = f"{'id':<4}{'dimension':<13}{'rank':>5}{'hit@'+str(hit_at):>7}{'score':>8}  penalty"
    print(header)
    for row in rows:
        rank = row["rank"] if row["rank"] is not None else "-"
        hitk = "Y" if scoring.hit_at_k(row["rank"], hit_at) else "n"
        score = row["score"] if row["score"] is not None else "-"
        penalty = ""
        if row.get("lang"):
            penalty = f"nolang_rank={row.get('nolang_rank')} nolang_score={row.get('nolang_score')}"
        print(f"{row['id']:<4}{row['dimension']:<13}{str(rank):>5}{hitk:>7}{str(score):>8}  {penalty}")

    print("  aggregates (MRR / hit@%d):" % hit_at)
    for dimension, metrics in scoring.aggregate(rows, k=hit_at).items():
        print(f"    {dimension:<13} n={metrics['n']:<3} "
              f"MRR={metrics['mrr']:<8} hit@{hit_at}={metrics[f'hit_at_{hit_at}']}")


def main():
    parser = argparse.ArgumentParser(description="Semantic-search A/B benchmark.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Embedding models to benchmark (ollama tags).")
    parser.add_argument("--k", type=int, default=10, help="Retrieval depth per query.")
    parser.add_argument("--hit-at", type=int, default=5, help="hit@k metric threshold.")
    parser.add_argument("--queryset", default=DEFAULT_QUERYSET)
    parser.add_argument("--verbose", action="store_true", help="Dump top-k per query.")
    args = parser.parse_args()

    queryset = _load_queryset(args.queryset)
    print(f"Query set: {len(queryset)} queries · retrieval k={args.k} · metric hit@{args.hit_at}")
    for model in args.models:
        rows = evaluate(model, queryset, args.k, args.verbose)
        _print_report(model, rows, args.k, args.hit_at)


if __name__ == "__main__":
    main()
