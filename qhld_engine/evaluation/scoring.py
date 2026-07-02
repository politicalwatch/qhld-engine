"""Pure scoring metrics for the semantic-search A/B benchmark.

No I/O — operates on ``SearchHit`` lists and the query set's expected references,
so it is unit-testable offline. Absolute cosine scores are not comparable across
embedding models, so we score by the *rank* of the expected result: MRR (mean
reciprocal rank) and hit@k.
"""

from collections import defaultdict


def first_rank(hits, expected_refs):
    """1-based rank of the first hit whose payload ``reference`` is in
    ``expected_refs``; ``None`` if no top hit matches."""
    expected = set(expected_refs)
    for position, hit in enumerate(hits, start=1):
        if hit.payload.get("reference") in expected:
            return position
    return None


def reciprocal_rank(rank):
    return 1.0 / rank if rank else 0.0


def hit_at_k(rank, k):
    return rank is not None and rank <= k


def aggregate(rows, k=5):
    """Aggregate per dimension (and overall). ``rows`` are dicts with ``dimension``
    and ``rank``. Returns ``{dimension: {n, mrr, hit_at_<k>}}``."""
    by_dim = defaultdict(list)
    for row in rows:
        by_dim[row["dimension"]].append(row["rank"])
    by_dim["overall"] = [row["rank"] for row in rows]

    result = {}
    for dimension, ranks in by_dim.items():
        n = len(ranks)
        mrr = sum(reciprocal_rank(r) for r in ranks) / n if n else 0.0
        hits = sum(1 for r in ranks if hit_at_k(r, k)) / n if n else 0.0
        result[dimension] = {
            "n": n,
            "mrr": round(mrr, 4),
            f"hit_at_{k}": round(hits, 4),
        }
    return result
