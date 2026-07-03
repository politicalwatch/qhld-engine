"""Pure scoring metrics for the semantic-search A/B benchmark.

No I/O — operates on ``SearchHit`` lists (or the distinct references derived
from them) and the query set's expected references, so it is unit-testable
offline. Absolute cosine scores are not comparable across embedding models, so
we score by *rank*: MRR + hit@k (first relevant), plus set-aware recall@k and
average precision (MAP) now that relabeling gives each query several relevant
references.
"""

from collections import defaultdict


def distinct_refs(hits):
    """The ``reference`` payloads of ``hits`` in rank order, de-duplicated
    (first occurrence wins). Passage-level results repeat a reference across its
    chunks; reference-level metrics must count each reference once."""
    seen = []
    for hit in hits:
        ref = hit.payload.get("reference")
        if ref is not None and ref not in seen:
            seen.append(ref)
    return seen


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


def recall_at_k(ranked_refs, expected_refs, k):
    """Fraction of the expected references retrieved within the top ``k``
    distinct references."""
    expected = set(expected_refs)
    if not expected:
        return 0.0
    found = set(ranked_refs[:k]) & expected
    return len(found) / len(expected)


def average_precision(ranked_refs, expected_refs):
    """Average precision over the distinct-reference ranking, normalised by the
    number of relevant references (standard AP; the mean over queries is MAP)."""
    expected = set(expected_refs)
    if not expected:
        return 0.0
    found = 0
    score = 0.0
    for position, ref in enumerate(ranked_refs, start=1):
        if ref in expected:
            found += 1
            score += found / position
    return score / len(expected)


def aggregate(rows, k=5):
    """Aggregate per dimension (and overall). Each row is a dict carrying
    ``dimension``, ``rank``, ``ranked_refs`` and ``expected_refs``. Returns
    ``{dimension: {n, mrr, hit_at_<k>, recall_at_<k>, map}}``."""
    by_dim = defaultdict(list)
    for row in rows:
        by_dim[row["dimension"]].append(row)
    by_dim["overall"] = list(rows)

    result = {}
    for dimension, dim_rows in by_dim.items():
        n = len(dim_rows)
        if not n:
            continue
        mrr = sum(reciprocal_rank(r["rank"]) for r in dim_rows) / n
        hits = sum(1 for r in dim_rows if hit_at_k(r["rank"], k)) / n
        recall = sum(
            recall_at_k(r["ranked_refs"], r["expected_refs"], k) for r in dim_rows
        ) / n
        mean_ap = sum(
            average_precision(r["ranked_refs"], r["expected_refs"]) for r in dim_rows
        ) / n
        result[dimension] = {
            "n": n,
            "mrr": round(mrr, 4),
            f"hit_at_{k}": round(hits, 4),
            f"recall_at_{k}": round(recall, 4),
            "map": round(mean_ap, 4),
        }
    return result
