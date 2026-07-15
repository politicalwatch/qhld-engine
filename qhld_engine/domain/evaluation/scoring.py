"""Pure scoring metrics for the semantic-search A/B benchmark.

No I/O — operates on ``SearchHit`` lists (or the distinct references derived
from them) and the query set's expected references, so it is unit-testable
offline. Absolute cosine scores are not comparable across embedding models, so
we score by *rank*: MRR + hit@k (first relevant), plus set-aware recall@k and
average precision (MAP) now that relabeling gives each query several relevant
references.

Off-domain probes (dimension ``offdomain``, empty ``expected_refs``) invert the
question: nothing in the corpus is relevant, so every returned hit is a leak.
They are scored by ``suppression`` (and excluded from the rank metrics and the
adjudication pool), which only means anything on reranked scores — a relevance
floor is a cutoff on those, so ``apply_floor`` re-scores a finished run at any
floor value without re-retrieving.
"""

from collections import defaultdict

# Dimension marking the junk probes: queries with no relevant answer in the
# corpus, scored by suppression instead of rank.
OFFDOMAIN = "offdomain"


def distinct_refs(hits):
    """The ``references`` payloads of ``hits`` in rank order, de-duplicated
    (first occurrence wins). Passage-level results repeat a reference across its
    chunks, and a speech from an accumulated debate carries several references —
    each counts once, at the rank it first appears."""
    seen = []
    for hit in hits:
        for ref in hit.payload.get("references") or []:
            if ref not in seen:
                seen.append(ref)
    return seen


def first_rank(hits, expected_refs):
    """1-based rank of the first hit addressing any of ``expected_refs``
    (a hit's payload ``references`` lists every initiative of its debate);
    ``None`` if no top hit matches."""
    expected = set(expected_refs)
    for position, hit in enumerate(hits, start=1):
        if expected.intersection(hit.payload.get("references") or []):
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


def apply_floor(rows, floor):
    """Re-score benchmark ``rows`` as if hits scoring below ``floor`` had been
    dropped: ``hits`` filtered, ``rank``/``ranked_refs``/``score`` recomputed.
    Because the floor is nothing but a cutoff on the (reranked) hit scores, one
    retrieval+rerank run yields every floor value's metrics — the sweep is free.
    A falsy floor returns the rows unchanged (the raw run)."""
    if not floor:
        return list(rows)
    floored = []
    for row in rows:
        hits = [hit for hit in (row.get("hits") or []) if hit.score >= floor]
        rank = first_rank(hits, row["expected_refs"])
        floored.append({
            **row,
            "hits": hits,
            "rank": rank,
            "ranked_refs": distinct_refs(hits),
            "score": round(hits[rank - 1].score, 4) if rank else None,
        })
    return floored


def suppression(rows):
    """Junk-suppression stats over the ``offdomain`` probe rows: how many junk
    queries returned zero hits (``suppressed``/``rate``) and the highest score
    any junk hit reached (``max_leak`` — the margin a floor must clear). Returns
    ``None`` when the row set carries no offdomain probes."""
    junk = [row for row in rows if row.get("dimension") == OFFDOMAIN]
    if not junk:
        return None
    suppressed = sum(1 for row in junk if not row.get("hits"))
    leaks = [hit.score for row in junk for hit in (row.get("hits") or [])]
    return {
        "n": len(junk),
        "suppressed": suppressed,
        "rate": round(suppressed / len(junk), 4),
        "max_leak": round(max(leaks), 4) if leaks else None,
    }


def pool_candidates(rows_by_cell):
    """Merge per-query results from several benchmark cells into an
    adjudication pool: every retrieved reference not yet judged for its query
    (neither in ``expected_refs`` nor ``rejected_refs``) becomes a candidate,
    keeping the best-ranked hit across cells as reviewing evidence.

    ``rows_by_cell`` maps a cell label to its ``RunBenchmark`` rows. Returns
    ``{query_id: [candidate, ...]}`` ordered by rank, where a candidate carries
    ``ref``, ``cell``, ``rank``, ``score`` and the hit's ``speaker``/``lang``/
    ``text`` payload. Queries with nothing new map to an empty list.
    """
    pool = {}
    for cell, rows in rows_by_cell.items():
        for row in rows:
            if row.get("dimension") == OFFDOMAIN:
                # Junk probes have no relevant answers by construction — their
                # hits are leaks to suppress, never candidates to adjudicate.
                continue
            judged = set(row.get("expected_refs") or []) | set(row.get("rejected_refs") or [])
            candidates = pool.setdefault(row["id"], {})
            for position, hit in enumerate(row.get("hits") or [], start=1):
                for ref in hit.payload.get("references") or []:
                    if ref in judged:
                        continue
                    best = candidates.get(ref)
                    if best is None or position < best["rank"]:
                        candidates[ref] = {
                            "ref": ref,
                            "cell": cell,
                            "rank": position,
                            "score": round(hit.score, 4),
                            "speaker": hit.payload.get("speaker"),
                            "lang": hit.payload.get("lang"),
                            "text": hit.payload.get("text"),
                        }
    return {
        query_id: sorted(candidates.values(), key=lambda candidate: candidate["rank"])
        for query_id, candidates in pool.items()
    }


def aggregate(rows, k=5):
    """Aggregate per dimension (and overall). Each row is a dict carrying
    ``dimension``, ``rank``, ``ranked_refs`` and ``expected_refs``. Returns
    ``{dimension: {n, mrr, hit_at_<k>, recall_at_<k>, map}}``. Rows without
    ``expected_refs`` (the offdomain probes) are excluded — rank metrics are
    undefined with no relevant answers and would only drag the averages;
    those rows are scored by ``suppression`` instead."""
    scoreable = [row for row in rows if row.get("expected_refs")]
    by_dim = defaultdict(list)
    for row in scoreable:
        by_dim[row["dimension"]].append(row)
    by_dim["overall"] = scoreable

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
