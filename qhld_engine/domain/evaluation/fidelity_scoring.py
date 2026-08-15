"""Pure scoring for the retrieval-layer fidelity instrument.

No I/O — operates on id lists, so it is unit-testable offline.

This measures something different from ``scoring``, and the difference is the
whole reason the module exists. ``scoring`` asks *is the answer relevant*, which
needs human labels, saturates on a 45-query set and is measured at the END of the
pipeline — after fusion and the reranker, both of which absorb whatever the
retrieval layer did. Fidelity asks *did the approximate search find what brute
force would*, which needs no labels at all: the ground truth is the same
collection searched exhaustively. That makes the query set unbounded, so a
retrieval-layer change can be measured with real statistical power instead of
n=4-6 per dimension.

The cost of that power is the thing to keep saying out loud: fidelity is not
relevance. A faithful search that returns the wrong answer is still wrong. What
fidelity does arbitrate is what ENTERS the pipeline — and a chunk that never
reaches the reranker can never be ranked by it, however good the reranker is.
"""

import math


def recall_at_k(approx_ids, truth_ids, k=None):
    """Fraction of the exhaustive search's top-k that the approximate search also
    found. ``1.0`` means the two agree as sets; order is deliberately ignored,
    because a reranker reorders the pool anyway and only membership decides what
    it gets to see.

    An empty ground truth (a filter matching nothing) returns ``None`` rather
    than 0.0 — there was nothing to find, so the query carries no evidence and
    must not be averaged in as a failure."""
    truth = set(truth_ids[:k] if k else truth_ids)
    if not truth:
        return None
    return len(truth & set(approx_ids[:k] if k else approx_ids)) / len(truth)


def aggregate(per_query):
    """Summarise one cell's per-query recalls: ``n``, ``mean``, ``perfect`` (how
    many queries agreed with brute force exactly) and ``worst``.

    ``perfect`` is reported beside the mean because the two answer different
    questions and can disagree sharply — a cell can sit at a comfortable 0.96
    mean while almost every individual query is missing something, which is what
    a candidate pool actually experiences. ``None`` entries (no ground truth) are
    dropped, not counted."""
    scored = [value for value in per_query if value is not None]
    if not scored:
        return {"n": 0, "mean": None, "perfect": 0, "worst": None}
    return {
        "n": len(scored),
        "mean": round(sum(scored) / len(scored), 4),
        "perfect": sum(1 for value in scored if value == 1.0),
        "worst": round(min(scored), 4),
    }


def paired_bootstrap(baseline, arm, resamples=20000, seed=2718):
    """Is ``arm`` really different from ``baseline``? Resamples the per-query
    differences to a 95 % interval, pairing the two arms query by query.

    Paired, because the arms answer the SAME queries: some queries are simply
    harder than others, and pairing removes that shared variance instead of
    letting it swamp the effect. Seeded, so a re-run is digit-identical — this
    instrument's claim to being citable rests on that.

    Queries where either arm has no ground truth are dropped from both.
    Returns ``{n, delta, lo, hi, significant}`` with the deltas in points
    (multiply by 100 for pp), or ``None`` when nothing is comparable."""
    import numpy as np

    pairs = [
        (base, other)
        for base, other in zip(baseline, arm)
        if base is not None and other is not None
    ]
    if not pairs:
        return None
    deltas = np.asarray([other - base for base, other in pairs])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    means = deltas[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": len(deltas),
        "delta": round(float(deltas.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "significant": bool(lo > 0 or hi < 0),
    }


def cell_label(cell):
    """Human-readable name for a ``(collection, hnsw_ef, rescore, oversampling)``
    cell, used as the report's row key and as the key of a ``--json`` dump — so
    it has to stay stable across runs and readable in a thesis table."""
    parts = [f"ef={cell.get('hnsw_ef') or 'server-default'}"]
    rescore = cell.get("rescore")
    if rescore is not None:
        parts.append(f"rescore={'on' if rescore else 'off'}")
    if cell.get("oversampling"):
        parts.append(f"oversampling={cell['oversampling']}")
    if cell.get("collection"):
        parts.append(cell["collection"])
    return " ".join(parts)


def ids_fingerprint(ids):
    """sha256 over the sorted point ids of a sample, as its identity.

    The sample itself is not stored: the corpus is re-extracted regularly and a
    stored id list would either rot silently or force a migration. A hash instead
    makes drift LOUD — the ids that came back either reproduce it or they do not,
    and the instrument says which. Same reasoning as the ``eval-XV-1`` corpus
    dump, where the artifact of record is the hash rather than the file."""
    import hashlib

    joined = "\n".join(sorted(str(value) for value in ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def reciprocal_rank_agreement(approx_ids, truth_ids):
    """Where in the exhaustive ranking the first MISSED item sits, as 1/rank.

    A companion to ``recall_at_k`` that says whether a cell loses the good stuff
    or the tail. Losing the exhaustive #1 and losing its #50 both cost the same
    recall point, but only the first is likely to change what a user sees.
    ``0.0`` means nothing was missed."""
    found = set(approx_ids)
    for position, point_id in enumerate(truth_ids, start=1):
        if point_id not in found:
            return 1.0 / position
    return 0.0


def summarise_misses(per_query_rr):
    """Mean reciprocal rank of the first miss, and the shallowest miss seen
    (the ``worst`` case, since a shallow miss is the damaging one)."""
    scored = [value for value in per_query_rr if value is not None]
    if not scored:
        return {"mean_rr": None, "shallowest": None}
    worst = max(scored)
    return {
        "mean_rr": round(sum(scored) / len(scored), 4),
        "shallowest": round(1 / worst) if worst else None,
    }


def format_recall(value):
    """Render a recall for the report, keeping ``None`` visible as a dash rather
    than printing a misleading 0.0000."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "     —"
    return f"{value:.4f}"
