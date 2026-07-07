"""Pure scoring for the mention-extraction eval.

Scores the end-to-end producer: raw speech text → NER → resolved deputies, against
a hand-labelled gold set of the deputies each speech mentions. Identity is the
canonical deputy ``name`` ("Apellido, Nombre") — the same value stored in
``Mention.name`` and in ``Speech.speaker`` — so predicted and gold are compared as
name sets per speech: TP = correctly named, FP = spurious (precision leak, e.g. a
wrong fuzzy match), FN = missed (recall leak, e.g. an ambiguous surname we dropped
or a name NER never caught).

No I/O — takes predicted/gold name lists, so it is unit-testable offline. Mirrors
``parse_scoring`` (same ``_prf`` shape) for a consistent report.
"""


def _prf(tp, fp, fn) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score(rows: list[dict], pred_key="pred_deputies", gold_key="gold_deputies") -> dict:
    """Aggregate scored ``rows`` (each with ``pred_key``/``gold_key`` name lists and an
    optional ``latency``). Returns micro P/R/F1 over all mentions, the per-speech
    exact-match rate (predicted set == gold set) and mean latency. The key pair lets the
    same scorer report deputies and non-deputies as separate dimensions."""
    tp = fp = fn = 0
    exact = 0
    latencies = []
    for row in rows:
        pred = set(row.get(pred_key, []))
        gold = set(row.get(gold_key, []))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += 1 if pred == gold else 0
        if row.get("latency") is not None:
            latencies.append(row["latency"])

    n = len(rows)
    return {
        "n": n,
        "micro": _prf(tp, fp, fn),
        "exact_match": round(exact / n, 4) if n else 0.0,
        "mean_latency": round(sum(latencies) / len(latencies), 4) if latencies else None,
    }
