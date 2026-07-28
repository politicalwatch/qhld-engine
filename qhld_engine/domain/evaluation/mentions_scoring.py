"""Pure scoring for the mention-extraction eval.

Scores the end-to-end producer: raw speech text → NER → resolved deputies, against
a hand-labelled gold set of the deputies each speech mentions. Identity is the
canonical deputy ``name`` ("Apellido, Nombre") — the same value stored in
``Mention.name`` and in ``Speech.speaker`` — so predicted and gold are compared as
name sets per speech: TP = correctly named, FP = spurious (precision leak, e.g. a
wrong fuzzy match), FN = missed (recall leak, e.g. an ambiguous surname we dropped
or a name NER never caught).

A second, narrower dimension scores OCCURRENCE COUNTS (``score_counts``): how many
times each person is named, not merely whether they are named. A name set cannot see
a change that only attaches more occurrences to somebody the speech already names, so
counts are the only way to measure one. Gold counts are annotated per speech and are
deliberately partial — only the people whose occurrences a reader has adjudicated.

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


def count_mismatches(rows, pred_key="pred_counts", gold_key="expected_counts") -> list[dict]:
    """Every annotated (speech, person) pair whose predicted occurrence count differs
    from gold, in gold-set order. Returned rather than printed so the CLI stays thin
    and the detail is testable."""
    mismatches = []
    for row in rows:
        pred = row.get(pred_key) or {}
        for name, gold in sorted((row.get(gold_key) or {}).items()):
            got = pred.get(name, 0)
            if got != gold:
                mismatches.append({
                    "id": row.get("id", row.get("speech_id")),
                    "reference": row.get("reference", ""),
                    "name": name, "gold": gold, "pred": got})
    return mismatches


def score_counts(rows, pred_key="pred_counts", gold_key="expected_counts") -> dict:
    """Aggregate occurrence counts over the annotated (speech, person) pairs only —
    a person with no gold count is not scored, and a speech with no map contributes
    nothing.

    Under- and over-counting are reported SEPARATELY and never netted: they are
    different failures. Under-counting means occurrences the tagger did not credit —
    a span NER never caught, or an ambiguous surname it dropped. Over-counting means
    occurrences credited to the wrong person, typically a homonym outside the catalog
    absorbing the surname of somebody inside it.

    ``missing`` is the subset of under-counting where the person was not predicted at
    all, which is already visible as a false negative in ``score``; the rest is
    invisible there."""
    pairs = exact = gold_total = pred_total = under = over = missing = 0
    speeches = 0
    for row in rows:
        gold_counts = row.get(gold_key) or {}
        if not gold_counts:
            continue
        speeches += 1
        pred = row.get(pred_key) or {}
        for name, gold in gold_counts.items():
            got = pred.get(name, 0)
            pairs += 1
            exact += 1 if got == gold else 0
            gold_total += gold
            pred_total += got
            under += max(gold - got, 0)
            over += max(got - gold, 0)
            missing += 1 if got == 0 and gold > 0 else 0
    return {
        "speeches": speeches,
        "pairs": pairs,
        "exact": exact,
        "exact_rate": round(exact / pairs, 4) if pairs else 0.0,
        "gold_occurrences": gold_total,
        "pred_occurrences": pred_total,
        "under": under,
        "over": over,
        "missing": missing,
    }
