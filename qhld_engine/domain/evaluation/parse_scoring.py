"""Pure scoring for the LLM-vs-rule-based query-parser comparison.

Scores each parser at the *resolved-filter* level — the end-to-end useful signal.
Because both parsers share the same ``EntityResolver``, the resolver is a
constant and the metric isolates the difference in *parsing* (slot assignment,
relative-date reasoning, entity extraction). Per structured slot (speaker, role,
group, date, lang, legislature) we count TP/FP/FN → precision/recall/F1, plus a
per-query exact-match and a soft topic-overlap score for the residual query.

No I/O — takes predicted filter dicts and the gold labels, so it is unit-testable
offline. Gold values may be a scalar, a list (the exact multi-value set expected,
order-insensitive — a prediction missing one of the values is wrong), or a date
range ``{"gte"/"lte": YYYYMMDD}`` scored with a few days' tolerance (LLM
relative-date arithmetic is often a day off).
"""

from collections import defaultdict
from datetime import date

SLOTS = ("speaker", "role", "group", "constituency", "date", "lang", "legislature")
_DATE_TOLERANCE_DAYS = 7


def _to_date(yyyymmdd):
    s = str(yyyymmdd)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def date_matches(pred: dict | None, gold: dict, tol_days=_DATE_TOLERANCE_DAYS) -> bool:
    """True iff ``pred`` and ``gold`` carry the same bound keys (gte/lte) and each
    bound is within ``tol_days`` of gold."""
    pred = pred or {}
    if set(pred) != set(gold):
        return False
    for key, gold_value in gold.items():
        if abs((_to_date(pred[key]) - _to_date(gold_value)).days) > tol_days:
            return False
    return True


def value_matches(pred, gold, slot) -> bool:
    """Scalar and list values compare as sets, so a multi-value slot matches
    regardless of order but fails when a value is missing or extra."""
    if pred is None:
        return False
    if slot == "date":
        return date_matches(pred, gold)
    as_set = lambda v: set(v) if isinstance(v, list) else {v}  # noqa: E731
    return as_set(pred) == as_set(gold)


def slot_counts(pred_filters: dict, gold: dict, slot: str) -> tuple[int, int, int]:
    """(tp, fp, fn) for one slot. A wrong non-null prediction on a gold slot counts
    as both a false positive and a false negative."""
    g = gold.get(slot)
    p = pred_filters.get(slot)
    if g is not None:
        if value_matches(p, g, slot):
            return 1, 0, 0
        return 0, (1 if p is not None else 0), 1      # missed / wrong
    return 0, (1 if p is not None else 0), 0          # hallucinated filter / clean


def _prf(tp, fp, fn) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _tokens(text):
    return {t for t in (text or "").lower().split() if len(t) > 2}


def topic_f1(pred_topic: str, gold_topic: str) -> float:
    """Soft token-overlap F1 between the residual semantic query and the gold
    topic — reported, not a headline metric (the residual is inherently fuzzy)."""
    pred, gold = _tokens(pred_topic), _tokens(gold_topic)
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    inter = len(pred & gold)
    if not inter:
        return 0.0
    precision, recall = inter / len(pred), inter / len(gold)
    return round(2 * precision * recall / (precision + recall), 4)


def score(rows: list[dict]) -> dict:
    """Aggregate scored ``rows`` (each with ``pred_filters``, ``gold``, optional
    ``pred_topic``/``gold_topic``, ``latency``). Returns per-slot and micro P/R/F1,
    per-query exact-match rate, mean topic-F1, and mean latency."""
    per_slot = {slot: [0, 0, 0] for slot in SLOTS}
    exact = 0
    topic_scores = []
    latencies = []
    for row in rows:
        pred, gold = row["pred_filters"], row["gold"]
        all_correct = True
        for slot in SLOTS:
            tp, fp, fn = slot_counts(pred, gold, slot)
            per_slot[slot][0] += tp
            per_slot[slot][1] += fp
            per_slot[slot][2] += fn
            if fp or fn:
                all_correct = False
        exact += 1 if all_correct else 0
        if "gold_topic" in row:
            topic_scores.append(topic_f1(row.get("pred_topic"), row["gold_topic"]))
        if row.get("latency") is not None:
            latencies.append(row["latency"])

    slots = {slot: _prf(*counts) for slot, counts in per_slot.items()}
    micro = _prf(
        sum(c[0] for c in per_slot.values()),
        sum(c[1] for c in per_slot.values()),
        sum(c[2] for c in per_slot.values()))
    n = len(rows)
    return {
        "n": n,
        "slots": slots,
        "micro": micro,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "topic_f1": round(sum(topic_scores) / len(topic_scores), 4) if topic_scores else None,
        "mean_latency": round(sum(latencies) / len(latencies), 4) if latencies else None,
    }
