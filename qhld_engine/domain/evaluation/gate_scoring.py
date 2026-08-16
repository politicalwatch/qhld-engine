"""Pure scoring for the intent gate.

The gate refuses queries that are not speech searches. Measuring it needs two
arms that are scored the same way but read in opposite directions: over
LEGITIMATE probes a refusal is a false positive, over JUNK probes a refusal is
the point. Both come back as the same row shape, so this module scores an arm
and the caller says which reading applies.

Three properties of the gate shape the metric:

1. **The two refusal sites are different defects and are never blended.**
   ``refused:flag`` is the parser judging the query not to be a search;
   ``refused:empty`` is a parse that came back with no topic, no filters and
   nothing blocking. A single rate would hide which one is firing, and they have
   different fixes — a prompt change versus a resolution change.

2. **The verdict is not deterministic.** The gate runs on an LLM call, so the
   same query can pass on one run and be refused on the next. Averaging that away
   would report a stability the system does not have, so a query's outcome is
   summarised as a frequency ``k/R`` and the headline is a BAND across runs, per
   the memoria's rule for LLM-dependent numbers (cite a band, never a point).

3. **An intermittent refusal is a real failure.** A user whose query is refused
   two times in five has a broken search, not a 60%-working one. So ``ever``
   (refused at least once) is reported beside the median rate, and the two are
   expected to differ.

No I/O: takes outcome rows and returns numbers, so it is unit-testable offline.
"""

from collections import defaultdict

PASS = "pass"
REFUSED_FLAG = "refused:flag"
REFUSED_EMPTY = "refused:empty"
OUTCOMES = (PASS, REFUSED_FLAG, REFUSED_EMPTY)

# How an arm reads its refusals. The scoring is identical; only the name of the
# headline rate changes, and getting that backwards is the easiest way to publish
# a wrong conclusion — hence naming it in one place.
#
# NON_SEARCH and JUNK are both "should be refused" but they are NOT interchangeable
# and must not be merged into one rate. A non-search carries a real parliamentary
# topic, so nothing downstream stops it and a miss returns relevant speeches for a
# request nobody made. Off-domain junk that slips is still floored to zero results,
# so a miss there costs a paid pipeline and the wrong error message. Same label,
# different blast radius.
LEGITIMATE = "legitimate"
NON_SEARCH = "non-search"
JUNK = "junk"
ARMS = (LEGITIMATE, NON_SEARCH, JUNK)


def is_refusal(outcome: str) -> bool:
    return outcome != PASS


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def score_run(rows: list[dict]) -> dict:
    """One pass over one arm. ``rows`` carry ``id``, ``outcome`` and ``class``.

    Returns the refusal rate overall, split by site, and split by hazard class.
    """
    total = len(rows)
    refused = [r for r in rows if is_refusal(r["outcome"])]
    by_site = {site: sum(1 for r in refused if r["outcome"] == site)
               for site in (REFUSED_FLAG, REFUSED_EMPTY)}
    by_class = defaultdict(lambda: {"n": 0, "refused": 0})
    for row in rows:
        bucket = by_class[row.get("class", "?")]
        bucket["n"] += 1
        bucket["refused"] += int(is_refusal(row["outcome"]))
    for bucket in by_class.values():
        bucket["rate"] = _rate(bucket["refused"], bucket["n"])
    return {
        "n": total,
        "refused": len(refused),
        "rate": _rate(len(refused), total),
        "by_site": by_site,
        "by_site_rate": {site: _rate(count, total) for site, count in by_site.items()},
        "by_class": dict(by_class),
    }


def score_arm(runs: list[list[dict]]) -> dict:
    """Aggregate ``R`` passes over the same arm.

    The per-run rates become a band (min/median/max). Per query we keep how often
    it was refused and under which site(s) — a query refused under BOTH sites
    across runs is worth seeing, because it means the parse is moving enough to
    change which failure occurs, not merely whether one does.
    """
    if not runs:
        raise ValueError("no runs to score")
    per_run = [score_run(rows) for rows in runs]
    rates = sorted(run["rate"] for run in per_run)
    repeats = len(runs)

    queries: dict[str, dict] = {}
    for rows in runs:
        for row in rows:
            entry = queries.setdefault(row["id"], {
                "id": row["id"],
                "class": row.get("class", "?"),
                "query": row.get("query", ""),
                "refused": 0,
                "sites": set(),
                "routes": set(),
            })
            if is_refusal(row["outcome"]):
                entry["refused"] += 1
                entry["sites"].add(row["outcome"])
            if row.get("route"):
                entry["routes"].add(row["route"])

    for entry in queries.values():
        entry["repeats"] = repeats
        entry["frequency"] = _rate(entry["refused"], repeats)
        # Refused sometimes but not always: the gate's verdict on this query is
        # not a property of the query, it is noise in the parse. Called out
        # separately because it is invisible in any averaged rate.
        entry["unstable"] = 0 < entry["refused"] < repeats
        entry["sites"] = sorted(entry["sites"])
        entry["routes"] = sorted(entry["routes"])

    ever = sum(1 for e in queries.values() if e["refused"])
    always = sum(1 for e in queries.values() if e["refused"] == repeats)
    return {
        # An arm that never refused anything has NOT measured a rate of zero — it has
        # failed to find a refusal in n probes, which is a much weaker claim. The rule
        # of three gives the 95% upper bound, and the independent unit is the QUERY:
        # repeating the same probe tests the model's stability, not the product's
        # coverage, so passes do not enter the denominator.
        "ceiling_95": _rate(3, per_run[0]["n"]) if ever == 0 else None,
        "repeats": repeats,
        "n": per_run[0]["n"],
        "per_run": per_run,
        "band": {"min": rates[0], "median": _median(rates), "max": rates[-1]},
        # Two honest denominators for the same fact: how many queries the gate
        # refuses on a typical run, and how many it refuses at all.
        "ever": ever,
        "ever_rate": _rate(ever, per_run[0]["n"]),
        "always": always,
        "always_rate": _rate(always, per_run[0]["n"]),
        "unstable": sorted(
            (e for e in queries.values() if e["unstable"]),
            key=lambda e: (-e["refused"], e["id"])),
        "queries": sorted(queries.values(), key=lambda e: e["id"]),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2, 4)


def confusion(legitimate: dict, positives: dict) -> dict:
    """The 2x2 the gate has never had, built from two scored arms at their median
    run.

    Positive = "refuse". So a refusal on the ``positives`` arm is a true positive
    and a refusal on ``legitimate`` is a false positive; precision asks how much of
    what the gate blocks deserved it, recall how much of what it should block it
    caught.

    ``positives`` is ONE refuse-arm, never the two of them added together. The
    non-search and junk arms answer different questions — see the arm constants —
    and a combined recall would average an unbackstopped failure with a backstopped
    one into a number that describes neither.
    """
    n_legit, n_pos = legitimate["n"], positives["n"]
    fp = round(legitimate["band"]["median"] * n_legit)
    tp = round(positives["band"]["median"] * n_pos)
    fn = n_pos - tp
    tn = n_legit - fp
    precision = _rate(tp, tp + fp) if (tp + fp) else 0.0
    recall = _rate(tp, tp + fn) if (tp + fn) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "false_positive_rate": legitimate["band"]["median"],
        "suppression": positives["band"]["median"],
        "precision": precision, "recall": recall, "f1": f1,
    }
