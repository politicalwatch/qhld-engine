"""Scoring for "is this Spanish paragraph a rendering of that co-official one".

Two questions that are easy to confuse and that different instruments answer differently:

**RANK** — of all the Spanish paragraphs in this speech, is the true partner the
highest-scoring one? This is what the alignment needs, and it is a *within-speech*
question: a measure can rank perfectly while every score sits in the same narrow band.

**GATE** — is there a threshold that admits every true pair and rejects every non-pair?
This is what deciding *whether a rendering exists at all* needs, and it is much harder,
because within one speech every paragraph is about the same subject with the same names
and figures in it. Six embedding models, character n-grams, a translation pivot and a
cross-encoder all rank acceptably and all fail this for Basque.

Reporting one and calling it the other is how a measure gets adopted that cannot do the
job asked of it, so they are always printed side by side.
"""

_LANGS = ("ca", "gl", "eu")


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def score(rows):
    """Aggregate per-pair scores into the RANK and GATE views.

    ``rows`` is one entry per (original paragraph, candidate) comparison::

        {"lang": "eu", "video_id": "761240", "source": 3, "candidate": 11,
         "score": 0.81, "is_pair": True, "is_rendering": True}

    ``is_rendering`` marks a paragraph the goldset knows was never spoken. A candidate
    that is a rendering but not this source's pair is **excluded from both classes** —
    it is a rendering the labeller could not confidently attribute, so counting it as a
    negative would score the labelling rather than the instrument.
    """
    by_lang = {}
    for row in rows:
        bucket = by_lang.setdefault(row["lang"], {"true": [], "non": [], "ranks": {}})
        if row["is_pair"]:
            bucket["true"].append(row["score"])
        elif not row.get("is_rendering"):
            bucket["non"].append(row["score"])
        key = (row["video_id"], row["source"])
        best = bucket["ranks"].setdefault(key, {"best": None, "score": None,
                                                "partners": set()})
        if row["is_pair"]:
            best["partners"].add(row["candidate"])
        if best["score"] is None or row["score"] > best["score"]:
            best["score"], best["best"] = row["score"], row["candidate"]

    report = {}
    for lang, bucket in by_lang.items():
        ranked = [r for r in bucket["ranks"].values() if r["partners"]]
        top1 = sum(1 for r in ranked if r["best"] in r["partners"])
        true, non = bucket["true"], bucket["non"]
        report[lang] = {
            "rank_top1": top1,
            "rank_total": len(ranked),
            "true_min": min(true) if true else None,
            "true_median": _percentile(true, 0.5) if true else None,
            "non_max": max(non) if non else None,
            "non_p90": _percentile(non, 0.9) if non else None,
            # Positive means a threshold exists that separates every pair from every
            # non-pair. Negative is the size of the overlap, which is the honest figure.
            "gate": (min(true) - max(non)) if (true and non) else None,
            "n_true": len(true),
            "n_non": len(non),
        }
    return report


def score_decisions(rows):
    """For an instrument that answers yes/no rather than scoring — the LLM tier.

    Specificity is the number that matters and the one an accuracy figure hides: the
    negatives outnumber the pairs several times over, so "always no" scores well on
    accuracy while being useless.
    """
    report = {}
    for row in rows:
        if row["is_pair"] is False and row.get("is_rendering"):
            continue
        cell = report.setdefault(row["lang"], {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        said, real = bool(row["decision"]), bool(row["is_pair"])
        cell["tp" if (real and said) else "fn" if real
             else "fp" if said else "tn"] += 1
    for cell in report.values():
        pairs, non = cell["tp"] + cell["fn"], cell["tn"] + cell["fp"]
        cell["recall"] = cell["tp"] / pairs if pairs else None
        cell["specificity"] = cell["tn"] / non if non else None
        cell["precision"] = (cell["tp"] / (cell["tp"] + cell["fp"])
                             if cell["tp"] + cell["fp"] else None)
    return report


def speech_level(rows):
    """Does the instrument find at least ONE rendering in each speech that has one?

    The existence question needs a single confirmation, not all of them, so an
    instrument with mediocre per-pair recall can still be right about every speech —
    which is exactly what the LLM tier does and what per-pair recall alone hides.
    """
    seen, found = {}, {}
    for row in rows:
        if not row.get("has_renderings", True):
            continue
        key = row["video_id"]
        seen[key] = row["lang"]
        if row["is_pair"] and row.get("decision"):
            found[key] = True
    return {"speeches": len(seen), "detected": len(found),
            "missed": sorted(set(seen) - set(found))}


def score_split(rows):
    """Did the classifier put the right paragraphs in the rendering block?

    Reported three ways because they fail differently. **Exact** is the honest headline —
    a speech is right only if its rendering set matches gold exactly, since one paragraph
    on the wrong side is a paragraph missing from, or invented in, the record of what was
    said. **Recall** misses are renderings left in the as-delivered block (the
    self-translation defect). **Precision** misses are worse: spoken words removed from
    the record.
    """
    per_lang, exact, refused = {}, 0, 0
    tp = fp = fn = 0
    misses = []
    for row in rows:
        gold, pred = row["gold"], row["predicted"]
        cell = per_lang.setdefault(row["lang"], {"speeches": 0, "exact": 0,
                                                 "tp": 0, "fp": 0, "fn": 0})
        cell["speeches"] += 1
        refused += bool(row["undecided"])
        hit, spurious, missed = (gold & pred), (pred - gold), (gold - pred)
        tp += len(hit); fp += len(spurious); fn += len(missed)
        cell["tp"] += len(hit); cell["fp"] += len(spurious); cell["fn"] += len(missed)
        if gold == pred:
            exact += 1
            cell["exact"] += 1
        else:
            misses.append({"video_id": row["video_id"], "lang": row["lang"],
                           "missed": sorted(missed), "spurious": sorted(spurious),
                           "undecided": row["undecided"]})

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else None
        r = t / (t + f_n) if t + f_n else None
        f = (2 * p * r / (p + r)) if p and r else None
        return {"precision": p, "recall": r, "f1": f, "tp": t, "fp": f_p, "fn": f_n}

    for cell in per_lang.values():
        cell.update(prf(cell["tp"], cell["fp"], cell["fn"]))
    return {"speeches": len(rows), "exact": exact, "refused": refused,
            "micro": prf(tp, fp, fn), "per_lang": per_lang, "misses": misses}


def score_pairs(rows):
    """Did the classifier see that each original HAS a rendering — the other half.

    ``score_split`` asks which paragraphs left the record of what was said. It cannot see
    the opposite failure: an original whose translation the alignment missed, which then
    stands in the Spanish block *as well as* the as-delivered one. Because such a paragraph
    is in **both** blocks, ``score_split`` excludes it by construction, and the whole
    change that introduced it moved that score not at all.

    Two figures, from gold alone:

    **Pair recall** — a gold pair's SOURCE must not stand in the Spanish block. A source
    definitely has a translation, so this direction is sound.

    **The duplication census** — every paragraph in both blocks, in three classes with no
    unknown left over. ``defect`` is a gold pair source, i.e. a translation exists and was
    not found. ``es_side`` is in gold ``renderings``, i.e. never spoken and so wrong in the
    delivered block too — already counted by ``score_split``, and repeated here only so the
    census adds up. ``expected`` is neither: spoken, never translated, and therefore
    correctly in both, which is most of them.

    **Precision has no sound figure here and must not be invented.** Gold ``pairs`` is
    deliberately partial — a rendering the labeller could not confidently attribute is left
    out — so a paragraph absent from ``pairs`` is *unknown*, not untranslated. That is why
    the ``expected`` class is reported as a count to read, not as a score.

    Two definitions were tried and rejected, each silently dropping the cases most worth
    seeing, and they disagreed on the denominator (115 vs 123 vs 126):

    1. asking the aligner which originals it paired misses a source whose paragraph the
       detector read as SPANISH — which is the entire code-mixed class;
    2. asking only speeches that produced two blocks misses the one-block speeches, where
       most of ``score_split``'s own misses live.

    An original in a speech that produced **no Spanish block** is *unanswerable* rather than
    correct: nothing can stand in a block that does not exist. It is excluded from the
    denominator and reported on its own — the same treatment ``score`` gives an
    unattributable rendering, and for the same reason. Counting them as passes would both
    flatter the figure and move the denominator silently the day the one-block shape is
    fixed and those originals become answerable for the first time.
    """
    answerable = missed = unanswerable = 0
    census = {"defect": [], "es_side": [], "expected": []}
    for row in rows:
        gold, pairs = row["gold"], row["sources"]
        both = row["delivered"] & row["spanish"]
        for source in sorted(pairs):
            if not row["spanish"]:
                unanswerable += 1
                continue
            answerable += 1
            missed += source in row["spanish"]
        for paragraph in sorted(both):
            key = ("defect" if paragraph in pairs
                   else "es_side" if paragraph in gold else "expected")
            census[key].append({"video_id": row["video_id"], "lang": row["lang"],
                                "paragraph": paragraph})
    return {"answerable": answerable, "missed": missed, "unanswerable": unanswerable,
            "recall": (answerable - missed) / answerable if answerable else None,
            "census": census}


def format_pairs(report):
    lines = [
        f"  originals whose rendering was found "
        f"{report['answerable'] - report['missed']}/{report['answerable']}"
        + (f"  recall {report['recall']:.3f}" if report["recall"] is not None else "")
        + (f"   (+{report['unanswerable']} unanswerable — their speech has no Spanish "
           f"block)" if report["unanswerable"] else ""),
        f"  in both blocks: {len(report['census']['defect'])} a translation exists and was "
        f"not found · {len(report['census']['es_side'])} never spoken · "
        f"{len(report['census']['expected'])} spoken and never translated (correct)",
    ]
    for key, label in (("defect", "translation missed"), ("es_side", "never spoken")):
        for row in report["census"][key]:
            lines.append(f"    {row['video_id']} [{row['lang']}] "
                         f"paragraph {row['paragraph']} — {label}")
    return "\n".join(lines)


def format_split(report, label):
    lines = [label,
             f"  exact block match {report['exact']}/{report['speeches']} speeches"
             f"   ({report['refused']} refused)"]
    micro = report["micro"]
    lines.append(
        f"  paragraphs  P {micro['precision']:.2f}  R {micro['recall']:.2f}  "
        f"F1 {micro['f1']:.2f}   tp/fp/fn {micro['tp']}/{micro['fp']}/{micro['fn']}"
        if micro["f1"] is not None else "  paragraphs  n/a")
    for lang in _LANGS:
        cell = report["per_lang"].get(lang)
        if cell:
            lines.append(
                f"    {lang}  exact {cell['exact']}/{cell['speeches']}   "
                f"tp/fp/fn {cell['tp']}/{cell['fp']}/{cell['fn']}")
    return "\n".join(lines)


def format_report(report, label):
    """One line per language, RANK and GATE together, never one without the other."""
    lines = [f"{label}"]
    for lang in _LANGS:
        cell = report.get(lang)
        if not cell:
            continue
        gate = cell["gate"]
        lines.append(
            f"  {lang}  rank-1 {cell['rank_top1']:>3}/{cell['rank_total']:<3}"
            f"  true min {cell['true_min']:.3f} med {cell['true_median']:.3f}"
            f"  non-pair max {cell['non_max']:.3f} p90 {cell['non_p90']:.3f}"
            f"  GATE {gate:+.3f}"
            f"{'  <- separable' if gate and gate > 0 else ''}"
            if cell["true_min"] is not None and cell["non_max"] is not None else
            f"  {lang}  rank-1 {cell['rank_top1']}/{cell['rank_total']}")
    return "\n".join(lines)
