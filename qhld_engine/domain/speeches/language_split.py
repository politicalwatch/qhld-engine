"""How a speech's text divides into blocks, and what those blocks are to each other.

Pure logic — no HTTP, no DB. The language detector is injected as ``detect(str) -> str |
None``; the py3langid adapter that backs it lives in the infrastructure layer.

A block is a **role**, not a slice of the document: ``original`` holds what was delivered,
whatever languages it mixes, and a second block holds a Spanish rendering of it when the
Diario published one. Three shapes cover the corpus:

1. co-official **and** a Spanish rendering exists — appended at the end or interleaved
   paragraph by paragraph. Two blocks.
2. mostly co-official, Spanish only as quotations, no rendering. One block.
3. mostly Spanish, co-official only as a greeting or farewell. One block.

Telling them apart needs three things, because no one of them is sufficient:

- **whether the Spanish renders the original**, which distinguishes a translation from a
  speaker who simply switched languages — but which mistakes a code-mixed speech for a
  translation, because both languages discuss the same subject with the same names in it;
- **how long the clip runs**, which says how much of the text can actually have been
  spoken — but which cannot separate "a rendering plus some spoken Spanish" from "all of
  it was spoken";
- **how much of what was spoken is co-official**, which is what "mostly Catalan" versus
  "mostly Spanish with a Catalan greeting" actually means.

Where they disagree, or where there is no clip to measure against, the speech is left
undecided rather than guessed at: a wrong verdict here renames the language of a speech
in search, in the reader and in the corpus description.
"""

from collections import defaultdict
from typing import NamedTuple

from qhld_engine.domain.speeches.language_runs import (
    CO_LANGS,
    paragraph_spans,
    quotation_spans,
    sentence_spans,
)


class Block(NamedTuple):
    """One block of a speech. ``partial`` marks a rendering that covers only part of
    the original — never set on an ``original`` block, which is complete by
    construction.

    ``langs`` names every language the block is really in, commonest first, with ``lang``
    always the first of them. The as-delivered block holds what was spoken whatever
    languages that mixes, so calling a mostly-Spanish speech with a Basque passage simply
    ``es`` hides that the Basque was spoken at all — and calling it ``eu`` would be worse.
    """

    lang: str
    text: str
    original: bool
    partial: bool = False
    langs: tuple = ()


class Split(NamedTuple):
    """``language`` is the language of the delivered speech. ``undecided`` means the
    evidence did not settle the shape and ``blocks`` is a provisional reading."""

    language: str
    blocks: list
    undecided: bool = False


# How fast a speech is delivered, in characters per second. Measured over 3,057
# monolingual Spanish interventions of 200 characters or more against their own clips:
# p01 7.9, median 13.3, p99 18.4, widened 10%. Catalan (12.9) and Galician (12.3) sit
# inside it, and so does the one Basque speech whose original demonstrably covers its
# whole clip (13.0), so it is used for every language.
SPEECH_RATE_MIN = 8.8
SPEECH_RATE_MAX = 18.9


# How closely a paragraph must read as a rendering of another before the alignment will
# pair them. A cosine between multilingual sentence embeddings, not surface overlap.
#
# ONE figure, for every language pair, which is the whole reason for the change: how much
# vocabulary a translation shares with its original is a property of the pair (Galician
# reproduces about half the Spanish, Catalan a fifth, Basque almost none), so the measure
# it replaced needed a floor set per speech and a list of pairs it simply could not read.
# Meaning is comparable across pairs, so the exception list and the relative floor both
# go.
#
# Swept over the bitext gold set (`qhld eval bitext --instrument classifier`). There is a
# CLIFF just below 0.575 — Catalan false positives go 1 -> 8 and Basque 0 -> 19, i.e.
# spoken words being stripped out of the record — and a wide flat plateau above it, out to
# 0.775 (ca), 0.725 (gl) and 0.700 (eu), beyond which real renderings start being missed.
# 0.65 sits inside all three plateaus with room on both sides; the exact value is not
# load-bearing, its distance from the cliff is.
SIMILARITY_FLOOR = 0.65

# How much longer than a co-official paragraph everything claimed to render it may run,
# all of it together, before the claim stops being credible. A courtesy line does not
# render a paragraph twenty times its length, however much the two share by being about
# the same debate — and treating it as one takes that paragraph out of the record of what
# was said.
#
# Per SOURCE paragraph, not per Spanish one. The two are not the same test, and the
# difference is what the alignment gets wrong: nothing costs it anything to over-claim, so
# a paragraph that already has a good partner will happily take a second, weaker one as
# well. Checked against the Spanish side alone, that second claim looks proportionate on
# its own; checked against what it says it renders, it is one paragraph claiming to be
# said twice over. 742439 is the case — 2,040 characters of spoken Spanish claimed at 0.81
# by a 536-character paragraph whose real rendering scores 0.93.
#
# Swept over the bitext gold set: a flat plateau from 3.2 to 4.5, breaking at 5.0. Both
# edges are pinned by speeches in that set — 736259 needs more than 3.12 (its Catalan
# trails off mid-sentence and the Spanish finishes the thought), 742439 less than 4.70 and
# 726204 less than 5.27. Corpus-wide the median rendering runs 1.04 times its original.
RENDERING_MAX_EXPANSION = 4.0


# A translation exists when it covers this much of the original, and is complete rather
# than partial when it covers this much. Between the two it is a real but partial
# rendering; below the first the Spanish is not a translation at all.
COVERAGE_FLOOR = 0.5
COVERAGE_COMPLETE = 0.85


# What share of the delivered speech must be co-official for the speech to be one. This
# is the difference between a Catalan speech quoting Spanish (66-90%) and a Spanish
# speech with a Catalan passage in it (27-30%).
CO_SHARE_FLOOR = 0.5

# Characters of Spanish a complete rendering runs to, per character of original.
# Measured on the speeches whose original demonstrably covers its own clip. Used only
# where `renders` is blind, to guess whether a Basque speech's rendering is partial —
# n=7 behind the Basque figure, so treat it as provisional.
RENDERING_RATIO = {"ca": 1.00, "gl": 1.03, "eu": 1.24}

# What share of a block a language needs before it is named among the block's languages.
# A courtesy line should not make a speech bilingual: greetings sit at or below 3% of the
# block (726710 opens "Gracias, señor presidente. Eskerrik asko." — 41 of 1,430
# characters) while a genuine co-official passage is an order of magnitude larger (774737
# 13%, 760011 21%, 738994 27%). The gap is wide, so the exact figure is not delicate.
# The dominant language is exempt and always named, however short the block.
LANG_SHARE_FLOOR = 0.05

# A side shorter than this fraction of the speech is not a block of its own.
_MIN_RATIO = 0.15



def _langs_of(text, lang, detect):
    """Every language ``text`` is really in, commonest first, ``lang`` always first.

    Read from the block's own text rather than threaded down from the runs, so the
    two-block case, both single-block cases and the provisional reading are all described
    the same way and none of them can be forgotten.
    """
    weights = {}
    for language, start, end in paragraph_spans(text.strip(), detect):
        weights[language] = weights.get(language, 0) + (end - start)
    total = sum(weights.values())
    if not total:
        # Nothing in it is long enough to read — a one-line intervention.
        return (lang,)
    others = sorted((l for l in weights if l != lang),
                    key=lambda l: -weights[l])
    return (lang, *(l for l in others if weights[l] / total >= LANG_SHARE_FLOOR))


def _decided(language, blocks, undecided, detect):
    """Assemble the verdict, naming each block's languages from its own text."""
    return Split(language,
                 [block._replace(langs=_langs_of(block.text, block.lang, detect))
                  for block in blocks],
                 undecided)


def split_languages(text, detect, duration=None, similarity=None):
    """Split ``text`` into its blocks.

    ``duration`` is the clip's length in seconds, or ``None`` where the video is not
    published yet. ``similarity(a, b) -> float`` scores how much one paragraph reads as a
    rendering of another; it is injected like ``detect`` so this module stays pure, and
    without it the speech is left undecided rather than guessed at — refusing is the
    standing answer when the evidence is missing, and a silent fall back to a weaker
    instrument is how a wrong verdict gets stored with confidence.
    """
    stripped = text.strip()
    if not stripped:
        return Split("es", [], False)
    # Paragraphs, not merged runs: a rendering and the Spanish the speaker then went on
    # to deliver are adjacent paragraphs, and merging them would hide their boundary.
    runs = paragraph_spans(stripped, detect)
    if not runs:
        # Nothing in it is long enough for the detector to read — a one-line
        # intervention ("Sí."). It is still a speech and still needs its block; Spanish
        # is the safe assumption, being the language all but a fortieth are given in.
        return _decided("es", [Block("es", text, True)], False, detect)

    co_runs = [run for run in runs if run[0] in CO_LANGS]
    es_runs = [run for run in runs if run[0] == "es"]
    if not co_runs:
        return _decided("es", [Block("es", text, True)], False, detect)

    co_lang = _dominant(co_runs)
    co_chars = sum(end - start for _, start, end in co_runs)
    if not es_runs or co_chars / len(stripped) > 1 - _MIN_RATIO:
        return _decided(co_lang, [Block(co_lang, text, True)], False, detect)

    if similarity is None:
        return _decided(co_lang, _single_cut(stripped, text, detect, co_lang),
                    True, detect)

    comparable = True
    rendered_co, rendering_es = _align_renderings(stripped, co_runs, es_runs,
                                                  similarity)
    rendered = sum(co_runs[j][2] - co_runs[j][1] for j in rendered_co)
    coverage = rendered / co_chars if co_chars else 0.0
    # Everything that is not a rendering was delivered. Built by subtraction rather
    # than by selection, because the as-delivered block is the record of what was said:
    # anything the alignment cannot account for belongs in it, including a greeting the
    # detector misreads. Selecting only what is *provably* delivered loses those.
    #
    # Only where the score means something, though. Basque shares no vocabulary with
    # Spanish, so NOTHING matches and "not a rendering" is true of the entire
    # interpretation — subtracting there would copy the whole Spanish side into the
    # record of what was said. With no evidence either way, claim none of it.
    delivered_es = ({i for i in range(len(es_runs)) if i not in rendering_es}
                    if comparable else set())
    spoken = co_chars + sum(end - start for i, (_, start, end) in enumerate(es_runs)
                            if i in delivered_es)

    if _renders_whole_speech(comparable, coverage, co_chars, spoken, stripped,
                             duration, es_runs, co_lang):
        blocks = _rendered_blocks(stripped, co_runs, es_runs, delivered_es, co_lang,
                                  coverage, comparable)
        return _decided(co_lang, blocks, duration is None, detect)
    if _all_of_it_was_spoken(comparable, coverage, co_chars, spoken, stripped, duration):
        dominant = _dominant(runs)
        return _decided(dominant, [Block(dominant, text, True)], duration is None,
                        detect)

    # Nothing places it. Keep the shape the single-boundary reading gives, so a speech
    # that is probably an ordinary pair does not lose its Spanish block while it waits,
    # and mark it so the adjudication pass can find it.
    return _decided(co_lang, _single_cut(stripped, text, detect, co_lang),
                    True, detect)


def _renders_whole_speech(comparable, coverage, co_chars, spoken, stripped, duration,
                          es_runs, co_lang):
    """Is this a co-official speech with a Spanish rendering of it?"""
    fits = _fits(spoken, duration)
    if comparable:
        return (coverage >= COVERAGE_FLOOR
                and co_chars / spoken >= CO_SHARE_FLOOR
                and fits is not False)
    # Basque: no vocabulary in common, so the clip decides alone. The original alone
    # accounting for the audio is what makes the Spanish a translation of it.
    return _fits(co_chars, duration) is True and _fits(len(stripped), duration) is False


def _all_of_it_was_spoken(comparable, coverage, co_chars, spoken, stripped, duration):
    """Was the whole document delivered — no rendering, just a speech that changed
    language?"""
    if _fits(len(stripped), duration) is False:
        return False
    if comparable:
        return coverage < COVERAGE_FLOOR or co_chars / spoken < CO_SHARE_FLOOR
    return _fits(len(stripped), duration) is True


def _fits(chars, duration):
    """Could ``chars`` have been spoken in ``duration`` seconds? ``None`` when there is
    no clip to ask."""
    if not duration:
        return None
    return SPEECH_RATE_MIN <= chars / duration <= SPEECH_RATE_MAX



def _rendered_blocks(stripped, co_runs, es_runs, delivered_es, co_lang, coverage,
                     comparable):
    """The two blocks of a speech that has a rendering.

    The original block carries every co-official run **plus** any Spanish that renders
    nothing — a citation read aloud, an aside, a closing stretch the speaker delivered in
    Spanish. All of it was spoken, so all of it belongs in the record of what was said;
    the Spanish block keeps the whole Spanish side as the Diario prints it, so those
    stretches appear in both. That duplication is deliberate: each block is complete for
    its own purpose.

    A quotation on its own paragraph is one of those stretches and cannot be found by the
    alignment, which works in runs: having no language of its own it attaches to the
    paragraph before it, and when that paragraph is a rendering it leaves the record along
    with it. So it is added back here, by position.
    """
    delivered = [(start, end) for _, start, end in
                 co_runs + [run for i, run in enumerate(es_runs) if i in delivered_es]]
    delivered += [span for span in quotation_spans(stripped)
                  if not _within(span, delivered)]
    original = _join(stripped, sorted(delivered))
    spanish = _join(stripped, [(start, end) for _, start, end in es_runs])
    partial = _is_partial(co_runs, es_runs, coverage, comparable, co_lang)
    return [Block(co_lang, original, True), Block("es", spanish, False, partial)]


def _within(span, spans):
    start, end = span
    return any(s <= start and end <= e for s, e in spans)


def _is_partial(co_runs, es_runs, coverage, comparable, co_lang):
    if comparable:
        return coverage < COVERAGE_COMPLETE
    # Without a usable rendering score, length is the only evidence: a complete rendering
    # runs to a known multiple of its original, so a much shorter one is missing some.
    co_chars = sum(end - start for _, start, end in co_runs)
    es_chars = sum(end - start for _, start, end in es_runs)
    expected = RENDERING_RATIO.get(co_lang, 1.0) * co_chars
    return es_chars < 0.8 * expected


def _align_renderings(stripped, co_runs, es_runs, similarity):
    """Which paragraphs render which, as ``(rendered co indices, rendering es indices)``.

    A monotone alignment rather than a pairing, because **the Diario does not translate
    paragraph for paragraph**. The interpreter re-paragraphs freely: one Catalan paragraph
    may be rendered by two Spanish ones, two may collapse into one, and the Spanish
    typically runs longer. Insisting on a one-to-one correspondence therefore leaves most
    of a fully-translated speech looking unrendered — measured at 0.48 coverage on a
    speech that is rendered nearly end to end.

    What does hold is the **order**: a rendering follows what it renders, and the two
    sides advance together. So this walks both sequences once, allowing each side to
    consume several of the other's paragraphs, and lets a paragraph that matches nothing
    fall out — on the Spanish side, that is a stretch the speaker delivered in Spanish.

    Nothing in that walk costs anything to over-claim: leaving a Spanish paragraph unpaired
    scores zero, so any pairing above the floor is free profit, and a paragraph that
    already has a good partner will take a second, weaker one as well. So proportionality
    is enforced afterwards, measured **per source paragraph** over everything claimed for
    it — a paragraph cannot have been said four times over, however plausible each claim
    looks on its own.

    Enforced afterwards rather than inside the pairing because the two sides are not
    symmetrical in what they cost. Dropping a Spanish paragraph returns it to the record of
    what was said; dropping a co-official one merely lowers ``coverage``, which would mark
    complete renderings ``partial`` and, far enough down, refuse healthy speeches outright.
    Measured: enforcing it inside the pairing refuses six speeches whose blocks are textbook
    translation pairs.
    """
    floor = SIMILARITY_FLOOR
    co_text = [stripped[start:end] for _, start, end in co_runs]
    es_text = [stripped[start:end] for _, start, end in es_runs]
    reads_as = {}

    def score(i, j):
        # A pairing is worth how much one paragraph reads like the other, and nothing
        # else. It used to be weighted by the length it accounted for, so that the walk
        # would prefer explaining long paragraphs over collecting incidental short ones —
        # but that weight is capped by the SHORTER side, so a long paragraph outbids a
        # short one for the same rendering however much better the short one reads.
        # 741705 is the case: 0.66 x 795 characters beat 0.893 x 351 for the same Spanish
        # paragraph, and the true pair lost. Refusing incidental pairings is now
        # SIMILARITY_FLOOR's job, and refusing disproportionate ones is
        # RENDERING_MAX_EXPANSION's; neither existed when the weight was written, and
        # both do it on better evidence than length.
        value = reads_as[i, j] = similarity(co_text[i], es_text[j])
        return value if value >= floor else 0.0

    rows, cols = len(co_runs), len(es_runs)
    best = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    move = [[None] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        for j in range(cols + 1):
            if not i and not j:
                continue
            top, choice = -1.0, None
            if i and best[i - 1][j] > top:
                top, choice = best[i - 1][j], ("skip_co", i - 1, j)
            if j and best[i][j - 1] > top:
                top, choice = best[i][j - 1], ("skip_es", i, j - 1)
            if i and j:
                pairing = score(i - 1, j - 1)
                if pairing:
                    # 1:1, or either side consuming another of the other's paragraphs.
                    for previous in ((i - 1, j - 1), (i, j - 1), (i - 1, j)):
                        if best[previous[0]][previous[1]] + pairing > top:
                            top = best[previous[0]][previous[1]] + pairing
                            choice = ("match", *previous)
            best[i][j], move[i][j] = top, choice

    rendered_co = set()
    claimed = defaultdict(dict)  # co paragraph -> {es paragraph: how well it reads}
    i, j = rows, cols
    while i or j:
        kind, previous_i, previous_j = move[i][j]
        if kind == "match":
            rendered_co.add(i - 1)
            claimed[i - 1][j - 1] = reads_as[i - 1, j - 1]
        i, j = previous_i, previous_j

    # `rendered_co` is deliberately left alone: a co-official paragraph that the Spanish
    # covers only briefly is still covered, and taking it back out of `coverage` here is
    # what would turn a complete rendering into a `partial` one.
    def _length(j):
        return es_runs[j][2] - es_runs[j][1]

    for source, picks in claimed.items():
        allowance = RENDERING_MAX_EXPANSION * (co_runs[source][2] - co_runs[source][1])
        length = sum(_length(j) for j in picks)
        # Weakest first: of everything this paragraph claims, that is the claim the
        # alignment was least sure of, and dropping it returns the paragraph to the
        # record of what was said rather than deleting anything. Ties broken by dropping
        # the LONGEST — within one source paragraph its own length is fixed, so the
        # longest claim is the least proportionate, which is the same question the
        # allowance asks. Without that the choice falls out of dictionary order: a
        # 209-character paragraph claiming both its own 216-character rendering and a
        # 1,276-character neighbour kept the neighbour.
        while picks and length > allowance:
            weakest = min(picks, key=lambda j: (picks[j], -_length(j)))
            length -= _length(weakest)
            del picks[weakest]
    return rendered_co, {j for picks in claimed.values() for j in picks}


def _join(stripped, spans):
    return "\n\n".join(stripped[start:end].strip() for start, end in spans)


def _dominant(runs):
    weights = {}
    for lang, start, end in runs:
        weights[lang] = weights.get(lang, 0) + (end - start)
    return max(weights, key=weights.get) if weights else "es"


# ---- the provisional reading, for a speech nothing places --------------------

def _single_cut(stripped, text, detect, co_lang):
    """The one-boundary reading: cut where it best separates a co-official front from a
    Spanish tail. Kept only for speeches the evidence cannot place, so that they hold the
    same shape they held before rather than losing a block while they wait."""
    spans = sentence_spans(stripped)
    sentences = [stripped[start:end] for start, end in spans]
    langs = [detect(s) if len(s) >= 12 else None for s in sentences]
    total = sum(len(s) for s in sentences)
    k = _boundary(sentences, langs)
    original = stripped[:spans[k - 1][1]] if k else ""
    castellano = stripped[spans[k][0]:] if k < len(spans) else ""
    if not original or len(castellano) / total < _MIN_RATIO:
        return [Block(co_lang, text, True)]
    return [Block(co_lang, original, True), Block("es", castellano, False)]


def _boundary(sentences, langs):
    """The cut maximising co-official characters before it plus Spanish after it."""
    lengths = [len(s) for s in sentences]
    is_co = [lang in CO_LANGS for lang in langs]
    is_es = [lang == "es" for lang in langs]

    co_prefix = 0
    es_suffix = sum(length for length, es in zip(lengths, is_es) if es)
    best_k, best_score = 0, es_suffix
    for k in range(1, len(sentences) + 1):
        length = lengths[k - 1]
        if is_co[k - 1]:
            co_prefix += length
        if is_es[k - 1]:
            es_suffix -= length
        score = co_prefix + es_suffix
        if score > best_score:
            best_score, best_k = score, k
    return best_k
