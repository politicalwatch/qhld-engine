"""How a speech's text divides into blocks, and what those blocks are to each other.

Pure logic — no HTTP, no DB. The language detector is injected as ``detect(str) -> str |
None``; the py3langid adapter that backs it lives in the infrastructure layer.

A block is a **role**, not a slice of the document, and each role holds the *whole*
intervention: ``original`` is the speech as delivered, whatever languages it mixes, and the
second block is the same speech in Spanish when the Diario published a rendering. Anything
that was never translated — a greeting, a citation read aloud, a stretch the speaker gave
in Spanish — has no counterpart, so it stands in both blocks unchanged. The two are
therefore near mirror images, and the only thing that separates them in content is a
translation the Diario left incomplete. Four shapes cover the corpus:

1. co-official **and** a Spanish rendering exists — appended at the end or interleaved
   paragraph by paragraph. Two blocks.
2. mostly co-official, Spanish only as quotations, no rendering. One block.
3. mostly Spanish, co-official only as a greeting or farewell. One block.
4. mostly Spanish, but a passage of it was delivered in a co-official language **and the
   Diario printed a rendering of that passage**. Two blocks, both Spanish: it is a
   Spanish speech, and the rendering is no more spoken than any other interpretation.

The fourth is the awkward one, because it is shape 1 and shape 3 at once — a speech
nobody would call co-official, carrying words nobody said. Read as shape 3 it stores the
rendering as though the speaker had uttered it; read as shape 1 it would rename a Spanish
intervention. So the shapes are tried in the order above, and this one only where no
reading of the document supports a translation of the whole speech.

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
    PARAGRAPH_BREAK,
    co_official_spelling_spans,
    contesting_language,
    paragraph_spans,
    quotation_spans,
    sentence_spans,
    spanish_spelling_spans,
)


class Block(NamedTuple):
    """One block of a speech. ``partial`` marks a Spanish block part of which is not a
    translation at all but the original standing in for one the Diario never printed —
    never set on an ``original`` block, which has nothing to stand in for.

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

# How much of the co-official passage of a mostly-Spanish speech must be rendered before
# the speech is read as carrying a rendering at all.
#
# Deliberately far below COVERAGE_FLOOR, and it is a different question: that one asks
# whether a co-official SPEECH has a translation, this one whether a PASSAGE inside a
# Spanish one does. A speech is not made co-official by the passage, so there is no
# proportion of it that has to be covered — only enough that the reading rests on more
# than a single weak pairing.
#
# The gold set cannot pin this figure and does not pretend to: every value from 0.10 to
# 0.85 scores the same 17/20 there. Corpus-wide the quantity is bimodal — of 190
# single-block Spanish speeches with a co-official run, 182 sit at or above 0.25 and 174
# at or above 0.85, because the passage is usually one or two paragraphs that are either
# rendered or not. So the plateau is 0.15 to 0.47, its upper edge pinned by 774737, whose
# Basque runs on past the paragraph the Diario rendered (0.472). 0.25 sits inside it.
RENDERED_PASSAGE_COVERAGE = 0.25

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
        return _unplaced(stripped, text, detect, co_lang)

    # Two readings of the same document, in order of preference, and the clip decides
    # between them. Refusing an over-claim is a hypothesis about the text; how long the
    # speaker was on their feet is independent evidence about how much of it can have been
    # spoken at all. Where the stricter reading leaves a speech that could not have been
    # delivered, it is the reading that is wrong, not the speech.
    #
    # This is the module's own rule — no one instrument decides — applied to a place the
    # text alone genuinely cannot settle. The similarity of two paragraphs is comparable
    # across language pairs, which is what lets ONE floor serve all of them; the DIFFERENCE
    # between two such scores is not, because the range it varies over is a property of
    # the pair. Measured over 3,034 speeches, the difference the over-claim test turns on
    # is near-symmetric about zero at small magnitudes, so no threshold can tell a decided
    # comparison from a coin toss. The clip can still tell an impossible outcome.
    # The reading comes first and the shapes are tried within it, so a reading that
    # places the document at all is never passed over for a later one. Which matters
    # most where the two disagree about how much was spoken: the lenient reading leaves
    # an over-claim standing, so it counts spoken Spanish as rendering, which shrinks
    # what the speech is measured against and lifts its co-official share above the
    # floor. 726702 and 730547 are the cases — 29% and 31% Basque by the document, read
    # as Basque speeches by the lenient arm and as Spanish ones with a rendered passage
    # by the strict arm, which is what they are.
    for strict in (True, False):
        read = _read(stripped, runs, co_lang, detect, similarity, strict)
        for shape in (_a_translated_speech, _a_rendered_passage, _a_speech_as_spoken):
            placed = shape(read, stripped, text, runs, co_lang, duration, detect)
            if placed is not None:
                return placed

    # Nothing places it. Keep the shape the single-boundary reading gives, so a speech
    # that is probably an ordinary pair does not lose its Spanish block while it waits,
    # and mark it so the adjudication pass can find it.
    return _unplaced(stripped, text, detect, co_lang)


class _Read(NamedTuple):
    """One reading of the document: which runs are the speech, which render it, and the
    three quantities the shape rules are decided on."""

    co_runs: list
    es_runs: list
    rendered_co: set
    delivered_es: set
    coverage: float
    co_chars: int
    spoken: int


def _read(stripped, runs, co_lang, detect, similarity, strict):
    """Read the document once, under one pruning of the alignment's claims.

    ``runs`` is the partition the detector handed down; the one this works from may
    differ, because a paragraph it misread can be offered to the alignment and kept.
    """
    co_runs, es_runs, rendered_co, rendering_es = _partition(
        stripped, [run for run in runs if run[0] in CO_LANGS],
        [run for run in runs if run[0] == "es"],
        co_lang, detect, similarity, strict)
    # Recounted, because the partition may have moved a run across: everything below
    # measures how much of the speech is co-official, and a paragraph the detector
    # misread was being counted on the wrong side of that.
    co_chars = sum(end - start for _, start, end in co_runs)
    rendered = sum(co_runs[j][2] - co_runs[j][1] for j in rendered_co)
    # Everything that is not a rendering was delivered. Built by subtraction rather
    # than by selection, because the as-delivered block is the record of what was said:
    # anything the alignment cannot account for belongs in it, including a greeting the
    # detector misreads. Selecting only what is *provably* delivered loses those.
    delivered_es = {i for i in range(len(es_runs)) if i not in rendering_es}
    return _Read(
        co_runs, es_runs, rendered_co, delivered_es,
        rendered / co_chars if co_chars else 0.0, co_chars,
        co_chars + sum(end - start for i, (_, start, end) in enumerate(es_runs)
                       if i in delivered_es))


def _a_translated_speech(read, stripped, text, runs, co_lang, duration, detect):
    """Shape 1: a co-official speech the Diario published a Spanish rendering of."""
    if not _renders_whole_speech(read.coverage, read.co_chars, read.spoken, duration):
        return None
    return _decided(co_lang, _rendered_blocks(stripped, read, co_lang, co_lang),
                    duration is None, detect)


def _a_rendered_passage(read, stripped, text, runs, co_lang, duration, detect):
    """Shape 4: a Spanish speech, part of which was delivered in a co-official language
    and printed in Spanish as well.

    The blocks are the ones any other rendering gets — everything except the Spanish
    that renders something, and everything except the co-official that something
    rendered — so the passage stands in the record of what was said and its
    interpretation stands beside it, where it cannot be mistaken for a spoken word.

    What differs is only the name: the speech is Spanish and stays Spanish, because a
    passage does not make it otherwise. So the as-delivered block is named for the
    language most of it is in, which leaves ``langs`` to say that the co-official
    passage was spoken at all, and leaves the corpus, the reader and the language filter
    describing the speech as what it is.

    Tried only where no reading supports shape 1, so it needs no co-official share of
    its own: whether the speech is co-official has already been asked and answered.
    """
    if read.coverage < RENDERED_PASSAGE_COVERAGE:
        return None
    # The one thing the text cannot argue with. Subtracting a rendering that was in fact
    # spoken leaves a speech too slow for its own clip, and a speaker who translates
    # themselves aloud is exactly the case this shape must not swallow.
    #
    # It is deliberately NOT also asked whether the document as printed is too long to
    # have been spoken, which would be the stronger evidence: the passage is a tenth of
    # the speech, so printing it twice moves the rate by a c/s or two and leaves it well
    # inside a band four wide. 774737 reads at 16.80 with the rendering in it and would
    # pass such a test. What the clip settles here is the population rather than the
    # speech: over the 219 the corpus offers, the document runs at 15.7 c/s against a
    # monolingual median of 13.3, and taking out exactly what the alignment claims puts
    # them back at 13.5.
    if _fits(read.spoken, duration) is False:
        return None
    spoken_lang = _dominant(runs)
    return _decided(spoken_lang, _rendered_blocks(stripped, read, spoken_lang, co_lang),
                    duration is None, detect)


def _a_speech_as_spoken(read, stripped, text, runs, co_lang, duration, detect):
    """Shapes 2 and 3: the whole document was delivered, in one language or two."""
    if not _all_of_it_was_spoken(read.coverage, read.co_chars, read.spoken, stripped,
                                 duration):
        return None
    # From the runs the DETECTOR read, never the promoted ones. A speech nothing
    # rendered is named by the language it is mostly written in, and relabelling a
    # paragraph on the strength of a dispute — when no rendering of it turned up to
    # settle that dispute — would rename plainly Spanish speeches Catalan.
    dominant = _dominant(runs)
    return _decided(dominant, [Block(dominant, text, True)], duration is None, detect)


def _renders_whole_speech(coverage, co_chars, spoken, duration):
    """Is this a co-official speech with a Spanish rendering of it?"""
    return (coverage >= COVERAGE_FLOOR
            and co_chars / spoken >= CO_SHARE_FLOOR
            and _fits(spoken, duration) is not False)


def _all_of_it_was_spoken(coverage, co_chars, spoken, stripped, duration):
    """Was the whole document delivered — no rendering, just a speech that changed
    language?"""
    if _fits(len(stripped), duration) is False:
        return False
    return coverage < COVERAGE_FLOOR or co_chars / spoken < CO_SHARE_FLOOR


def _fits(chars, duration):
    """Could ``chars`` have been spoken in ``duration`` seconds? ``None`` when there is
    no clip to ask."""
    if not duration:
        return None
    return SPEECH_RATE_MIN <= chars / duration <= SPEECH_RATE_MAX



def _rendered_blocks(stripped, read, lang, co_lang):
    """The two blocks of a speech that has a rendering.

    ``lang`` names the as-delivered block: the co-official language of a speech given in
    one, and Spanish where only a passage of it was.

    **Both blocks hold the whole speech**, and each is built by subtracting from it the
    stretches the other one accounts for. They are exact mirrors:

    - the original block is everything except the Spanish that renders something;
    - the Spanish block is everything except the co-official that something rendered.

    So a stretch with no counterpart on the other side — a greeting nobody translated, a
    citation read aloud, an aside, a closing stretch the speaker delivered in Spanish, a
    switch to Spanish that runs to the end of the speech — appears in **both**. That
    duplication is the point rather than a cost: neither block is a slice of the document,
    each is the whole intervention seen one way, and the only thing that can make them
    asymmetric is a translation the Diario left incomplete.

    Which is what ``partial`` marks, and the mirror sharpens what it means: not that this
    block is missing anything, but that some of what it holds is the original rather than a
    translation of it.

    A quotation on its own paragraph cannot be found by either subtraction, because both
    work in runs: having no language of its own it attaches to the paragraph before it, and
    when that paragraph is accounted for on the other side it leaves along with it. So it
    is added back on both sides, by position.

    A paragraph that spells a co-official language is put back the same way, but on **one
    side only**. It has the same problem — too short to have a language, so it rides
    whichever run absorbed it — and when that run is a rendering it takes a line the
    speaker actually said out of the record. The asymmetry is the point: a word that cannot
    be Spanish proves the line is not purely a rendering, so it belongs in the as-delivered
    block; it says nothing about whether a translation of it exists, so the Spanish block is
    left exactly as the subtraction found it. The rule can therefore only ever ADD to the
    record of what was said.
    """
    quotations = quotation_spans(stripped)
    delivered = [(start, end) for _, start, end in
                 read.co_runs + [run for i, run in enumerate(read.es_runs)
                                 if i in read.delivered_es]]
    delivered += [span for span in quotations if not _within(span, delivered)]
    delivered += [span for span in co_official_spelling_spans(stripped)
                  if not _within(span, delivered)]
    spanish = [(start, end) for _, start, end in read.es_runs] + [
        (start, end) for i, (_, start, end) in enumerate(read.co_runs)
        if i not in read.rendered_co]
    spanish += [span for span in quotations if not _within(span, spanish)]
    # And the mirror of it, on the Spanish side. `_within` is what makes the rule exact
    # rather than generous: a short Spanish line is missing from this list ONLY when the run
    # that absorbed it was a co-official original something rendered. Such a line sits inside
    # the as-delivered stretch, so it was spoken — and delivered Spanish belongs in both
    # blocks. Every other Spanish line is already here and is skipped.
    spanish += [span for span in spanish_spelling_spans(stripped, co_lang)
                if not _within(span, spanish)]
    return [Block(lang, _join(stripped, sorted(delivered)), True),
            Block("es", _join(stripped, sorted(spanish)), False,
                  read.coverage < COVERAGE_COMPLETE)]


def _within(span, spans):
    start, end = span
    return any(s <= start and end <= e for s, e in spans)


def _partition(stripped, co_runs, es_runs, co_lang, detect, similarity, strict):
    """Which runs are the speech and which are the rendering of it, decided rather than
    taken on trust — ``(co runs, es runs, rendered co indices, rendering es indices)``.

    The detector hands down a partition, and for a handful of paragraphs it is wrong in
    one direction only: a co-official paragraph carrying enough Spanish reads as Spanish,
    never the reverse. Filed among the interpreter's paragraphs, nothing can be a
    rendering *of* such a paragraph, so its own rendering has no source to belong to and
    stays in the record of what was said. One speech in the gold set loses five judgements
    to three of them.

    No better reading exists (see ``contesting_language``), so a disputed paragraph is not
    reclassified on the strength of the dispute. It is **offered** to the alignment, and
    kept only if a rendering of it then turns up — which is evidence that bears on the
    question, since a paragraph nobody translated is one nobody had to. An offer nothing
    takes up leaves no trace.

    A run already serving as somebody's rendering is not offered. A translation is not a
    misread original, and taking it away from the source it explains would refuse a healthy
    speech to fix a defect it does not have. So the first alignment is run for that alone,
    and a speech with nothing disputed pays only it.

    Unless the language disputing it is the very one it is supposed to render — and then
    the claim refutes itself, because a Catalan paragraph does not render Catalan. 776200
    is the case: its second paragraph is Catalan, reads as Spanish to the vote (971
    characters against 436) and is claimed as the rendering of the Catalan before it, so
    the spoken Catalan leaves the record of what was said and the paragraph that really
    does render it stays in. Three speeches corpus-wide are decided by this.
    """
    align = lambda co, es: _align_renderings(stripped, co, es, similarity, strict)
    rendered_co, rendering_es = align(co_runs, es_runs)
    disputed = [(co_lang, start, end)
                for index, (_, start, end) in enumerate(es_runs)
                if _worth_offering(stripped, start, end, detect, co_lang,
                                   index in rendering_es)]
    if not disputed:
        return co_runs, es_runs, rendered_co, rendering_es

    offered = sorted(co_runs + disputed, key=lambda run: run[1])
    kept_es = [run for run in es_runs if (co_lang, run[1], run[2]) not in set(disputed)]
    grown_co, grown_es = align(offered, kept_es)
    earned = {offered[index] for index in grown_co}.intersection(disputed)
    if len(earned) == len(disputed):
        return offered, kept_es, grown_co, grown_es
    if not earned:
        return co_runs, es_runs, rendered_co, rendering_es
    # Some were offered and not taken. Those go back, and the alignment is settled on the
    # partition actually kept rather than on the one that was tried.
    final_co = sorted(co_runs + list(earned), key=lambda run: run[1])
    final_es = [run for run in es_runs if (co_lang, run[1], run[2]) not in earned]
    return (final_co, final_es, *align(final_co, final_es))


def _worth_offering(stripped, start, end, detect, co_lang, is_a_rendering):
    """Should this Spanish run be offered to the alignment as a co-official one?

    Any dispute is enough for a run nothing has claimed. For one already serving as a
    rendering the dispute has to name ``co_lang`` itself, which is the only case where
    the claim contradicts what the run is being read as — a paragraph a quarter of which
    reads as Catalan is not the Spanish translation of Catalan. Contests by some *other*
    co-official language are commonly detector noise (a Catalan paragraph draws a few
    Galician sentences), and those must not unpick a translation.
    """
    contests = {contesting_language(paragraph, detect)
                for paragraph in stripped[start:end].split(PARAGRAPH_BREAK)} - {None}
    if not contests:
        return False
    return co_lang in contests if is_a_rendering else True


def _align_renderings(stripped, co_runs, es_runs, similarity, strict=True):
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

    if strict:
        for source, picks in claimed.items():
            _drop_what_reads_worse_together(co_text[source], es_text, picks, similarity)

    rendered_co |= _merged_into_a_neighbour(co_text, es_text, claimed, similarity)
    return rendered_co, {j for picks in claimed.values() for j in picks}


def _drop_what_reads_worse_together(source, es_text, picks, similarity):
    """Drop a claim that does not help this source read as rendered.

    Proportionality refuses a claim for being too long for its source. This refuses one
    for not belonging to it, whatever its length — which is what is left over once the
    alignment has run: a source that found its rendering and took the paragraph beside it
    as well. A speaker carrying the same thought on in Spanish, or repeating a line of it
    after an interruption, both read as the source, because they say what the source says.
    Neither was rendered by anybody, and both are then struck out of the record of what
    was said.

    The test is the one ``_merged_into_a_neighbour`` makes, from the other side: **does
    joining the extra to the strongest claim make the reading worse?** A rendering the
    Diario broke across a paragraph break supplies the half that was missing, so joining
    it improves the match; a paragraph that merely continues the same argument dilutes it,
    and the dilution is the evidence. Over the gold set that separates fifteen claims that
    are no such thing from the two that are.

    Dilution has to be **strict**, like the improvement the merge test asks for, so that
    both rules act only on evidence. Nothing turns on it with a real instrument — two
    cosines over different texts are not equal — but a coarse similarity scores every
    pairing alike, and a rule that dropped on a tie would throw away whatever such a
    measure happened to claim, having been told nothing either way.

    Deliberately no threshold, for the reason the merge test has none: anything the
    traceback paired already scored at or above ``SIMILARITY_FLOOR``, so improving on it
    clears the floor by construction, and there is nothing to re-calibrate when the
    embedding model changes.

    Each extra is weighed against the strongest claim rather than against a join that
    grows, so the answer cannot depend on the order they are considered in.

    **Only this source's ``picks`` shrinks.** ``rendered_co`` is untouched, so ``coverage``
    and ``partial`` cannot move at all; and a Spanish paragraph some OTHER source also
    claims stays a rendering, which is how one paragraph rendering two originals survives
    a source that is only part of its business. What does move is ``spoken``: a dropped
    claim returns to the record of what was said, which is the direction it costs nothing
    to be wrong in.
    """
    if len(picks) < 2:
        return
    # Ties broken on length, as the pruning above breaks them: between two equal readings
    # the shorter claim is the more proportionate one to keep.
    best = max(picks, key=lambda j: (picks[j], -len(es_text[j])))
    for other in sorted(picks):
        if other == best:
            continue
        together = PARAGRAPH_BREAK.join(es_text[j] for j in sorted((best, other)))
        if similarity(source, together) < picks[best]:
            del picks[other]


def _merged_into_a_neighbour(co_text, es_text, claimed, similarity):
    """The sources the Diario rendered *together with* the paragraph next to them.

    Whitespace is not a translated quantity: the interpreter re-paragraphs freely, so two
    co-official paragraphs may be published as one Spanish paragraph (and one as two). The
    alignment already allows either side to consume several of the other's paragraphs — what
    defeats it is the SCORE, which compares a whole paragraph against a whole paragraph
    however much of the other one is about something else. 726204 is the case: one Spanish
    paragraph renders the speaker's opening line and the paragraph after it, so the opening
    scores 0.588 against it while the longer neighbour scores 0.895. Below the floor, so the
    opening looks untranslated, and the Spanish block then carries the Catalan original
    beside the translation of it.

    The test is whether the merge **reads better than the neighbour did alone**. If that
    Spanish paragraph really renders both, supplying the half that was missing must improve
    the match; if it renders only the neighbour, adding foreign text dilutes it. On 726204
    the true merge goes 0.895 -> 0.931 while a wrong one — the neighbour plus the paragraph
    after it, which has a rendering of its own — goes 0.895 -> 0.783.

    Deliberately no threshold. Anything the traceback paired already scored at or above
    ``SIMILARITY_FLOOR``, so an improvement on it clears the floor too, and there is nothing
    here to calibrate or to re-calibrate when the embedding model changes. A length band was
    measured instead and rejected: it needed two constants, its lower bound rested on a
    single counter-example, it could not tell the two candidate neighbours apart, and the
    sweep found no plateau to anchor a value to.

    Only ``rendered_co`` grows. Which Spanish runs are renderings is decided by the traceback
    and left alone, so nothing here can take a spoken word out of the record — the
    as-delivered block holds every co-official run whatever this returns.
    """
    rescued = set()
    for source in range(len(co_text)):
        if source in claimed:
            continue
        for neighbour in (source - 1, source + 1):
            if neighbour not in claimed:
                continue
            first, second = min(source, neighbour), max(source, neighbour)
            # Joined, never sliced out of the document: the two may have Spanish between
            # them, and slicing would pull that into the hypothesis.
            merged = co_text[first] + PARAGRAPH_BREAK + co_text[second]
            if any(similarity(merged, es_text[rendering]) > alone
                   for rendering, alone in claimed[neighbour].items()):
                rescued.add(source)
                break
    return rescued


def _join(stripped, spans):
    return "\n\n".join(stripped[start:end].strip() for start, end in spans)


def _dominant(runs):
    weights = {}
    for lang, start, end in runs:
        weights[lang] = weights.get(lang, 0) + (end - start)
    return max(weights, key=weights.get) if weights else "es"


# ---- the provisional reading, for a speech nothing places --------------------

def _unplaced(stripped, text, detect, co_lang):
    """The verdict for a speech the evidence cannot place.

    A speech is named by the language of the block that holds what was delivered, which
    is the invariant the whole corpus already satisfies — so the name follows the cut
    rather than being decided ahead of it. That distinction only shows when the cut
    finds no real second block: the document is then one speech in one language, and
    calling it by whatever co-official paragraph it happens to contain is how a Spanish
    speech comes to be filed as Galician on the strength of a single quotation.
    """
    blocks = _single_cut(stripped, text, detect, co_lang)
    return _decided(blocks[0].lang, blocks, True, detect)


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
        # No second block, so there is no co-official front to name: what is left is
        # the whole document, and it belongs to whichever language most of it is in.
        # The sibling branch above already reads an unsplittable speech this way.
        return [Block(_dominant([(lang, start, end)
                                 for (start, end), lang in zip(spans, langs) if lang]),
                      text, True)]
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
