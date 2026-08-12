"""The language runs of a speech, and whether one stretch renders another.

Pure text logic — no HTTP, no DB. The language detector is injected as a callable,
``detect(str) -> str | None``, so this module stays trivially testable and never loads
py3langid; the adapter that backs it lives in the infrastructure layer.

**The unit is the paragraph, not the sentence.** The Diario de Sesiones translates a
co-official speech paragraph by paragraph, and a paragraph may embed a quotation in
another language — read aloud in that language, as part of that paragraph. Deciding a
language sentence by sentence therefore shatters a Catalan paragraph around its Spanish
citation and scatters both halves into the wrong runs. Measured on one speech, that lost
roughly a quarter of its Catalan and turned four clean paragraph pairs into nine runs.

Which is also why a quotation does not get a vote on the language of the paragraph
carrying it: it is the one span reliably written in a language other than the speech's.
A stenographer's annotation gets no vote for the same reason and more strongly: it is not
the speaker's voice at all, and it is written in Spanish whatever language the speech was
delivered in, so it is the one span guaranteed to vote wrong.
"""

import re
import unicodedata
from collections import Counter


# Co-official languages of Spain that appear (alongside Spanish) in the Diario.
CO_LANGS = frozenset({"ca", "eu", "gl"})

PARAGRAPH_BREAK = "\n\n"

# Below this length the detector is too unreliable to trust, so a short sentence
# contributes no vote — it still occupies its place in the paragraph.
MIN_DETECTABLE_CHARS = 12

# And below this, a whole paragraph gets no vote either. A courtesy line is the case
# that forces it: "Grazas, señora presidenta." reads as Spanish, because "señora
# presidenta" outweighs the one Galician word, and it keeps reading as Spanish however
# much of the sentence the detector is given. Left to stand as its own run, a Galician
# speech opens with a Spanish paragraph and a Spanish rendering closes with a Galician
# one. Joining its neighbour puts each where it was said instead.
MIN_VOTING_CHARS = 60

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?»])\s+")

# The Diario quotes with angle quotes, and occasionally with curly doubles. Anything
# unbalanced is left alone rather than swallowing the rest of the paragraph.
_QUOTED = re.compile(r"«[^»]*»|“[^”]*”")

# The Diario's parenthesized stage directions — "(Aplausos)", "(Rumores.―El señor Tellado
# Filgueira: Ábalos…)". Deliberately a second copy of qhld-ai's `_ANNOTATION_RE` (see
# `qhld_ai.domain.annotations`, which owns the fuller treatment: stripping them before
# mention NER, and mining the interruptions out of them). Copied rather than imported
# because that module reaches for the persistence models, and nothing in this layer may.
_ANNOTATED = re.compile(r"\([^()]*\)")

# A token short enough to be shared by chance carries no evidence that one text renders
# another; digits do, at any length, because a translator copies them.
_MIN_TOKEN_CHARS = 4
_TOKEN = re.compile(r"[a-z0-9]+")


def _votable(paragraph):
    """``paragraph`` with everything that has no say in its language taken out: what the
    stenographer noted, and what the speaker quoted.

    Only the reading is taken over this — the paragraph itself is never altered. An
    annotation is part of the printed record and stays in whatever block carries it, which
    is the whole distinction: it is evidence about the sitting, not about the language.
    """
    return _QUOTED.sub(" ", _ANNOTATED.sub(" ", paragraph)).strip()


def sentence_spans(text):
    """The ``(start, end)`` span of each sentence, inter-sentence whitespace excluded."""
    spans = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def paragraph_spans(text, detect):
    """``[(lang, start, end)]`` — one entry per paragraph that has a language.

    The spans **partition** ``text``: every character belongs to exactly one, separators
    included, so a caller can measure how much of a speech is in each language without
    losing anything between them. A slice therefore carries the paragraph break that
    follows it; callers assembling text from spans should strip.

    A paragraph too short to read gets no run of its own, and **which neighbour it joins
    is decided by the same detection, used only to pick a side.** That reading is too
    weak to found a run on — a courtesy line reads as Spanish whatever language it is in
    — but it is good enough to choose between two candidates, and the choice matters in
    both directions: an interpretation opens with *"Gracias, presidente."* and belongs
    with the Spanish that follows, while *"Besterik ez. Eskerrik asko."* closes the
    Basque immediately before it. Attaching every short paragraph to one side or the
    other misplaces one of those two cases.

    A paragraph that is nothing but a quotation has no reading at all, and joins what
    precedes it: a quoted paragraph follows the sentence that announces it
    ("...se puede leer lo siguiente:"), so it continues what it interrupts.

    **No paragraph may join backwards across one that declined to.** Courtesy lines come
    in twos and threes and their weak readings need not agree: 767786 closes its Galician
    with *"Moito obrigado."* and opens its Spanish with *"Gracias, señora presidenta."*
    then *"Muy buenas tardes."*, of which only the second reads as Galician. Each choice
    was defensible alone, and taken in order the last one reached back over the one before
    it and took it along, so both Spanish greetings ended up inside the Galician run with
    no way out of it. Spans are contiguous, so the two choices cannot both hold; the
    earlier paragraph keeps its own, being the one nearer the run it chose.

    Unmerged on purpose. Consecutive paragraphs of one language are a single *run* for
    describing a speech's shape, but they are separate units for deciding what renders
    what: a translation and the Spanish the speaker then went on to deliver are adjacent
    paragraphs, and merging them hides the boundary that separates them.
    """
    paragraphs = []
    position = 0
    for paragraph in text.split(PARAGRAPH_BREAK):
        end = position + len(paragraph)
        paragraphs.append((paragraph, position, end))
        position = end + len(PARAGRAPH_BREAK)

    strong = [paragraph_language(body, detect) for body, _, _ in paragraphs]
    if not any(strong):
        return []

    runs = []
    pending = 0  # start of the text not yet attributed to a run
    waiting = False  # something already pending chose the run that comes next
    for index, ((body, _, end), lang) in enumerate(zip(paragraphs, strong)):
        if lang is None:
            # Short-circuited on purpose: once a paragraph ahead is waiting for the next
            # run, this one's own reading cannot send it backwards over that paragraph,
            # so there is nothing left for the detector to decide.
            if waiting or _joins_what_follows(body, detect, strong, index):
                waiting = True
                continue  # stays pending; the next run will begin before it
            if runs:
                runs[-1][2] = end
                pending = end + len(PARAGRAPH_BREAK)
            continue
        runs.append([lang, pending, end])
        pending = end + len(PARAGRAPH_BREAK)
        waiting = False

    for earlier, later in zip(runs, runs[1:]):
        earlier[2] = later[1]
    runs[-1][2] = len(text)
    runs[0][1] = 0
    return [(lang, start, end) for lang, start, end in runs]


def _joins_what_follows(body, detect, strong, index):
    """Should an unreadable paragraph attach to the run after it rather than before?

    Its own reading decides, when it has one and one of its neighbours matches. With no
    reading — a paragraph that is only a quotation, or only an annotation — it stays with
    what precedes it.

    Read from the same text the strong vote is taken over, and that matters: a paragraph
    denied a vote because most of it is an annotation must not then be placed on the
    strength of that very annotation.
    """
    stripped = _votable(body)
    if len(stripped) < MIN_DETECTABLE_CHARS:
        return False
    weak = detect(stripped)
    if weak is None:
        return False
    before = next((lang for lang in reversed(strong[:index]) if lang), None)
    after = next((lang for lang in strong[index + 1:] if lang), None)
    if weak == before:
        return False
    if weak == after:
        return True
    # It reads as neither. An opening is far more common than a closing at a boundary,
    # and a leading paragraph has nothing before it to join.
    return before is None or after is not None


def paragraph_runs(text, detect):
    """``paragraph_spans`` with same-language neighbours merged — the shape of a speech
    (``CSCS``…) rather than its units."""
    runs = []
    for lang, start, end in paragraph_spans(text, detect):
        if runs and runs[-1][0] == lang:
            runs[-1][2] = end
        else:
            runs.append([lang, start, end])
    return [(lang, start, end) for lang, start, end in runs]


def is_quotation(paragraph):
    """Is this paragraph nothing but a quotation?

    Such a paragraph is read aloud once, in whatever language it is written in, and
    printed once — so it cannot be a rendering of anything, a rendering needing a source.
    The Diario may nonetheless print it inside the Spanish stretch, after the rendering of
    the sentence that announces it, which is where it gets mistaken for one.

    Both halves of the test are load-bearing. Asking only whether the remainder is too
    short to read calls ``"Gracias."`` a quotation, and that closing line really is a
    rendering of the ``"Moito obrigado."`` before it.

    The remainder is what the paragraph itself says, so applause minuted after a quotation
    does not stop it being one — the same reading of the paragraph that decides its
    language.
    """
    if not _QUOTED.search(paragraph):
        return False
    return len(_votable(paragraph)) < MIN_DETECTABLE_CHARS


def quotation_spans(text):
    """The ``(start, end)`` span of every paragraph that is nothing but a quotation."""
    spans = []
    position = 0
    for paragraph in text.split(PARAGRAPH_BREAK):
        end = position + len(paragraph)
        if is_quotation(paragraph):
            spans.append((position, end))
        position = end + len(PARAGRAPH_BREAK)
    return spans


def paragraph_language(paragraph, detect):
    """The language holding most of ``paragraph``, ignoring what it quotes and what the
    stenographer noted in it.

    ``None`` when nothing in it is long enough to read, which includes a paragraph that
    is only a quotation, and a courtesy line that reaches the voting length only because
    of the applause minuted after it (741705's closing *"Gracias. (Aplausos de las señoras
    y los señores diputados…)"* is 119 characters, of which 8 are the speaker's).
    """
    body = _votable(paragraph)
    if len(body) < MIN_VOTING_CHARS:
        return None
    weights = {}
    for start, end in sentence_spans(body):
        sentence = body[start:end]
        lang = detect(sentence) if len(sentence) >= MIN_DETECTABLE_CHARS else None
        if lang:
            weights[lang] = weights.get(lang, 0) + len(sentence)
    if not weights:
        return None
    return max(weights, key=weights.get)


def renders(source, candidate):
    """How much of ``source`` reappears in ``candidate``, 0.0-1.0.

    A rendering keeps what a translator cannot translate — figures, names, the
    institutions and places being argued about — and Spanish shares a great deal of
    ordinary vocabulary with Catalan and Galician besides. So the fraction of one text's
    tokens found in the other separates a rendering from an unrelated stretch of the same
    debate, for those two.

    It does **not** work for Basque, which shares almost nothing with Spanish: a full
    interpretation scores near zero. Callers must not read a low score as "no rendering
    exists" without knowing which languages they are comparing.
    """
    return containment(token_counts(source), token_counts(candidate))


def token_counts(text):
    """The tokens ``renders`` compares, counted. Exposed so a caller comparing many
    pairs of the same texts counts each of them once."""
    return Counter(_tokens(text))


def containment(wanted, found):
    """``renders`` over already-counted tokens."""
    total = sum(wanted.values())
    if not total:
        return 0.0
    return sum(min(count, found[token]) for token, count in wanted.items()) / total


def _tokens(text):
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return [t for t in _TOKEN.findall(folded)
            if len(t) >= _MIN_TOKEN_CHARS or t.isdigit()]
