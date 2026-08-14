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

# How much of a paragraph the vote called Spanish must read as one co-official language
# before that reading counts as disputed rather than settled.
#
# It decides nothing. It marks a paragraph as worth a second look, and the alignment then
# settles it on whether a rendering of that paragraph actually turns up — which is the
# evidence that bears on the question, since a paragraph nobody translated is one nobody
# had to. That division of labour is what makes the figure usable at all: reading these
# paragraphs correctly is a problem no configuration of the detector has solved, but every
# one of them still reads them as NEARLY co-official, and that much is worth keeping.
#
# Swept over the bitext gold set and 3,034 captured speeches. Plateau 0.25-0.28, with both
# edges pinned by real speeches rather than by a preference:
#
#   below ~0.23  750542's opening paragraph (1,646 characters, 0.171 Catalan) is offered,
#                finds a partner among the speech's OWN Spanish, and a plainly Spanish
#                intervention is refused and relabelled Catalan. 749862 the same.
#   above 0.287  771953's "Però, escolti'm, dubtes interpretatius, garrotada" stops being
#                offered, and the speech this whole change exists to fix reverts.
#
# Below the plateau the promotion is more generous and finds more renderings (recall 0.924
# against 0.907) at the cost of two false positives — rejected because a false positive
# takes spoken words out of the record, which is the error that cannot be seen once made.
CONTESTED_SHARE = 0.25

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


# Words that exist in a co-official language and NOT in Spanish, accent-folded and lowercase.
# Curated editorial data, like `qhld_ai`'s group-alias and stop-entity lists, kept as a
# literal because nothing in this layer may read a file.
#
# THE ADMISSION RULE, and it is the point of the list rather than a note about it: an entry
# must be part of a **thanks, greeting or farewell formula, or a form of address**. Nothing
# else, ever. Two admissions are refused by that rule and were removed after measuring them:
# `vull` ("I want") and `acabant` ("finishing") each rescued exactly ONE speech, which is
# what patching a corpus one bug at a time looks like. Without the rule the list grows by a
# word per failing speech and becomes an artifact of one legislature; with it the list is
# closed, because a chamber has only so many ways to say thank you and to address a chair.
# The cost is measured and accepted: phrasings outside the formulas are not rescued — 3 of
# the 45 as-delivered cases and 11 of the 42 Spanish-side ones.
#
# Entries must be impossible in Spanish, which accent folding never breaks: `señor`/`señora`
# fold to `senor`/`senora` and stay distinct from Catalan `senyor`/`senyora`. Refused for
# that reason: `tarda` (Spanish third person of `tardar`), `dia` (Spanish `día` folds onto
# it), `presidenta` and `acabo` (identical in Spanish), `bo` and `ez` (two letters match too
# much to be evidence of anything).
_CO_OFFICIAL_ONLY = frozenset({
    # thanks, greetings and farewells
    "eskerrik", "asko", "mila", "esker", "milesker", "agur", "arratsalde", "egun",  # eu
    "gracies", "moltes", "merces", "bona", "bon", "adeu",                          # ca
    "grazas", "moitas", "moito", "moitisimo", "obrigado", "obrigada", "boas",       # gl
    # forms of address, the other half of the same formulas
    "jauna", "andrea", "anderea", "presidentea", "guztioi",                         # eu
    "president", "senyor", "senyora", "senyores", "senyories", "tothom",            # ca
    "prezado",                                                                      # gl
})


def _folded_tokens(text):
    """Lowercase alphabetic tokens with accents removed.

    Folded because the Diario is inconsistent about accents and a lexicon should not have
    to carry both spellings. **Do not reuse ``_TOKEN`` here**: it is ``[a-z0-9]+``, so it
    splits ``"gràcies"`` into ``"gr"`` and ``"cies"`` and the lexicon silently never
    matches. Folding first is what makes a plain ``[a-z]+`` correct.

    Folding is safe for this test because it never collides a Spanish courtesy word with a
    co-official one: ``"señor"`` folds to ``senor`` and stays distinct from ``senyor``.
    """
    folded = "".join(char for char in unicodedata.normalize("NFKD", text.lower())
                     if not unicodedata.combining(char))
    return set(re.findall(r"[a-z]+", folded))


def carries_co_official_spelling(paragraph):
    """Does this paragraph contain a word that exists in a co-official language and not in
    Spanish — so that it cannot be purely a Spanish rendering?

    Only asked of a paragraph too short to have a language of its own, which is where the
    detector is least reliable and where no other evidence exists. Such a paragraph is
    absorbed into a neighbouring run, and if that run turns out to be a rendering it is
    subtracted along with it — so ``"Eskerrik asko."`` printed between two Spanish
    paragraphs leaves the record of what was said. Spelling is the only signal that reaches
    these lines: the sentence vote cannot, by construction.

    **The reading of the paragraph is deliberately NOT consulted.** Keying on that is the
    refuted approach — ``"Grazas, señor presidente."`` reads as Spanish however much of it
    the detector is given, which is the very reason ``MIN_VOTING_CHARS`` exists.

    The lexicon is a closed class and deliberately incomplete. Missing a word only means a
    paragraph is not rescued, which is the state of affairs today; a wrong word would put a
    rendering back into the as-delivered block, so entries are limited to ones that cannot
    be Spanish at all.
    """
    return bool(_folded_tokens(_votable(paragraph)) & _CO_OFFICIAL_ONLY)


def co_official_spelling_spans(text):
    """The ``(start, end)`` span of every paragraph too short to have a language of its own
    that nonetheless spells a co-official language.

    Same shape as ``quotation_spans`` and for the same reason: a paragraph neither
    subtraction can find, which has to be put back by position.
    """
    spans = []
    position = 0
    for paragraph in text.split(PARAGRAPH_BREAK):
        end = position + len(paragraph)
        if (len(_votable(paragraph)) < MIN_VOTING_CHARS
                and carries_co_official_spelling(paragraph)):
            spans.append((position, end))
        position = end + len(PARAGRAPH_BREAK)
    return spans


# The mirror of the set above: closed-class courtesy words that exist in Spanish and NOT in
# any co-official language, accent-folded. Same admission rule — an entry must be impossible
# in ca/eu/gl, so `gracias` is in and `presidente` is in (Catalan has `president`) while
# nothing shared is. The two sets are disjoint by construction; `test_the_two_lexicons_are_
# disjoint` pins it, because a word in both would make each rule silently ignore it.
# The mirror side, and it needs a different shape. "This word is Spanish" is NOT evidence
# that a line is not co-official, because Spanish shares most of its courtesy vocabulary with
# Galician and some with Catalan. A flat list of "Spanish-only" words pushed ten co-official
# lines into the Spanish block — and SEVEN of the ten were one word, `acabo`, which is spelled
# identically in Catalan and Galician and so says nothing at all.
#
# So the table records **how each language says the word**, and the rule derives from it: a
# word is evidence only where the two spellings DIFFER. `acabo` is then excluded by the rule
# rather than by a judgement call, `presidente` counts against Catalan (`president`) but not
# against Galician, and `nada` counts against Catalan (`res`) but not Galician. Every row is a
# fact about a language that can be checked in a dictionary, not an observation about this
# corpus — which is what stops it drifting per legislature.
#
# Same admission rule as above: thanks/greeting/farewell formulas and forms of address only.
# A language absent from a row has no cognate, so the word counts against it (nothing in
# Basque resembles any of these).
_SPANISH_COGNATES = {
    # thanks, greetings and farewells
    "gracias":    {"ca": "gracies", "gl": "grazas"},
    "muchas":     {"ca": "moltes", "gl": "moitas"},
    "muchisimas": {"ca": "moltissimes", "gl": "moitisimas"},
    "buenas":     {"ca": "bones", "gl": "boas"},
    "buenos":     {"ca": "bons", "gl": "bos"},
    "dias":       {"ca": "dies", "gl": "dias"},
    "salud":      {"ca": "salut", "gl": "saude"},
    "nada":       {"ca": "res", "gl": "nada"},
    # forms of address
    "senor":      {"ca": "senyor", "gl": "senor"},
    "senora":     {"ca": "senyora", "gl": "senora"},
    "senores":    {"ca": "senyors", "gl": "senores"},
    "senorias":   {"ca": "senyories", "gl": "senorias"},
    "presidente": {"ca": "president", "gl": "presidente"},
    "presidenta": {"ca": "presidenta", "gl": "presidenta"},
    "ministro":   {"ca": "ministre", "gl": "ministro"},
    "ministra":   {"ca": "ministra", "gl": "ministra"},
    "diputados":  {"ca": "diputats", "gl": "deputados"},
    "diputadas":  {"ca": "diputades", "gl": "deputadas"},
    # kept as a row precisely BECAUSE it discriminates nowhere: it caused seven of the ten
    # false positives, and the rule now refuses it without anyone having to remember why.
    "acabo":      {"ca": "acabo", "gl": "acabo"},
}


def spanish_evidence(co_lang):
    """The Spanish words that are evidence against ``co_lang`` — those it spells otherwise."""
    return frozenset(word for word, forms in _SPANISH_COGNATES.items()
                     if forms.get(co_lang, "") != word)


def carries_spanish_spelling(paragraph, co_lang):
    """Does this paragraph spell Spanish in a way ``co_lang`` does not — so that it cannot be
    part of a co-official original?

    The mirror of ``carries_co_official_spelling``, for the opposite loss. A short Spanish
    line absorbed into a co-official run is subtracted from the **Spanish** block when that
    run turns out to have been rendered, so delivered Spanish goes missing from the block
    that is supposed to hold every Spanish word of the speech.

    A paragraph carrying a co-official word too is refused rather than guessed at: it may be
    a co-official courtesy line whose translation exists, and adding that to the Spanish
    block would duplicate a line the Diario already rendered there. This is the undecidable
    case, and refusing it is what keeps the rule one-sided.
    """
    tokens = _folded_tokens(_votable(paragraph))
    return (bool(tokens & spanish_evidence(co_lang))
            and not (tokens & _CO_OFFICIAL_ONLY))


def spanish_spelling_spans(text, co_lang):
    """The ``(start, end)`` span of every paragraph too short to have a language of its own
    that spells Spanish in a way ``co_lang`` does not."""
    spans = []
    position = 0
    for paragraph in text.split(PARAGRAPH_BREAK):
        end = position + len(paragraph)
        if (len(_votable(paragraph)) < MIN_VOTING_CHARS
                and carries_spanish_spelling(paragraph, co_lang)):
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
    weights = language_votes(paragraph, detect)
    if not weights:
        return None
    return max(weights, key=weights.get)


def language_votes(paragraph, detect):
    """``{language: characters}`` — how this paragraph's sentences voted.

    The distribution rather than the winner alone, because how close the vote was is
    evidence in its own right: see ``contesting_language``.
    """
    body = _votable(paragraph)
    if len(body) < MIN_VOTING_CHARS:
        return {}
    weights = {}
    for start, end in sentence_spans(body):
        sentence = body[start:end]
        lang = detect(sentence) if len(sentence) >= MIN_DETECTABLE_CHARS else None
        if lang:
            weights[lang] = weights.get(lang, 0) + len(sentence)
    return weights


def contesting_language(paragraph, detect):
    """The co-official language disputing a paragraph the vote called Spanish, or
    ``None`` where the reading was not disputed.

    A speaker may argue in Catalan about a *partido de Estado*, quote a Spanish minister
    and hang a joke about *tres huevos duros* on it. The printed paragraph is then Catalan
    carrying enough Spanish to read as Spanish — and filed among the interpreter's
    paragraphs, where nothing can be a rendering *of* it. Its own Spanish rendering,
    printed further down, finds nobody to belong to and stays in the record of what was
    said, as though the speaker had delivered both.

    Reading such a paragraph correctly is not on offer: every configuration of the
    detector tried calls it Spanish, sentence by sentence or whole, and restricting the
    candidate languages does not help. But every one of them also calls it *nearly*
    Catalan, and that is the part worth keeping. So this reports a dispute and stops
    there, leaving the alignment to decide on evidence of its own.

    Only the strongest single claimant counts, never their sum: a Catalan paragraph
    commonly draws a few Galician sentences as well, and adding that noise in would let
    two weak readings pass for one strong one.
    """
    weights = language_votes(paragraph, detect)
    if not weights or max(weights, key=weights.get) != "es":
        return None
    contenders = {lang: weight for lang, weight in weights.items() if lang in CO_LANGS}
    if not contenders:
        return None
    strongest = max(contenders, key=contenders.get)
    if contenders[strongest] / sum(weights.values()) < CONTESTED_SHARE:
        return None
    return strongest


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
