"""Unit tests for language runs — the detector is a fake keyed on marker words, so
these are deterministic and never load py3langid.
"""

import pytest

from qhld_engine.domain.speeches.language_runs import (
    CONTESTED_SHARE,
    MIN_VOTING_CHARS,
    carries_co_official_spelling,
    carries_spanish_spelling,
    co_official_spelling_spans,
    contesting_language,
    is_quotation,
    language_votes,
    paragraph_language,
    paragraph_runs,
    quotation_spans,
    renders,
    sentence_spans,
    spanish_spelling_spans,
)

pytestmark = pytest.mark.unit

# Catalan-only fragments, none a substring of its Spanish cognate.
_CA_MARKERS = ("gràcies", "senyories", "aquesta", "veritat", "nosaltres", "però",
               "això", "perquè", "ciència")


def _fake_detect(text):
    low = text.lower()
    return "ca" if any(m in low for m in _CA_MARKERS) else "es"


CA_PARAGRAPH = ("Gràcies, senyories. Això és la veritat que nosaltres volem defensar "
                "aquí aquesta tarda, perquè ningú altre ho farà per nosaltres.")
ES_PARAGRAPH = ("Muchas gracias, señorías. Esto es lo que queremos defender aquí esta "
                "tarde, porque nadie más lo va a hacer en nuestro lugar.")


def test_a_paragraph_takes_the_language_holding_most_of_it():
    assert paragraph_language(CA_PARAGRAPH, _fake_detect) == "ca"


def test_a_paragraph_too_short_to_read_has_no_language():
    # A courtesy line is exactly where the detector misreads, so it gets no vote.
    assert paragraph_language("Grazas, señora presidenta.", _fake_detect) is None


def test_a_quotation_does_not_decide_the_language_of_its_paragraph():
    # The Diario prints a Spanish citation inside a Catalan paragraph, read aloud in
    # Spanish. Counting it would outvote the Catalan around it and lose the paragraph.
    paragraph = (
        "Això és la veritat, senyories, i nosaltres ho volem dir aquí ben clar. "
        "«Los sepultureros más eficaces de un imperio suelen ser los mismos "
        "imperialistas, y no hay nada que discutir sobre ello en esta Cámara»."
    )
    assert paragraph_language(paragraph, _fake_detect) == "ca"


def test_a_paragraph_that_is_only_a_quotation_has_no_language():
    assert paragraph_language(
        "«A pesar de ello, nuestros tradicionales enemigos nos acusan de crueles»",
        _fake_detect) is None


def test_a_paragraph_that_is_only_a_quotation_is_recognised_as_one():
    assert is_quotation(
        "«A pesar de ello, nuestros tradicionales enemigos nos acusan de crueles».")


def test_a_short_ordinary_paragraph_is_not_a_quotation():
    # The half of the test that is easy to leave out, and the one that matters: a closing
    # courtesy line is also too short to read, and it really can be a rendering of the
    # line before it.
    assert not is_quotation("Gracias.")
    assert not is_quotation("Moito obrigado.")


def test_a_paragraph_that_merely_contains_a_quotation_is_not_one():
    assert not is_quotation(
        "Això és la veritat, senyories, i ho volem dir ben clar aquí aquesta tarda. "
        "«Los sepultureros más eficaces de un imperio suelen ser los imperialistas»."
    )


def test_quotation_spans_locate_the_quoted_paragraphs():
    quotation = "«A pesar de ello, nuestros enemigos nos acusan de crueles»."
    text = f"{CA_PARAGRAPH}\n\n{quotation}\n\n{ES_PARAGRAPH}"
    assert [text[start:end] for start, end in quotation_spans(text)] == [quotation]


def test_consecutive_paragraphs_of_one_language_are_one_run():
    text = f"{CA_PARAGRAPH}\n\n{CA_PARAGRAPH}\n\n{ES_PARAGRAPH}"
    runs = paragraph_runs(text, _fake_detect)

    assert [lang for lang, _, _ in runs] == ["ca", "es"]
    assert text[runs[0][1]:runs[0][2]].startswith("Gràcies")
    assert text[runs[1][1]:runs[1][2]].startswith("Muchas gracias")


def test_every_character_belongs_to_exactly_one_run():
    # A quotation-only paragraph has no language of its own; it must still be carried,
    # or the arithmetic that compares text length against the clip loses it.
    text = f"{CA_PARAGRAPH}\n\n«Una cita en castellà»\n\n{ES_PARAGRAPH}"
    runs = paragraph_runs(text, _fake_detect)

    covered = sum(end - start for _, start, end in runs)
    assert covered == len(text)
    assert runs[0][1] == 0 and runs[-1][2] == len(text)


def test_a_leading_unreadable_paragraph_joins_the_run_after_it():
    text = f"«Cita».\n\n{CA_PARAGRAPH}"
    runs = paragraph_runs(text, _fake_detect)

    assert [lang for lang, _, _ in runs] == ["ca"]
    assert runs[0][1] == 0


def test_a_courtesy_line_cannot_be_pulled_back_over_one_that_chose_to_go_forward():
    # 767786's boundary: the Galician closes with "Moito obrigado." and the Spanish opens
    # with two courtesy lines, of which only the SECOND reads as Galician. Each choice is
    # defensible alone — and taken in order the last one used to reach back over the one
    # before it and take it into the Galician run, so both Spanish greetings ended up in
    # the record of what was said with no way out of it.
    text = "\n\n".join([CA_PARAGRAPH, "Moltes gràcies.", "Muchas gracias, presidenta.",
                        "Gràcies i bona tarda.", ES_PARAGRAPH])
    runs = paragraph_runs(text, _fake_detect)

    assert [lang for lang, _, _ in runs] == ["ca", "es"]
    # the Catalan run ends after its own closing line...
    assert text[runs[0][1]:runs[0][2]].strip().endswith("Moltes gràcies.")
    # ...and BOTH greetings are on the Spanish side, including the one reading as Catalan
    assert text[runs[1][1]:runs[1][2]].strip().startswith("Muchas gracias, presidenta.")
    assert "Gràcies i bona tarda." in text[runs[1][1]:runs[1][2]]


def test_a_closing_line_is_not_lengthened_into_a_vote_by_the_applause_after_it():
    # 741705's closing "Gracias." is 8 characters of speaker and 111 of stenographer. Left
    # to count, the annotation carries it over the voting length and it founds a run of its
    # own — which is how a courtesy line that IS a rendering became impossible to pair,
    # its 8-character original having no run of its own to be paired with.
    applauded = ("Gracias. (Aplausos de las señoras y los señores diputados del Grupo "
                 "Parlamentario Plurinacional SUMAR, puestos en pie).")
    assert len(applauded) > MIN_VOTING_CHARS
    assert paragraph_language(applauded, _fake_detect) is None


def test_an_annotation_does_not_decide_the_language_of_its_paragraph():
    # The stenographer writes in Spanish whatever language the speech is in, so an
    # annotation is the one span guaranteed to vote for the wrong one.
    paragraph = (f"{CA_PARAGRAPH} (Rumores.―El señor presidente pide silencio y reclama "
                 "a las señoras y los señores diputados que respeten el turno de "
                 "palabra del orador que está en la tribuna).")
    assert paragraph_language(paragraph, _fake_detect) == "ca"


def test_empty_text_has_no_runs():
    assert paragraph_runs("", _fake_detect) == []
    assert paragraph_runs("   ", _fake_detect) == []


# A Catalan paragraph the vote calls Spanish: the speaker is arguing about a Spanish
# phrase and quoting it back, so only the closing sentence is unmistakably his own
# language. This is the shape no detector configuration reads correctly.
CODE_MIXED = ("Y esto, señorías, no lo hace un partido de Estado, se lo digo yo. "
              "Se lo digo con todo el cariño del mundo, de verdad se lo digo. "
              "Però nosaltres això no ho farem mai, senyories, mai de la vida.")


def test_a_paragraph_the_vote_calls_spanish_can_still_be_reported_as_disputed():
    assert paragraph_language(CODE_MIXED, _fake_detect) == "es"
    assert contesting_language(CODE_MIXED, _fake_detect) == "ca"


def test_an_unmistakably_spanish_paragraph_is_not_disputed():
    spanish = ("Señorías, comparezco hoy para hablar del cierre de la fábrica y de "
               "sus efectos sobre el empleo en la comarca durante los próximos años.")

    assert paragraph_language(spanish, _fake_detect) == "es"
    assert contesting_language(spanish, _fake_detect) is None


def test_a_paragraph_already_read_as_co_official_is_not_disputed():
    # The dispute only ever runs one way: a co-official paragraph carrying Spanish reads
    # as Spanish, never the reverse. Reporting the mirror case would offer the alignment
    # paragraphs it has no business reconsidering.
    assert contesting_language(CA_PARAGRAPH, _fake_detect) is None


def test_a_language_present_only_in_passing_does_not_dispute_the_reading():
    # One short Catalan courtesy clause inside a long Spanish paragraph is a greeting,
    # not a misreading, and promoting on it would relabel plainly Spanish speeches.
    passing = ("Gràcies. Señorías, comparezco hoy para hablar del cierre de la fábrica "
               "y de sus efectos sobre el empleo en la comarca durante los próximos "
               "años, que es lo que de verdad preocupa a las familias afectadas.")

    assert paragraph_language(passing, _fake_detect) == "es"
    assert contesting_language(passing, _fake_detect) is None


# Disputed, but not enough. This pins CONTESTED_SHARE FROM BELOW, which nothing else
# does: lower the constant past this paragraph's share and it is offered to the
# alignment, where it can pair with the speech's own Spanish and take a plainly Spanish
# intervention out of the record — 749862 and 750542 are the corpus cases, at 0.235 and
# 0.171. Neither can be pinned by the bitext gold set: they carry no renderings, so the
# metric there ("which paragraphs left the as-delivered block") is empty whether the
# classifier is right or wrong, and only the verdict moves.
NEARLY_DISPUTED = (
    "Señorías, comparezco hoy para hablar del cierre de la fábrica y de sus efectos "
    "sobre el empleo en la comarca. Las cifras del último trimestre son mucho peores "
    "de lo que el Gobierno reconoce. Però nosaltres això no ho farem mai, senyories, "
    "mai de mai.")


def test_a_reading_disputed_but_not_enough_is_left_where_the_vote_put_it():
    votes = language_votes(NEARLY_DISPUTED, _fake_detect)
    share = votes["ca"] / sum(votes.values())

    # Sized to 749862's own 0.235, so this forbids exactly the floors that break it. A
    # fixture further below the constant would pass while the constant was lowered into
    # the range that breaks real speeches, and would guard nothing worth guarding.
    assert 0.23 < share < CONTESTED_SHARE
    assert contesting_language(NEARLY_DISPUTED, _fake_detect) is None


def test_language_votes_report_every_language_that_voted():
    votes = language_votes(CODE_MIXED, _fake_detect)

    assert set(votes) == {"es", "ca"}
    assert votes["es"] > votes["ca"] > 0


def test_a_paragraph_too_short_to_vote_has_no_votes():
    assert language_votes("Gràcies.", _fake_detect) == {}


def test_renders_scores_a_translation_above_an_unrelated_stretch():
    original = ("Tota la ideologia de VOX està a l'Enciclopedia Álvarez, "
                "editat l'any 1964, un dels manuals del franquisme.")
    translation = ("Toda la ideología de VOX está en la enciclopedia Álvarez, "
                   "editado en el año 1964, uno de los manuales del franquismo.")
    unrelated = ("Señorías, hoy debatimos el cierre de la fábrica y sus efectos "
                 "sobre el empleo en la comarca.")

    assert renders(original, translation) > renders(original, unrelated)


def test_renders_is_zero_against_nothing():
    assert renders("", "algo") == 0.0
    assert renders("Gràcies, senyories.", "") == 0.0


def test_sentence_spans_exclude_the_whitespace_between_them():
    text = "Una frase. Una altra frase."
    spans = sentence_spans(text)

    assert [text[a:b] for a, b in spans] == ["Una frase.", "Una altra frase."]


# ---- co-official spelling: the one signal that reaches a courtesy line ----------------

def test_a_co_official_courtesy_line_is_recognised_by_its_spelling():
    for line in ("Eskerrik asko.", "Moltes gràcies.", "Moito obrigado.",
                 "Besterik ez. Eskerrik asko.", "Grazas, señor presidente.",
                 "Acabo ja. Creo que se ha entendido. Gràcies."):
        assert carries_co_official_spelling(line), line


def test_a_spanish_courtesy_line_is_not():
    """The whole precision of the rule. These are the lines that are genuinely a rendering
    of a co-official farewell, and 20 sampled corpus speeches confirm they belong in the
    Spanish block alone — the Diario splits an embedded co-official farewell out as its own
    Spanish paragraph."""
    for line in ("Muchas gracias.", "Gracias.", "Buenas tardes.", "Gracias, presidente.",
                 "Muchas gracias, señora presidenta.", "Nada más y muchas gracias.",
                 "Buenos días a todas y a todos.", "Termino inmediatamente."):
        assert not carries_co_official_spelling(line), line


def test_the_lexicon_matches_an_accented_word():
    """`_TOKEN` is `[a-z0-9]+`, which splits "gràcies" into "gr" and "cies". Reusing it
    here would make the lexicon silently never match a correctly accented Catalan word —
    i.e. the commonest spelling in the Diario."""
    assert carries_co_official_spelling("Moltes gràcies.")
    assert carries_co_official_spelling("Moltes gracies.")   # the Diario drops accents too


def test_a_spanish_word_that_folds_onto_a_co_official_one_is_not_admitted():
    """Folding is only safe because no admitted entry collides with Spanish. These are the
    near misses that were deliberately kept out of the lexicon."""
    assert not carries_co_official_spelling("Muy buenos días, señora presidenta.")
    assert not carries_co_official_spelling("La respuesta tarda en llegar.")
    assert not carries_co_official_spelling("Acabo, señora presidenta.")


def test_only_a_paragraph_too_short_to_have_a_language_is_offered():
    """A longer co-official paragraph founds its own run and is already in the as-delivered
    block, so it needs no rescue — and rescuing it would reach the code-mixed class, which
    the alignment settles on evidence instead."""
    short = "Moltes gràcies."
    long_enough = ("Moltes gràcies, senyora presidenta, i moltes gràcies també a "
                   "totes les senyories que han intervingut aquesta tarda en el debat.")
    assert len(long_enough) > MIN_VOTING_CHARS
    text = f"{short}\n\n{long_enough}"

    assert co_official_spelling_spans(text) == [(0, len(short))]


def test_an_annotation_cannot_carry_a_paragraph_over_the_length_gate():
    """Same reading `paragraph_language` uses: what the stenographer noted is not the
    speaker's voice, so it neither votes nor lends length."""
    line = ("Gràcies. (Aplausos de las señoras y los señores diputados del Grupo "
            "Parlamentario Republicano, puestos en pie).")
    assert len(line) > MIN_VOTING_CHARS
    assert co_official_spelling_spans(line) == [(0, len(line))]

# ---- the mirror: delivered Spanish absorbed into a co-official original ---------------
#
# This side cannot use a flat "Spanish-only" list, because Spanish shares most of its
# courtesy vocabulary with Galician. The table records how each language says the word and
# a word is evidence only where the spellings DIFFER.

def test_a_spanish_formula_is_evidence_where_the_language_spells_it_otherwise():
    for line in ("Gracias.", "Muchas gracias.", "Buenas tardes.", "Salud y república."):
        assert carries_spanish_spelling(line, "ca"), line
        assert carries_spanish_spelling(line, "gl"), line
        assert carries_spanish_spelling(line, "eu"), line


def test_a_form_of_address_counts_against_catalan_but_not_against_galician():
    """Catalan says `president`, Galician says `presidente`. So the same line is evidence in
    a Catalan speech and none at all in a Galician one — which is why the flat list pushed
    `'O presidente.'` into the Spanish block of a Galician speech."""
    assert carries_spanish_spelling("Sí, presidente.", "ca")
    assert not carries_spanish_spelling("Sí, presidente.", "gl")

    assert carries_spanish_spelling("Señor ministro.", "ca")      # ca: senyor ministre
    assert not carries_spanish_spelling("Señor ministro.", "gl")  # gl: señor ministro


def test_a_word_spelled_the_same_in_both_languages_is_refused_everywhere():
    """`acabo` caused SEVEN of the ten false positives a flat list produced. It is spelled
    identically in Catalan and Galician, so the rule refuses it without anyone having to
    remember the incident."""
    for lang in ("ca", "gl"):
        assert not carries_spanish_spelling("Acabo.", lang)
        assert not carries_spanish_spelling("Acabo ja immediatament.", lang)
    assert not carries_spanish_spelling("Benvolguda ministra.", "ca")   # ca: ministra
    assert not carries_spanish_spelling("Máis nada.", "gl")             # gl: nada


def test_nada_counts_against_catalan_which_says_res():
    assert carries_spanish_spelling("Nada más.", "ca")
    assert not carries_spanish_spelling("Nada más.", "gl")


def test_a_line_spelling_the_co_official_language_is_refused_rather_than_guessed_at():
    """The undecidable case. Its translation may already stand in the Spanish block, so
    adding it would duplicate a line the Diario rendered. Refusing costs a rescue; guessing
    costs a duplicate."""
    assert not carries_spanish_spelling("Grazas, señor presidente.", "gl")
    assert not carries_spanish_spelling("Gracias, presidente. Eskerrik asko.", "eu")
    assert not carries_spanish_spelling("Moltes gràcies, senyora presidenta.", "ca")


def test_only_a_short_paragraph_is_offered_to_the_spanish_block():
    short = "Gracias, presidente."
    long_enough = ("Muchas gracias, señora presidenta, y muchas gracias también a todas "
                   "las señorías que han intervenido esta tarde en el debate.")
    assert len(long_enough) > MIN_VOTING_CHARS
    text = f"{short}\n\n{long_enough}"

    assert spanish_spelling_spans(text, "ca") == [(0, len(short))]


def test_both_lexicons_admit_only_formulas_and_forms_of_address():
    """The maintainability guarantee, as an assertion rather than a comment: adding a word
    means changing this test, which means reading the admission rule above the data. Without
    it the list grows by a word per failing speech and becomes an artifact of one legislature.

    `vull` ("I want") and `acabant` ("finishing") were removed after measuring that each
    rescued exactly one speech — ordinary vocabulary, not a formula.
    """
    from qhld_engine.domain.speeches.language_runs import (
        _CO_OFFICIAL_ONLY, _SPANISH_COGNATES)

    assert len(_CO_OFFICIAL_ONLY) == 33, "a new word needs the admission rule applied to it"
    assert len(_SPANISH_COGNATES) == 19
    for word in ("vull", "acabant", "besterik", "amaitzen", "bukatzen", "moi",
                 "diputats", "diputades"):
        assert word not in _CO_OFFICIAL_ONLY, word
    # No word may sit on both sides: the co-official set is also this side's veto, so a
    # shared word would make each rule silently ignore it.
    assert not (set(_SPANISH_COGNATES) & _CO_OFFICIAL_ONLY)
    # Every row must say how BOTH close relatives spell it, or the omission reads as
    # "no cognate" and the word counts as evidence by accident.
    for word, forms in _SPANISH_COGNATES.items():
        assert set(forms) == {"ca", "gl"}, word
