"""Unit tests for language runs — the detector is a fake keyed on marker words, so
these are deterministic and never load py3langid.
"""

import pytest

from qhld_engine.domain.speeches.language_runs import (
    paragraph_language,
    paragraph_runs,
    renders,
    sentence_spans,
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


def test_empty_text_has_no_runs():
    assert paragraph_runs("", _fake_detect) == []
    assert paragraph_runs("   ", _fake_detect) == []


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
