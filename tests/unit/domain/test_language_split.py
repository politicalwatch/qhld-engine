"""Unit tests for the block classifier.

The detector is a fake keyed on marker words, so these are deterministic and never load
py3langid. Real-detector behaviour on real data is covered by the speech-extraction
characterization test.

Clip lengths are derived from the text rather than hardcoded — what the classifier asks
is whether the text could have been spoken in the time, so a test that pins seconds
would break whenever the sample text is reworded.
"""

import pytest

from qhld_engine.domain.speeches.language_split import split_languages

pytestmark = pytest.mark.unit

# Catalan-only fragments, none a substring of its Spanish cognate.
_CA_MARKERS = ("senyories", "aquesta", "veritat", "nosaltres", "però", "això",
               "perquè", "molt", "gràcies", "hauria", "vostès")

DELIVERY_RATE = 13.3  # characters per second, the measured median


def _fake_detect(text):
    low = text.lower()
    return "ca" if any(m in low for m in _CA_MARKERS) else "es"


def _clip(*spoken):
    """A clip exactly long enough to deliver ``spoken`` at the median rate."""
    return sum(len(part) for part in spoken) / DELIVERY_RATE


# A paragraph and its rendering: they share the names and figures a translator copies,
# which is what the rendering test keys on.
CATALAN = (
    "La veritat, senyories, aquesta reforma del transport a Barcelona costarà 4000 "
    "milions d'euros segons el ministeri competent. Però nosaltres creiem que la "
    "xifra real hauria de ser molt més alta, perquè el projecte de Barcelona no "
    "inclou el manteniment. Això és el que vostès no expliquen.")
SPANISH = (
    "La verdad, señorías, esta reforma del transporte en Barcelona costará 4000 "
    "millones de euros según el ministerio competente. Pero creemos que la cifra "
    "real debería ser mucho más alta, porque el proyecto de Barcelona no incluye el "
    "mantenimiento. Eso es lo que ustedes no explican.")


def test_a_monolingual_spanish_speech_is_one_block():
    text = ("Muchas gracias, presidente. Comparezco hoy ante la Cámara para hablar "
            "del cierre de la fábrica y de sus efectos sobre el empleo.")
    split = split_languages(text, _fake_detect, _clip(text))

    assert split.language == "es"
    assert [(b.lang, b.text, b.original) for b in split.blocks] == [("es", text, True)]
    assert not split.undecided


def test_a_rendered_co_official_speech_is_two_blocks():
    # Only the Catalan was spoken; the Spanish is the Diario's rendering of it, so the
    # clip is long enough for one of them and not for both.
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(CATALAN))

    assert split.language == "ca"
    assert [(b.lang, b.original, b.partial) for b in split.blocks] == [
        ("ca", True, False), ("es", False, False)]
    assert split.blocks[0].text.startswith("La veritat")
    assert split.blocks[1].text.startswith("La verdad")
    assert not split.undecided


def test_a_spanish_speech_with_a_co_official_greeting_is_one_spanish_block():
    # The whole thing was spoken, and it is mostly Spanish: one block, in the language
    # it was delivered in, greeting included.
    greeting = "Moltes gràcies, senyora presidenta, però això és molt greu."
    body = (
        "Señorías, comparezco para hablar del cierre de la fábrica y de sus efectos "
        "sobre el empleo en la comarca durante los próximos años. Las cifras del "
        "último trimestre son mucho peores de lo que el Gobierno reconoce, y la "
        "situación de las familias afectadas empeora cada mes que pasa sin acuerdo.")
    text = f"{greeting}\n\n{body}"
    split = split_languages(text, _fake_detect, _clip(text))

    assert split.language == "es"
    assert len(split.blocks) == 1
    assert split.blocks[0].lang == "es"
    assert "Moltes gràcies" in split.blocks[0].text  # the greeting is not discarded


def test_a_co_official_speech_with_no_rendering_is_one_block():
    # Two languages, both delivered, no rendering of either: one block, and it keeps
    # every word.
    aside = ("Señor ministro, estoy verdaderamente emocionado de escuchar esto hoy "
             "aquí en esta Cámara, se lo digo de corazón.")
    text = f"{CATALAN}\n\n{aside}"
    split = split_languages(text, _fake_detect, _clip(text))

    assert split.language == "ca"
    assert len(split.blocks) == 1
    assert aside in split.blocks[0].text


def test_a_rendering_that_covers_only_part_is_marked_partial():
    # Two paragraphs delivered, one of them rendered: the Spanish block is real but
    # does not stand for the whole speech.
    untranslated = (
        "Però aquesta esmena, senyories, nosaltres la vam presentar fa mesos al "
        "registre i encara no hauria estat contestada per ningú del Govern. Això és "
        "el que volem denunciar aquí avui, perquè no és la primera vegada.")
    text = f"{CATALAN}\n\n{untranslated}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, untranslated))

    assert len(split.blocks) == 2
    assert split.blocks[1].partial is True


def test_a_speech_nothing_places_is_left_undecided():
    # Far more text than the clip can hold under any reading of it.
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, duration=5.0)

    assert split.undecided is True
    assert split.blocks  # a provisional reading, not nothing


def test_without_a_clip_the_verdict_is_provisional():
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, duration=None)

    assert split.undecided is True


def test_a_one_line_intervention_still_gets_a_block():
    # Nothing in it is long enough for the detector to read.
    split = split_languages("Sí.", _fake_detect, 4.0)

    assert [(b.lang, b.text) for b in split.blocks] == [("es", "Sí.")]


def test_empty_text_has_no_blocks():
    split = split_languages("   ", _fake_detect, None)

    assert split.blocks == []
    assert split.language == "es"


def test_paragraph_breaks_survive_inside_each_block():
    original = f"{CATALAN}\n\n{CATALAN}"
    text = f"{original}\n\n{SPANISH}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(original))

    assert len(split.blocks) == 2
    assert "\n\n" in split.blocks[0].text
    assert "\n\n" in split.blocks[1].text
