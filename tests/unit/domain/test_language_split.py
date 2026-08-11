"""Unit tests for the block classifier.

The detector is a fake keyed on marker words, so these are deterministic and never load
py3langid. Real-detector behaviour on real data is covered by the speech-extraction
characterization test.

Clip lengths are derived from the text rather than hardcoded — what the classifier asks
is whether the text could have been spoken in the time, so a test that pins seconds
would break whenever the sample text is reworded.
"""

import re

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


def _words(text):
    return {w for w in re.findall(r"[\wàèéíòóúïüç]{5,}", text.lower())}


def _fake_similarity(source, candidate):
    """Stands in for the multilingual embedding, deterministically and without a model.

    The real instrument scores meaning, which is why it can see a rendering that shares
    almost no spelling with its original. A unit test cannot reproduce that and should
    not pretend to: what it needs is something that says "these two are a pair" for the
    fixtures that ARE pairs and not for the ones that are not. The fixtures pair Catalan
    with Spanish text carrying the same figures and proper nouns, so three shared long
    words is exactly that signal — and a paragraph the fixtures mean as unrelated shares
    at most one.
    """
    shared = _words(source) & _words(candidate)
    return 0.95 if len(shared) >= 3 else 0.05


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
    split = split_languages(text, _fake_detect, _clip(text), similarity=_fake_similarity)

    assert split.language == "es"
    assert [(b.lang, b.text, b.original) for b in split.blocks] == [("es", text, True)]
    assert not split.undecided


def test_a_rendered_co_official_speech_is_two_blocks():
    # Only the Catalan was spoken; the Spanish is the Diario's rendering of it, so the
    # clip is long enough for one of them and not for both.
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(CATALAN), similarity=_fake_similarity)

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
    split = split_languages(text, _fake_detect, _clip(text), similarity=_fake_similarity)

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
    split = split_languages(text, _fake_detect, _clip(text), similarity=_fake_similarity)

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
    split = split_languages(text, _fake_detect, _clip(CATALAN, untranslated), similarity=_fake_similarity)

    assert len(split.blocks) == 2
    assert split.blocks[1].partial is True


def test_a_short_paragraph_does_not_make_a_long_one_its_rendering():
    # The closing Catalan line names Barcelona and the ministry, and so does the Spanish
    # the speaker went on to deliver — enough shared vocabulary to pair them, and nowhere
    # near enough length for one to be a rendering of the other. Left paired, that Spanish
    # would leave the record of what was said, and the speech would no longer fit its clip.
    closing = "Però això, senyories, el ministeri de Barcelona no ho explica."
    spoken_tail = (
        "Y termino, señorías. No voy a pedir aquí al ministerio que gestione un poco "
        "mejor la reforma del transporte de Barcelona; voy a pedir que la retire "
        "entera, porque las cifras que hemos conocido esta misma semana no admiten "
        "ninguna otra lectura razonable.")
    text = f"{CATALAN}\n\n{closing}\n\n{SPANISH}\n\n{spoken_tail}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, closing, spoken_tail), similarity=_fake_similarity)

    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    # it was spoken, so it is in the record of what was said...
    assert spoken_tail in split.blocks[0].text
    # ...and still in the Spanish side as the Diario prints it
    assert spoken_tail in split.blocks[1].text
    # the real rendering is unaffected
    assert split.blocks[1].text.startswith("La verdad")
    # ...and the Spanish side is correctly PARTIAL, because `closing` is Catalan that
    # nothing renders. Whole-token overlap used to pair `closing` with the Spanish tail
    # on their shared proper nouns and call the rendering complete; scoring meaning
    # instead declines that pairing, which is the whole point of the change.
    assert split.blocks[1].partial is True


def test_every_block_names_its_own_languages_starting_with_the_dominant_one():
    """`lang` is what four other things key off — the subtitle track id, the aligner, the
    taggable text and VTT selection — so `langs` may extend it but must never disagree
    with it."""
    aside = ("Señor ministro, estoy verdaderamente emocionado de escuchar esto hoy "
             "aquí en esta Cámara, se lo digo de corazón. Y se lo repito porque hace "
             "falta decirlo más veces en esta Cámara y fuera de ella.")
    text = f"{CATALAN}\n\n{aside}"
    split = split_languages(text, _fake_detect, _clip(text), similarity=_fake_similarity)

    # one block, delivered, and it really is in both languages
    assert len(split.blocks) == 1
    assert split.blocks[0].langs == ("ca", "es")
    assert all(b.langs[0] == b.lang for b in split.blocks)


def test_a_monolingual_speech_names_one_language():
    text = ("Muchas gracias, presidente. Comparezco hoy ante la Cámara para hablar "
            "del cierre de la fábrica y de sus efectos sobre el empleo.")
    split = split_languages(text, _fake_detect, _clip(text), similarity=_fake_similarity)

    assert split.blocks[0].langs == ("es",)


def test_a_language_present_only_in_passing_is_not_named():
    """A speech is not bilingual because it opened with a greeting."""
    greeting = "Moltes gràcies, senyora presidenta."
    body = " ".join(
        ["Señorías, comparezco para hablar del cierre de la fábrica y de sus efectos "
         "sobre el empleo en la comarca durante los próximos años."] * 12)
    split = split_languages(f"{greeting}\n\n{body}", _fake_detect,
                            _clip(greeting, body), similarity=_fake_similarity)

    assert split.blocks[0].lang == "es"
    assert split.blocks[0].langs == ("es",)   # the greeting is kept, but not named
    assert "Moltes gràcies" in split.blocks[0].text


def test_a_one_line_intervention_still_names_its_language():
    split = split_languages("Sí.", _fake_detect, 4.0, similarity=_fake_similarity)

    assert split.blocks[0].langs == ("es",)


def test_a_speech_nothing_places_is_left_undecided():
    # Far more text than the clip can hold under any reading of it.
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, duration=5.0)

    assert split.undecided is True
    assert split.blocks  # a provisional reading, not nothing


def test_without_a_clip_the_verdict_is_provisional():
    text = f"{CATALAN}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, duration=None, similarity=_fake_similarity)

    assert split.undecided is True


def test_a_one_line_intervention_still_gets_a_block():
    # Nothing in it is long enough for the detector to read.
    split = split_languages("Sí.", _fake_detect, 4.0, similarity=_fake_similarity)

    assert [(b.lang, b.text) for b in split.blocks] == [("es", "Sí.")]


def test_empty_text_has_no_blocks():
    split = split_languages("   ", _fake_detect, None)

    assert split.blocks == []
    assert split.language == "es"


def test_paragraph_breaks_survive_inside_each_block():
    original = f"{CATALAN}\n\n{CATALAN}"
    text = f"{original}\n\n{SPANISH}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(original), similarity=_fake_similarity)

    assert len(split.blocks) == 2
    assert "\n\n" in split.blocks[0].text
    assert "\n\n" in split.blocks[1].text
