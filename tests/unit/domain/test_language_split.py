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

# A second pair, for the fixtures that need a speech long enough to have a shape. It
# shares its three long words with its own rendering and none with the pair above, so the
# alignment has a wrong answer available and has to decline it.
CATALAN_TARRAGONA = (
    "Senyories, l'hospital de Tarragona fa anys que espera que la Generalitat hi "
    "destini els recursos comarcals que li pertoquen. Nosaltres això ho hem denunciat "
    "aquí moltes vegades i mai no hem obtingut cap resposta concreta de ningú.")
SPANISH_TARRAGONA = (
    "Señorías, el hospital de Tarragona lleva años esperando que la Generalitat le "
    "destine los recursos comarcales que le corresponden. Lo hemos denunciado aquí "
    "muchas veces y nunca hemos obtenido ninguna respuesta concreta de nadie.")


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


UNTRANSLATED = (
    "Però aquesta esmena, senyories, nosaltres la vam presentar fa mesos al "
    "registre i encara no hauria estat contestada per ningú del Govern. Això és "
    "el que volem denunciar aquí avui, perquè no és la primera vegada.")


def test_a_rendering_that_covers_only_part_is_marked_partial():
    # Two paragraphs delivered, one of them rendered: the Spanish block is real but part
    # of what it holds is the original rather than a translation of it.
    text = f"{CATALAN}\n\n{UNTRANSLATED}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, UNTRANSLATED), similarity=_fake_similarity)

    assert len(split.blocks) == 2
    assert split.blocks[1].partial is True


def test_a_stretch_nobody_translated_stands_in_both_blocks():
    # Each block is the whole speech seen one way, so a stretch with no counterpart on the
    # other side belongs to both: the Diario printed no Spanish for this paragraph, so the
    # Spanish side has nothing of its own to put in its place and keeps the Catalan.
    text = f"{CATALAN}\n\n{UNTRANSLATED}\n\n{SPANISH}"
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, UNTRANSLATED),
        similarity=_fake_similarity).blocks

    assert UNTRANSLATED in delivered.text
    assert UNTRANSLATED in spanish.text
    # ...while the pair that IS translated is split, each side in its own block only
    assert CATALAN in delivered.text and CATALAN not in spanish.text
    assert SPANISH in spanish.text and SPANISH not in delivered.text
    # so between them the two blocks account for every paragraph of the document
    assert not {p for p in text.split("\n\n")} - set(
        delivered.text.split("\n\n")) - set(spanish.text.split("\n\n"))
    # and the Spanish block names the language it had to borrow
    assert spanish.langs == ("es", "ca")


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
    assert SPANISH in split.blocks[1].text
    # and `closing` stands in both blocks: nothing renders it, so the Spanish side has
    # nothing of its own to put there
    assert closing in split.blocks[0].text
    assert closing in split.blocks[1].text
    # ...and the Spanish side is correctly PARTIAL, because `closing` is Catalan that
    # nothing renders. Whole-token overlap used to pair `closing` with the Spanish tail
    # on their shared proper nouns and call the rendering complete; scoring meaning
    # instead declines that pairing, which is the whole point of the change.
    assert split.blocks[1].partial is True


OPENING_CA = ("Gràcies, presidenta. Aquesta vegada parlaré en català perquè així m'ho "
              "demana el meu grup, i això no ho hauria de sorprendre ningú.")
# One Spanish paragraph rendering BOTH the opening and the paragraph after it — the shape
# the Diario produces whenever it re-paragraphs, and the reason whole-paragraph similarity
# is not enough on its own.
SPANISH_MERGED = ("Esta vez hablaré en catalán porque así me lo pide mi grupo, y eso no "
                  "debería sorprender a nadie. " + SPANISH)


def _graded_similarity(source, candidate):
    """Scores the merge above the neighbour alone, which a binary stub cannot express."""
    if "no explican" not in candidate:
        return 0.05
    opening, body = "parlaré en català" in source, "aquesta reforma" in source
    if opening and body:
        return 0.95              # both halves: the best reading of this Spanish paragraph
    if body:
        return 0.90              # the neighbour alone already pairs
    if opening:
        return 0.60              # the opening alone falls under the floor
    return 0.05


def test_a_source_the_diario_merged_with_its_neighbour_is_still_a_source():
    # The opening scores 0.60 against the Spanish paragraph that renders it, because
    # two-thirds of that paragraph renders the NEXT one — under the floor, so the alignment
    # leaves it unpaired and the Spanish block ends up carrying the Catalan original beside
    # the translation of it.
    text = f"{OPENING_CA}\n\n{CATALAN}\n\n{SPANISH_MERGED}"
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(OPENING_CA, CATALAN),
        similarity=_graded_similarity).blocks

    assert OPENING_CA in delivered.text
    # ...and NOT in the Spanish block, because a translation of it exists there already
    assert OPENING_CA not in spanish.text
    # the whole original is accounted for, so nothing is missing a translation
    assert spanish.partial is False


def test_a_merge_that_reads_worse_than_the_neighbour_alone_is_refused():
    """The negative control, and what it guards is the COMPARISON. Any paragraph can be glued
    to a paired neighbour and the glued text will still read like a rendering of something —
    here it scores 0.70, comfortably over the floor. Only measuring it against what the
    neighbour scored alone can refuse it, so this test fails if the rule is ever relaxed to
    "the merge clears the floor"."""
    def _worse_merged(source, candidate):
        if "no explican" not in candidate:
            return 0.05
        if "aquesta reforma" in source and UNTRANSLATED[:20] in source:
            return 0.70          # the merge reads WORSE than the neighbour did alone
        return 0.90 if "aquesta reforma" in source else 0.05

    text = f"{CATALAN}\n\n{UNTRANSLATED}\n\n{SPANISH}"
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, UNTRANSLATED),
        similarity=_worse_merged).blocks

    # refused, so the paragraph is still nobody's source and still stands in both blocks
    assert UNTRANSLATED in delivered.text
    assert UNTRANSLATED in spanish.text
    assert spanish.partial is True


def test_a_quotation_read_aloud_stays_in_the_record_of_what_was_said():
    # The Diario prints the citation on its own paragraph, AFTER the Spanish rendering of
    # the sentence that announces it. Having no language of its own it joins the run
    # before it, so removing that rendering used to take the citation with it — and the
    # citation was spoken, once, as part of the Catalan delivery.
    quotation = ("«Los sepultureros más eficaces de un imperio suelen ser los mismos "
                 "imperialistas, y no hay nada más que discutir sobre ello».")
    text = f"{CATALAN}\n\n{SPANISH}\n\n{quotation}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, quotation),
                            similarity=_fake_similarity)

    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert quotation in split.blocks[0].text
    # ...and still in the Spanish side as the Diario prints it
    assert quotation in split.blocks[1].text


def test_a_paragraph_is_not_rendered_by_far_more_text_than_itself():
    # Nothing costs the alignment anything to over-claim, so a closing line will take a
    # much longer Spanish paragraph that merely takes up its point. Only proportion can
    # refuse that claim, and refusing it is what keeps 300 characters the speaker said in
    # the record of what was said.
    closing = "Però això, senyories, el ministeri de Barcelona no ho explica."
    spoken_tail = (
        "Y termino, señorías. No voy a pedir aquí que se gestione un poco mejor; voy a "
        "pedir que la retire entera, porque las cifras que hemos conocido esta misma "
        "semana no admiten ninguna otra lectura razonable en esta Cámara ni fuera de "
        "ella, y ustedes lo saben perfectamente.")

    def _similarity(source, candidate):
        if "retire entera" in candidate:
            # It reads as a plausible rendering of the closing line, and of nothing else
            # in the speech — but a weaker one than a real pair.
            return 0.70 if "no ho explica" in source else 0.05
        return _fake_similarity(source, candidate)

    text = (f"{CATALAN}\n\n{CATALAN_TARRAGONA}\n\n{closing}\n\n"
            f"{SPANISH}\n\n{SPANISH_TARRAGONA}\n\n{spoken_tail}")
    split = split_languages(
        text, _fake_detect, _clip(CATALAN, CATALAN_TARRAGONA, closing, spoken_tail),
        similarity=_similarity)

    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert spoken_tail in split.blocks[0].text
    # the real rendering is unaffected
    assert SPANISH not in split.blocks[0].text
    assert split.blocks[1].text.startswith("La verdad")


def test_a_rendering_may_run_several_times_the_length_of_its_original():
    # The other edge of the same rule, and the reason it is not simply set tight: the
    # Diario prints a Catalan paragraph the speaker was cut off mid-sentence, and the
    # interpreter finishes the thought. The Spanish runs three times as long and is a
    # rendering all the same.
    cut_off = ("Senyories, aquesta renovació no servirà de res si no anem molt més "
               "enllà del que vostès proposen avui aquí, si no fem realment…")
    finished = (
        "Señorías, esa renovación no va a servir para nada si no vamos mucho más allá "
        "de lo que ustedes proponen hoy aquí, si no hacemos realmente una reforma del "
        "sistema del Poder Judicial y del sistema democrático que vaya mucho más allá "
        "de este acuerdo al que acaban de llegar. O nos lo tomamos en serio las "
        "personas demócratas y progresistas y hacemos acciones políticas a favor de la "
        "mayoría, o no servirá absolutamente para nada de nada.")

    def _similarity(source, candidate):
        if "Poder Judicial" in candidate:
            return 0.95 if "no servirà de res" in source else 0.05
        return _fake_similarity(source, candidate)

    text = f"{CATALAN}\n\n{cut_off}\n\n{SPANISH}\n\n{finished}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, cut_off),
                            similarity=_similarity)

    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert finished not in split.blocks[0].text
    assert finished in split.blocks[1].text


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
