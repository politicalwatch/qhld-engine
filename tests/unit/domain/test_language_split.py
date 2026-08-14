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


# ---- a paragraph the printed page reads as Spanish ---------------------------------
#
# The speaker argues about a Spanish phrase and quotes it back, so only his closing
# sentence is unmistakably Catalan and the vote calls the paragraph Spanish. Filed among
# the interpreter's paragraphs, nothing can be a rendering OF it — and its own rendering
# then has no source to belong to.
CODE_MIXED_CA = (
    "Y esto, señorías, no lo hace un partido de Estado, se lo digo yo. "
    "Se lo digo con todo el cariño del mundo, de verdad se lo digo. "
    "Però nosaltres això no ho farem mai, senyories, mai de la vida.")
RENDERS_CODE_MIXED = (
    "Y esto no lo hace un partido de Estado, se lo digo yo, se lo digo con todo el "
    "cariño del mundo. Pero nosotros eso no lo haremos nunca, señorías, nunca jamás.")


def _pairs(*couples):
    """A similarity that pairs exactly the given (original, rendering) couples."""
    def similarity(source, candidate):
        return 0.95 if any(a in source and b in candidate for a, b in couples) else 0.05
    return similarity


def test_a_paragraph_read_as_spanish_is_an_original_when_something_renders_it():
    # The whole point: it is offered to the alignment, a rendering of it turns up, and it
    # takes its place in the record of what was said instead of standing among the
    # interpreter's paragraphs with its own translation orphaned beside it.
    text = f"{CATALAN}\n\n{CODE_MIXED_CA}\n\n{SPANISH}\n\n{RENDERS_CODE_MIXED}"
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, CODE_MIXED_CA),
        similarity=_pairs((CATALAN[:30], SPANISH[:30]),
                          (CODE_MIXED_CA[:30], RENDERS_CODE_MIXED[:30]))).blocks

    assert CODE_MIXED_CA in delivered.text
    # ...and its rendering leaves the record of what was said, which it could not do
    # while the paragraph it renders was filed as Spanish
    assert RENDERS_CODE_MIXED not in delivered.text
    assert RENDERS_CODE_MIXED in spanish.text
    assert CODE_MIXED_CA not in spanish.text


def test_a_paragraph_read_as_spanish_stays_spanish_when_nothing_renders_it():
    """The offer has to be earned. A disputed reading is not a better reading — no
    configuration of the detector reads these paragraphs correctly — so the only thing
    that may move one is a rendering of it actually turning up. Nothing here renders it,
    so it stays where the vote put it and, having no counterpart, stands in both blocks."""
    text = f"{CATALAN}\n\n{CODE_MIXED_CA}\n\n{SPANISH}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, CODE_MIXED_CA),
                            similarity=_pairs((CATALAN[:30], SPANISH[:30])))
    delivered, spanish = split.blocks

    assert split.language == "ca"
    assert CODE_MIXED_CA in delivered.text
    assert CODE_MIXED_CA in spanish.text
    # And it is still counted as Spanish, which is what `partial` reveals: kept as an
    # original the alignment never paired, it would read as a co-official stretch the
    # Diario failed to translate, and the Spanish block would be flagged incomplete over
    # a paragraph whose translation is missing only because none was ever needed.
    assert spanish.partial is False


def test_a_paragraph_already_serving_as_a_rendering_is_never_offered():
    """A translation is not a misread original. Offering one would take it away from the
    source it explains, and a healthy speech would be refused to fix a defect it does not
    have — so a run the alignment has already spoken for is not a candidate, however its
    own vote went."""
    text = f"{CATALAN}\n\n{CODE_MIXED_CA}"
    split = split_languages(text, _fake_detect, _clip(CATALAN),
                            similarity=_pairs((CATALAN[:30], CODE_MIXED_CA[:30])))
    delivered, spanish = split.blocks

    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert CODE_MIXED_CA not in delivered.text
    assert CODE_MIXED_CA in spanish.text


# ---- an extra claim must not read worse joined --------------------------------------
#
# The same comparison `_merged_into_a_neighbour` makes, from the other side. Nothing costs
# the alignment anything to over-claim, and proportionality can only refuse a claim for
# being too LONG for its source, never for simply not belonging to it.
SPOKEN_ON = (
    "Y termino, señorías, con una última consideración sobre ese mismo proyecto: el "
    "ministerio sabe desde hace meses que la cifra no cuadra y aun así la sigue "
    "defendiendo en esta Cámara como si nada hubiera pasado.")

# One rendering the Diario broke across a paragraph break — SPANISH, in two halves.
SPANISH_OPENING = ("La verdad, señorías, esta reforma del transporte en Barcelona "
                   "costará 4000 millones de euros según el ministerio competente. "
                   "Pero creemos que la cifra real debería ser mucho más alta, porque "
                   "el proyecto de Barcelona no incluye el mantenimiento.")
# Over MIN_VOTING_CHARS on purpose: below it this paragraph gets no vote, joins the run
# before it, and the two halves become ONE claim — at which point the rule under test
# never runs and the fixture passes whatever it does.
SPANISH_REST = ("Eso es lo que ustedes no explican, señorías, y conviene decirlo hoy "
                "aquí con toda claridad.")


def test_a_source_does_not_claim_the_words_spoken_next_to_its_rendering():
    """The speaker carries the same thought on in Spanish after the interpretation. It
    says what the original says, so it reads as a rendering of it (0.70, over the floor)
    and is proportionate enough to survive the length check — and claimed, it leaves the
    record of what was said. Only measuring it AGAINST the real rendering refuses it."""
    def _similarity(source, candidate):
        if "aquesta reforma" not in source:
            return 0.05
        if SPANISH in candidate and SPOKEN_ON in candidate:
            return 0.80          # joined, it reads worse than the rendering alone
        if SPANISH in candidate:
            return 0.90
        if SPOKEN_ON in candidate:
            return 0.70
        return 0.05

    text = f"{CATALAN}\n\n{SPANISH}\n\n{SPOKEN_ON}"
    split = split_languages(text, _fake_detect, _clip(CATALAN, SPOKEN_ON),
                            similarity=_similarity)

    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert SPOKEN_ON in split.blocks[0].text
    # the real rendering is untouched, and still the only thing removed
    assert SPANISH not in split.blocks[0].text
    assert SPANISH in split.blocks[1].text
    assert split.blocks[1].partial is False


def test_a_rendering_the_diario_broke_in_two_keeps_both_halves():
    """The negative control, and what it guards is the COMPARISON rather than a level.
    The second half reads no better against the original than the spoken paragraph above
    did — 0.70 either way. What separates them is that supplying it IMPROVES the match,
    because it is the part that was missing, so this fails if the rule is ever relaxed to
    anything the second claim can be judged on alone."""
    def _similarity(source, candidate):
        if "aquesta reforma" not in source:
            return 0.05
        if SPANISH_OPENING in candidate and SPANISH_REST in candidate:
            return 0.95
        if SPANISH_OPENING in candidate:
            return 0.85
        if SPANISH_REST in candidate:
            return 0.70
        return 0.05

    text = f"{CATALAN}\n\n{SPANISH_OPENING}\n\n{SPANISH_REST}"
    # Long enough for the original AND the tail, deliberately: sized to the original
    # alone, dropping the tail makes the speech unplaceable and the clip falls back to
    # the unpruned reading, which is the right answer for the wrong reason — and this
    # then passes even for a rule that drops every extra claim it sees.
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, SPANISH_REST),
        similarity=_similarity).blocks

    assert SPANISH_OPENING not in delivered.text
    assert SPANISH_REST not in delivered.text
    assert SPANISH_OPENING in spanish.text and SPANISH_REST in spanish.text


def test_one_paragraph_may_render_two_originals_and_belong_to_only_one_of_them():
    """A Spanish paragraph rendering the end of one original and the whole of the next is
    claimed by both, and reads as the first one poorly — it is mostly about the second.
    Refusing it to the first must not take it from the second: what is dropped is a
    CLAIM, and a paragraph stays a rendering while any source still owns it. Judged per
    paragraph instead, the translation lands back in the record of what was said as if
    the speaker had read it out."""
    def _similarity(source, candidate):
        if "aquesta reforma" in source:
            if SPANISH in candidate and SPANISH_TARRAGONA in candidate:
                return 0.80      # joined, worse than its own rendering alone
            if SPANISH in candidate:
                return 0.95
            if SPANISH_TARRAGONA in candidate:
                return 0.70      # over the floor, so the alignment takes it too
        if "hospital de Tarragona" in source:
            return 0.95 if SPANISH_TARRAGONA in candidate else 0.05
        return 0.05

    text = f"{CATALAN}\n\n{CATALAN_TARRAGONA}\n\n{SPANISH}\n\n{SPANISH_TARRAGONA}"
    # A clip long enough for BOTH readings, deliberately: sized to the originals alone,
    # losing the shared rendering makes the speech unplaceable, the clip falls back to
    # the unpruned reading and quietly produces the right answer for the wrong reason.
    # Then this passes even when the drop is applied per PARAGRAPH, which is the mistake
    # it exists to catch.
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, CATALAN_TARRAGONA, SPANISH_TARRAGONA),
        similarity=_similarity).blocks

    assert SPANISH_TARRAGONA not in delivered.text
    assert SPANISH not in delivered.text
    assert SPANISH in spanish.text and SPANISH_TARRAGONA in spanish.text


def test_a_reading_the_clip_cannot_place_gives_way_to_one_it_can():
    """Refusing an over-claim is a hypothesis about the text; the clip is independent
    evidence about how much of it can have been spoken. Here the strict reading returns
    500 characters to a speech that only had time for 270, so the speech becomes
    unplaceable — and it is the reading that is wrong, not the speech. Without the
    fallback this is refused, and a decided pair loses its Spanish block."""
    extra = ("Y termino ya, señorías, insistiendo una vez más en que el ministerio "
             "conocía perfectamente estas cifras desde hace muchos meses y aun así no "
             "ha movido un solo dedo para corregirlas.")

    def _similarity(source, candidate):
        if "aquesta reforma" not in source:
            return 0.05
        if SPANISH in candidate and extra in candidate:
            return 0.80          # the strict rule would drop `extra`...
        if SPANISH in candidate:
            return 0.90
        if extra in candidate:
            return 0.70
        return 0.05

    text = f"{CATALAN}\n\n{SPANISH}\n\n{extra}"
    split = split_languages(text, _fake_detect, _clip(CATALAN), similarity=_similarity)

    # ...but only the Catalan fits the clip, so the reading that keeps `extra` a
    # rendering is the one the evidence supports
    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("ca", True), ("es", False)]
    assert extra not in split.blocks[0].text
    assert extra in split.blocks[1].text


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


def test_a_refused_speech_is_named_by_what_most_of_it_is_in():
    # A Spanish speech quoting one Catalan paragraph in the middle of it. The cut has no
    # co-official front to separate — Spanish both precedes and follows the quotation —
    # so the reading is a single block, and naming that block after the quotation would
    # file a Spanish speech under Catalan.
    quotation = ("Però nosaltres això ho hem denunciat aquí moltes vegades i mai no "
                 "hem obtingut cap resposta.")
    text = f"{SPANISH}\n\n{quotation}\n\n{SPANISH_TARRAGONA}"
    split = split_languages(text, _fake_detect, duration=5.0,
                            similarity=_fake_similarity)

    assert split.undecided is True
    assert len(split.blocks) == 1
    assert split.blocks[0].text == text   # nothing is dropped by naming it
    assert split.blocks[0].lang == "es"
    assert split.language == "es"


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


# ---- shape 4: a Spanish speech carrying a passage the Diario rendered ----------------

SPANISH_BODY = (
    "Señorías, comparezco para hablar del cierre de la fábrica y de sus efectos sobre "
    "el empleo en la comarca durante los próximos años. Las cifras del último "
    "trimestre son mucho peores de lo que el Gobierno reconoce, y la situación de las "
    "familias afectadas empeora cada mes que pasa sin acuerdo.")
SPANISH_TAIL = (
    "Termino pidiendo al Gobierno que convoque la mesa sectorial antes de que acabe el "
    "mes, porque cada semana que pasa sin convocarla cuesta empleos que después ya no "
    "se recuperan nunca.")


def test_a_spanish_speech_whose_co_official_passage_was_rendered_gets_two_blocks():
    # Most of it was given in Spanish, one passage in Catalan, and the Diario printed
    # that passage in Spanish as well. The rendering was never spoken, so it cannot stay
    # in the record of what was said — but the speech is Spanish and must not be renamed
    # over one passage.
    text = f"{SPANISH_BODY}\n\n{CATALAN}\n\n{SPANISH}\n\n{SPANISH_TAIL}"
    split = split_languages(text, _fake_detect,
                            _clip(SPANISH_BODY, CATALAN, SPANISH_TAIL),
                            similarity=_pairs((CATALAN[:30], SPANISH[:30])))
    delivered, spanish = split.blocks

    assert split.language == "es"
    assert not split.undecided
    assert [(b.lang, b.original) for b in split.blocks] == [("es", True), ("es", False)]
    # `lang` names both blocks Spanish; `langs` is what says the Catalan was spoken
    assert delivered.langs == ("es", "ca")
    assert CATALAN in delivered.text and CATALAN not in spanish.text
    assert SPANISH in spanish.text and SPANISH not in delivered.text
    # everything delivered in Spanish stands in both, as it does in any other pair
    assert SPANISH_BODY in delivered.text and SPANISH_BODY in spanish.text
    assert SPANISH_TAIL in delivered.text and SPANISH_TAIL in spanish.text


def test_a_spanish_speech_whose_co_official_passage_was_not_rendered_stays_one_block():
    """The negative control. The same document without the rendering is shape 3, and the
    difference is only whether a rendering of the passage turns up — so this fails if the
    new shape is ever allowed to fire on the passage alone."""
    text = f"{SPANISH_BODY}\n\n{CATALAN}\n\n{SPANISH_TAIL}"
    split = split_languages(text, _fake_detect,
                            _clip(SPANISH_BODY, CATALAN, SPANISH_TAIL),
                            similarity=_pairs((CATALAN[:30], SPANISH[:30])))

    assert split.language == "es"
    assert len(split.blocks) == 1
    assert CATALAN in split.blocks[0].text


def test_a_rendered_passage_with_no_clip_is_flagged():
    # No video published yet, so nothing can say the shortened speech is still deliverable.
    # The reading stands, because leaving the rendering in the record is not the safer
    # answer, but it is marked for the adjudication pass like every other unconfirmed one.
    text = f"{SPANISH_BODY}\n\n{CATALAN}\n\n{SPANISH}\n\n{SPANISH_TAIL}"
    split = split_languages(text, _fake_detect, None,
                            similarity=_pairs((CATALAN[:30], SPANISH[:30])))

    assert split.undecided is True
    assert [(b.lang, b.original) for b in split.blocks] == [("es", True), ("es", False)]


# ---- a claimed rendering in the language it claims to render -------------------------

def _detect_with_galician(text):
    """The fake detector, plus a marker for a language that is neither the speech's nor
    Spanish — the noise a paragraph draws when it is read sentence by sentence."""
    return "gl" if "moito" in text.lower() else _fake_detect(text)


def test_a_paragraph_disputed_by_the_language_it_would_render_is_offered_after_all():
    # 776200: a Catalan paragraph the vote reads as Spanish, claimed as the rendering of
    # the Catalan before it. A Catalan paragraph does not render Catalan, so the claim
    # refutes itself — and left standing it takes the spoken Catalan out of the record
    # while the paragraph that really does render it stays in.
    text = f"{CATALAN}\n\n{CODE_MIXED_CA}\n\n{RENDERS_CODE_MIXED}"
    delivered, spanish = split_languages(
        text, _fake_detect, _clip(CATALAN, CODE_MIXED_CA),
        similarity=_pairs((CATALAN[:30], CODE_MIXED_CA[:30]),
                          (CODE_MIXED_CA[:30], RENDERS_CODE_MIXED[:30]))).blocks

    assert CODE_MIXED_CA in delivered.text
    assert RENDERS_CODE_MIXED not in delivered.text
    assert RENDERS_CODE_MIXED in spanish.text


def test_a_rendering_disputed_by_some_other_language_is_left_alone():
    """The guard this narrows, and the reason it is narrowed rather than dropped. A real
    translation commonly draws a few sentences of a third language, and taking it away
    from the source it explains would refuse a healthy speech to fix a defect it does not
    have. Only a dispute naming the source's OWN language contradicts the claim.

    Sized so that offering it CHANGES the answer: something here would render it, so a
    rule that offered every disputed run would move it into the record of what was said
    and this would fail. Without that, the offer is simply never taken up and the test
    passes whatever the rule does."""
    noisy = ("Y esto, señorías, no lo hace un partido de Estado, se lo digo yo. "
             "Se lo digo con todo el cariño del mundo, de verdad se lo digo. "
             "Moito obrigado, señora presidenta, de verdade llo digo hoxe aquí.")
    spoken_after = (
        "Y añado una última cosa, señorías, que conviene recordar hoy aquí: el "
        "ministerio lleva meses sin contestar al registro y nadie ha dado todavía "
        "ninguna explicación pública de por qué.")
    text = f"{CATALAN}\n\n{noisy}\n\n{spoken_after}"
    delivered, spanish = split_languages(
        text, _detect_with_galician, _clip(CATALAN, spoken_after),
        similarity=_pairs((CATALAN[:30], noisy[:30]),
                          (noisy[:30], spoken_after[:30]))).blocks

    # it stays the rendering it was read as, and stays out of the record of what was said
    assert noisy not in delivered.text
    assert noisy in spanish.text
    # ...and the paragraph that would have rendered it is still a spoken one
    assert spoken_after in delivered.text


# ---- a co-official courtesy line absorbed into a rendering ---------------------------

# 726702's and 750415's shape: the Diario prints the original, then its Spanish rendering,
# and then a co-official farewell. Too short to have a language, it joins the run before it
# — which is the rendering — and leaves the record of what was said along with it.
CO_FAREWELL = "Moltes gràcies."


def test_a_co_official_closing_absorbed_into_a_rendering_stays_in_the_record():
    text = f"{CATALAN}\n\n{SPANISH}\n\n{CO_FAREWELL}"
    split = split_languages(
        text, _fake_detect, _clip(CATALAN, CO_FAREWELL), similarity=_fake_similarity)

    delivered, spanish = split.blocks
    assert CO_FAREWELL in delivered.text, "spoken words left the as-delivered block"
    # The rule is one-sided: it proves the line is not purely a rendering, which says
    # nothing about whether a translation of it exists, so the Spanish block is left exactly
    # as the subtraction found it. The line therefore stands in both — the documented signal
    # for "the alignment found no partner here".
    assert CO_FAREWELL in spanish.text
    assert delivered.text.startswith("La veritat")
    assert spanish.text.startswith("La verdad")


def test_a_spanish_closing_absorbed_into_a_rendering_is_left_alone():
    """The precision half, and the reason the rule keys on spelling rather than position.
    "Muchas gracias." in exactly the same place IS the rendering of a co-official farewell
    embedded in the paragraph before it — confirmed on 20 sampled corpus speeches — so
    rescuing it would duplicate a line nobody said twice."""
    spanish_farewell = "Muchas gracias."
    text = f"{CATALAN}\n\n{SPANISH}\n\n{spanish_farewell}"
    split = split_languages(
        text, _fake_detect, _clip(CATALAN), similarity=_fake_similarity)

    delivered, spanish = split.blocks
    assert spanish_farewell not in delivered.text
    assert spanish_farewell in spanish.text


def test_a_co_official_closing_is_kept_in_the_fourth_shape_too():
    """Half the corpus cases are this shape — a mostly-Spanish speech whose co-official
    passage was rendered, so BOTH blocks are named `es` and the as-delivered one is the only
    place the Catalan survives."""
    text = f"{SPANISH_BODY}\n\n{CATALAN}\n\n{SPANISH}\n\n{CO_FAREWELL}"
    split = split_languages(
        text, _fake_detect, _clip(SPANISH_BODY, CATALAN, CO_FAREWELL),
        similarity=_pairs((CATALAN[:30], SPANISH[:30])))
    delivered, spanish = split.blocks

    assert split.language == "es"
    assert [(b.lang, b.original) for b in split.blocks] == [("es", True), ("es", False)]
    assert CO_FAREWELL in delivered.text
    assert delivered.langs == ("es", "ca")


def test_delivered_spanish_absorbed_into_the_original_stays_in_the_spanish_block():
    """751304's, 726204's and 742429's shape, and the defect that motivated this whole item:
    the speaker opens in Spanish, switches to Catalan, and the Diario renders the Catalan. The
    Spanish opening is too short to have a language, joins the Catalan run, and is subtracted
    from the SPANISH block along with it — so the block that must hold every Spanish word of
    the speech is missing the words actually delivered in Spanish."""
    opening = "Gracias, presidente."
    text = f"{opening}\n\n{CATALAN}\n\n{SPANISH}"
    split = split_languages(
        text, _fake_detect, _clip(opening, CATALAN), similarity=_fake_similarity)

    delivered, spanish = split.blocks
    assert opening in spanish.text, "delivered Spanish left the Spanish block"
    # Mirror of the other rule, and equally one-sided: it was spoken, so it is in both.
    assert opening in delivered.text
    assert CATALAN in delivered.text and CATALAN not in spanish.text


def test_a_co_official_opening_is_not_pushed_into_the_spanish_block():
    """The precision half. A Catalan courtesy line may already have a rendering standing in
    the Spanish block, and adding it would duplicate that line — so a paragraph spelling a
    co-official language is refused here, whatever the detector makes of it."""
    opening = "Moltes gràcies, president."
    text = f"{opening}\n\n{CATALAN}\n\n{SPANISH}"
    split = split_languages(
        text, _fake_detect, _clip(opening, CATALAN), similarity=_fake_similarity)

    assert opening not in split.blocks[1].text
    assert opening in split.blocks[0].text
