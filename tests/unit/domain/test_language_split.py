"""Unit tests for the pure language-split logic.

The detector is injected as a fake keyed on marker words, so these tests are
deterministic and never load py3langid — they exercise only the splitting
algorithm. Real-detector behavior on real data is covered by the speech-extraction
characterization test.
"""

import pytest

from qhld_engine.domain.speeches.language_split import split_languages

pytestmark = pytest.mark.unit


# Galician-only word fragments (none are substrings of their Spanish cognates:
# "silicose"⊄"silicosis", "traballadores"⊄"trabajadores", "quero"⊄"quiero").
_GL_MARKERS = (
    "grazas", "moito", "obrigado", "galego", "quero", "hoxe", "nosa",
    "dereitos", "traballadores", "silicose", "recoñecemento",
)


def _fake_detect(galician_markers=_GL_MARKERS):
    """A detector that tags a sentence Galician if it carries a marker word,
    else Spanish (mirroring the [original gl][castellano es] Diario structure)."""
    def detect(text):
        low = text.lower()
        return "gl" if any(m in low for m in galician_markers) else "es"
    return detect


def test_monolingual_spanish_single_block():
    text = "Muchas gracias, presidente. Comparezco hoy ante la Cámara. Gracias."
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "es"
    assert blocks == [("es", text, True)]


def test_clean_two_block_galician_then_spanish():
    text = (
        "Grazas, señora presidenta, por darme a palabra. "
        "Quero falar do problema da silicose. Moito obrigado. "
        "Muchas gracias, señora presidenta, por la palabra. "
        "Quiero hablar del problema de la silicosis. Muchas gracias."
    )
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "gl"
    assert len(blocks) == 2
    (lang0, original, orig0), (lang1, castellano, orig1) = blocks
    assert (lang0, orig0) == ("gl", True)
    assert (lang1, orig1) == ("es", False)
    assert original.startswith("Grazas")
    assert original.endswith("Moito obrigado.")
    assert castellano.startswith("Muchas gracias")
    assert castellano.endswith("Muchas gracias.")


def test_code_switch_boundary_slip_within_one_sentence():
    # An opening Spanish courtesy aside before switching to Galician: the cut may
    # slip by at most one sentence, but the two languages must stay separated into
    # an original (gl-dominant) block and a Spanish translation block.
    text = (
        "Buenos días a todas y a todos. "
        "Grazas, señora presidenta, por darme a palabra hoxe. "
        "Quero denunciar a situación dos traballadores. Moito obrigado. "
        "Muchas gracias, presidenta, por la palabra. "
        "Quiero denunciar la situación de los trabajadores. Muchas gracias."
    )
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "gl"
    assert len(blocks) == 2
    assert blocks[0][2] is True and blocks[0][0] == "gl"
    assert blocks[1][2] is False and blocks[1][0] == "es"
    # the Galician core lands in the original block, the Spanish core in castellano
    assert "Quero denunciar a situación" in blocks[0][1]
    assert "Quiero denunciar la situación" in blocks[1][1]


def test_no_translation_falls_back_to_single_original_block():
    # An all-Galician speech with no Spanish interpretation → one original block.
    text = (
        "Grazas, señora presidenta. "
        "Quero falar do galego e da nosa lingua. "
        "Defendemos os dereitos. Moito obrigado a todos."
    )
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "gl"
    assert blocks == [("gl", text, True)]


def test_co_official_below_threshold_treated_as_monolingual():
    # A single short Galician aside inside an otherwise Spanish speech stays below
    # the 0.15 co-official ratio → monolingual Spanish, one block.
    text = (
        "Muchas gracias, presidente, por darme la palabra en esta sesión. "
        "Comparezco hoy ante la Cámara para hablar de un asunto importante. "
        "Grazas. "
        "Quiero detallar las cifras del último trimestre con calma. "
        "Es cuanto tengo que decir, muchas gracias a todos ustedes."
    )
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "es"
    assert blocks == [("es", text, True)]


def test_empty_text():
    original_language, blocks = split_languages("   ", _fake_detect())
    assert original_language == "es"
    assert blocks == []


def test_split_preserves_paragraph_breaks_in_both_blocks():
    # The boundary is cut by slicing the text, not re-joining sentences, so the
    # "\n\n" paragraph separators survive inside each language block.
    text = (
        "Grazas, señora presidenta.\n\n"
        "Quero falar do problema da silicose. Moito obrigado.\n\n"
        "Muchas gracias, señora presidenta.\n\n"
        "Quiero hablar del problema de la silicosis. Muchas gracias."
    )
    original_language, blocks = split_languages(text, _fake_detect())

    assert original_language == "gl"
    (_, original, _), (_, castellano, _) = blocks
    assert original == ("Grazas, señora presidenta.\n\n"
                        "Quero falar do problema da silicose. Moito obrigado.")
    assert castellano == (
        "Muchas gracias, señora presidenta.\n\n"
        "Quiero hablar del problema de la silicosis. Muchas gracias.")
