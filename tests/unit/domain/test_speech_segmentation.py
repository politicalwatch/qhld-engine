"""Unit tests for the pure speech-segmentation domain logic — no HTTP, no DB.

Covers speaker parsing, the per-surname heading regex, session-text normalisation,
fuzzy typo-fixing, boilerplate cleanup, and the ``SpeechSegmenter`` cursor that
slices a Diario de Sesiones into per-speaker speeches (including stitching across
chair interruptions).
"""

import re

import pytest

from qhld_engine.domain.speeches import segmentation

pytestmark = pytest.mark.unit


def _regex(orador):
    return segmentation.build_speaker_regex(orador)


# --- parsing / helpers --------------------------------------------------------

def test_parse_speaker():
    speaker, group, surname = segmentation.parse_speaker(
        "Pérez Gómez, Juan (GP Socialista)")
    assert speaker == "Pérez Gómez, Juan"
    assert group == "GP Socialista"
    assert surname == "Pérez Gómez"


def test_parse_speaker_no_group_is_government_member():
    # Ministers/government members come from the API with no "(Grupo)" suffix.
    speaker, group, surname = segmentation.parse_speaker("Saiz Delgado, Elma")
    assert speaker == "Saiz Delgado, Elma"
    assert group is None
    assert surname == "Saiz Delgado"


def test_parse_speaker_empty_returns_nones():
    assert segmentation.parse_speaker("") == (None, None, None)


def test_build_speaker_regex_matches_role_based_government_heading():
    # The Diario heading for a minister is role-based with the surname in parens.
    regex = _regex("Saiz Delgado, Elma")
    heading = ("La señora MINISTRA DE INCLUSIÓN, SEGURIDAD SOCIAL Y "
               "MIGRACIONES (Saiz Delgado):")
    assert re.search(regex, heading, flags=re.IGNORECASE)


def test_government_speaker_is_bounded_not_swallowed():
    # diputado -> minister -> diputado, with no chair in between: each turn must be
    # sliced at the next heading, including the role-based ministerial heading.
    text = (
        "El señor REGO CANDAMIL: Discurso del diputado. "
        "La señora MINISTRA DE INCLUSIÓN, SEGURIDAD SOCIAL Y MIGRACIONES "
        "(Saiz Delgado): Respuesta de la ministra. "
        "El señor REGO CANDAMIL: Réplica del diputado."
    )
    seg = segmentation.SpeechSegmenter(text)
    assert seg.next_speech(_regex("Rego Candamil, Néstor (GMx)")) == "Discurso del diputado."
    assert seg.next_speech(_regex("Saiz Delgado, Elma")) == "Respuesta de la ministra."
    assert seg.next_speech(_regex("Rego Candamil, Néstor (GMx)")) == "Réplica del diputado."


def test_speaker_surname_upper():
    assert segmentation.speaker_surname_upper(
        "García, Ana (GP Popular)") == "GARCÍA"


def test_build_speaker_regex_matches_heading_case_insensitively():
    regex = _regex("Perez, Juan (GP)")
    assert re.search(regex, "El señor PEREZ:", flags=re.IGNORECASE)
    assert re.search(regex, "La señora GARCIA:", flags=re.IGNORECASE) is None


def test_normalize_session_text_trims_to_expediente():
    raw = ("Algo previo irrelevante (Número de expediente 161/000123). "
           "El señor PEREZ: hola.")
    out = segmentation.normalize_session_text(raw, "161/000123")
    assert "previo" not in out
    assert out.strip().startswith("El señor PEREZ")


def test_clean_speech_removes_stage_directions():
    cleaned = segmentation.clean_speech("Hola (Rumores). Adiós (Aplausos).  ")
    assert "Rumores" not in cleaned
    assert "Aplausos" not in cleaned
    assert cleaned.startswith("Hola")


def test_fix_speaker_typos_corrects_near_match():
    # Diario headings are uppercase; a realistic transcription typo (here a
    # dropped accent) stays uppercase and is normalised to the known surname.
    fixed = segmentation.fix_speaker_typos("La señora GARCIA: Hola.", ["GARCÍA"])
    assert "GARCÍA" in fixed
    assert "señora GARCIA:" not in fixed


# --- SpeechSegmenter ----------------------------------------------------------

def test_two_consecutive_speakers_each_extracted():
    text = "El señor PEREZ: Primer discurso. La señora GARCIA: Segundo discurso."
    seg = segmentation.SpeechSegmenter(text)
    assert seg.next_speech(_regex("Perez, J (G)")) == "Primer discurso."
    assert seg.next_speech(_regex("Garcia, A (G)")) == "Segundo discurso."


def test_chair_interruption_with_resume_is_stitched():
    text = (
        "El señor PEREZ: Parte uno. "
        "La señora PRESIDENTA: Silencio, por favor. "
        "El señor PEREZ: Parte dos. "
        "La señora GARCIA: Fin."
    )
    seg = segmentation.SpeechSegmenter(text)

    speech = seg.next_speech(_regex("Perez, J (G)"))
    assert "Parte uno." in speech
    assert "Parte dos." in speech
    assert "Silencio" not in speech  # the chair's words are not the speaker's
    # the cursor is left at García's heading, so she can still be extracted
    assert seg.next_speech(_regex("Garcia, A (G)")) == "Fin."


def test_chair_interruption_without_resume_ends_speech():
    text = (
        "El señor PEREZ: Solo una parte. "
        "La señora PRESIDENTA: Se levanta la sesión. "
        "La señora GARCIA: Otra cosa."
    )
    seg = segmentation.SpeechSegmenter(text)
    speech = seg.next_speech(_regex("Perez, J (G)"))
    assert speech == "Solo una parte."
    assert "Se levanta" not in speech


def test_missing_speaker_heading_returns_none():
    seg = segmentation.SpeechSegmenter("La señora GARCIA: Hola.")
    assert seg.next_speech(_regex("Perez, J (G)")) is None


def test_mid_speech_colleague_mention_does_not_truncate():
    # A speaker naming a colleague mid-sentence ("...el señor Tellado...") followed
    # later by a stray colon (here a page-header footer) must NOT be read as a new
    # speaker heading. Mentions are mixed-case; real headings are uppercase. Without
    # the uppercase constraint this truncated the speech at the mention.
    text = (
        "El señor PEREZ: Sigo con mi argumento. Ese es el señor Tellado, ese es "
        "el Partido Popular, al que los intereses de la gente y "
        "3 1 - L P : DIARIO DE SESIONES no le importan nada. Y concluyo aquí. "
        "La señora GARCIA: Mi turno."
    )
    seg = segmentation.SpeechSegmenter(text)
    speech = seg.next_speech(_regex("Perez, J (G)"))
    assert "el señor Tellado" in speech
    assert "Y concluyo aquí." in speech  # not cut off at the mention
    assert "Mi turno" not in speech       # and it stops at the real next heading
    assert seg.next_speech(_regex("Garcia, A (G)")) == "Mi turno."
