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


def test_build_speaker_regex_hyphen_and_space_interchangeable():
    # Compound surnames get printed with the hyphen dropped ("Grande Marlaska",
    # seen live in DSCD-15-PL-127) or with a space after it from line wrapping.
    regex = _regex("Grande-Marlaska Gómez, Fernando")
    for heading in ("El señor MINISTRO DEL INTERIOR (Grande-Marlaska Gómez):",
                    "El señor MINISTRO DEL INTERIOR (Grande Marlaska Gómez):",
                    "El señor MINISTRO DEL INTERIOR (Grande- Marlaska Gómez):"):
        assert re.search(regex, heading, flags=re.IGNORECASE), heading


def test_build_speaker_regex_matches_shortened_compound_surname():
    # Later turns shorten long compound surnames (seen live: "ÁLVAREZ DE TOLEDO
    # PERALTA-RAMOS" resumes as "ÁLVAREZ DE TOLEDO" in DSCD-15-PL-34, and
    # "(Grande-Marlaska Gómez)" as "(Grande-Marlaska)" in DSCD-15-PL-13).
    regex = _regex("Álvarez de Toledo Peralta-Ramos, Cayetana (GP)")
    assert re.search(regex, "La señora ÁLVAREZ DE TOLEDO PERALTA-RAMOS:",
                     flags=re.IGNORECASE)
    assert re.search(regex, "La señora ÁLVAREZ DE TOLEDO:", flags=re.IGNORECASE)
    regex = _regex("Grande-Marlaska Gómez, Fernando")
    assert re.search(regex, "El señor MINISTRO DE INTERIOR (Grande-Marlaska):",
                     flags=re.IGNORECASE)
    # A short single word left after truncation is too ambiguous to match.
    regex = _regex("Rego Candamil, Néstor (GMx)")
    assert re.search(regex, "El señor REGO CANDAMIL:", flags=re.IGNORECASE)
    assert not re.search(regex, "El señor REGO:", flags=re.IGNORECASE)


def test_build_role_regex_matches_office_heading():
    # Fallback for API records that credit the intervention to someone who never
    # took the floor: the printed heading carries the office, not their surname.
    regex = segmentation.build_role_regex(
        "Ministro de Política Territorial y Memoria Democrática")
    heading = ("El señor MINISTRO DE POLÍTICA TERRITORIAL Y MEMORIA "
               "DEMOCRÁTICA (Torres Pérez):")
    assert re.search(regex, heading, flags=re.IGNORECASE)
    assert segmentation.build_role_regex(None) is None
    assert not re.search(segmentation.build_role_regex("Diputado"),
                         "El señor PEREZ:", flags=re.IGNORECASE)


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


def test_normalize_session_text_without_anchor_keeps_full_text():
    raw = "Sin ancla alguna. El señor PEREZ: hola."
    out = segmentation.normalize_session_text(raw, "161/000123")
    assert out == raw


def test_normalize_ignores_vote_announcement_anchor_after_debate():
    # Voted initiative types (mociones, PNLs) re-print the expediente anchor in
    # the vote announcement at the end of the sitting. The debate anchor, not
    # the vote one, must win — the vote comes after every speech.
    raw = (
        "Sumario (Número de expediente 173/000004). ....... "
        "Otro debate anterior. El señor OTRO: Otra cosa. "
        "MOCIONES (Número de expediente 173/000004). "
        "El señor PEREZ: Defiendo la moción. "
        "La señora GARCIA: Fijamos posición. "
        "La señora PRESIDENTA: Votamos ahora la moción "
        "(Número de expediente 173/000004). Queda aprobada."
    )
    regexes = [_regex("Perez, J (G)"), _regex("Garcia, A (G)")]
    out = segmentation.normalize_session_text(raw, "173/000004", regexes)
    seg = segmentation.SpeechSegmenter(out)
    assert seg.next_speech(regexes[0]) == "Defiendo la moción."
    assert seg.next_speech(regexes[1]) == "Fijamos posición."


def test_normalize_tolerates_zero_padding_typo_in_debate_anchor():
    # Seen live (173/000004 in DSCD-15-PL-21): the debate heading prints the
    # number with an extra zero, so an exact match sees only the vote anchor.
    raw = (
        "MOCIONES (Número de expediente 173/0000004). "
        "El señor PEREZ: Defiendo la moción. "
        "La señora PRESIDENTA: Votamos (Número de expediente 173/000004). Aprobada."
    )
    regexes = [_regex("Perez, J (G)")]
    out = segmentation.normalize_session_text(raw, "173/000004", regexes)
    assert segmentation.SpeechSegmenter(out).next_speech(
        regexes[0]) == "Defiendo la moción."


def test_normalize_ignores_mid_debate_reanchor():
    # The chair sometimes re-prints the anchor mid-debate; the earlier anchor
    # that still yields every speaker must win over the later partial one.
    raw = (
        "DEBATE (Número de expediente 210/000113). "
        "El señor PEREZ: Comparezco. "
        "La señora PRESIDENTA: Continuamos (Número de expediente 210/000113). "
        "La señora GARCIA: Pregunto."
    )
    regexes = [_regex("Perez, J (G)"), _regex("Garcia, A (G)")]
    out = segmentation.normalize_session_text(raw, "210/000113", regexes)
    seg = segmentation.SpeechSegmenter(out)
    assert seg.next_speech(regexes[0]) == "Comparezco."
    assert seg.next_speech(regexes[1]) == "Pregunto."


def test_normalize_tie_breaks_to_latest_anchor():
    # Starting at the summary spans the whole sitting, where a similar heading
    # in someone else's debate can be latched onto (the wrong-text artifact).
    # On equal scores the latest anchor — the debate's own — must win.
    raw = (
        "Sumario (Número de expediente 173/000004). ....... "
        "Debate ajeno. El señor PEREZ: Texto de otro debate. "
        "MOCIONES (Número de expediente 173/000004). "
        "El señor PEREZ: Texto correcto."
    )
    regexes = [_regex("Perez, J (G)")]
    out = segmentation.normalize_session_text(raw, "173/000004", regexes)
    assert segmentation.SpeechSegmenter(out).next_speech(
        regexes[0]) == "Texto correcto."


def test_normalize_without_regexes_uses_last_anchor():
    raw = (
        "Primero (Número de expediente 161/000123). Texto A. "
        "Segundo (Número de expediente 161/000123). Texto B."
    )
    out = segmentation.normalize_session_text(raw, "161/000123")
    assert out.strip() == "Texto B."


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


def test_fix_speaker_typos_does_not_corrupt_correct_occurrences():
    # A typo that is a PREFIX of the correct surname (seen live: "RODRÍGUEZ
    # SALA" for "RODRÍGUEZ SALAS" in DSCD-15-PL-124) must be fixed without
    # rewriting the already-correct headings ("...SALAS" -> "...SALASS").
    text = ("El señor RODRÍGUEZ SALAS: Primer turno. "
            "El señor RODRÍGUEZ SALA: Segundo turno.")
    fixed = segmentation.fix_speaker_typos(text, ["RODRÍGUEZ SALAS"])
    assert "El señor RODRÍGUEZ SALAS: Primer turno." in fixed
    assert "El señor RODRÍGUEZ SALAS: Segundo turno." in fixed
    assert "SALASS" not in fixed


def test_build_speaker_regex_matches_surname_then_role_heading():
    # Inverted government form (seen live: DSCD-15-PL-130): surname first, the
    # office in a parenthetical before the colon.
    regex = _regex("Díaz Pérez, Yolanda")
    heading = ("La señora DÍAZ PÉREZ (vicepresidenta segunda y ministra "
               "de Trabajo y Economía Social):")
    assert re.search(regex, heading, flags=re.IGNORECASE)


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


def test_consecutive_turns_of_same_speaker_not_stitched():
    # Two consecutive interventions by the same speaker look, in the PDF, exactly
    # like a chair interruption with a resume. The lookahead (the next expected
    # intervention is the same speaker) is what keeps them apart; without it the
    # first call swallows both turns and every later speaker falls behind the
    # cursor (seen live in DSCD-15-PL-73, 210/000046).
    text = (
        "El señor MATUTE: Primer turno. "
        "La señora PRESIDENTA: Gracias. "
        "El señor MATUTE: Segundo turno. "
        "La señora GARCIA: Fin."
    )
    matute, garcia = _regex("Matute, O (G)"), _regex("Garcia, A (G)")
    seg = segmentation.SpeechSegmenter(text)
    assert seg.next_speech(matute, matute) == "Primer turno."
    assert seg.next_speech(matute, garcia) == "Segundo turno."
    assert seg.next_speech(garcia, None) == "Fin."


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
