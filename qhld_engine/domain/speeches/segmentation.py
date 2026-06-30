"""Pure speech-segmentation logic — no HTTP, no DB, no logging.

Turns the raw text of a Diario de Sesiones into per-speaker speeches. Everything
here operates on plain strings so it is trivially unit-testable and reusable; the
I/O (intervention API, PDF download, persistence) lives in the application service
that drives this module.

The Diario separates speeches by speaker headings ("El señor X:" / "La señora Y:").
Within a session the initiative's debate is anchored by ``(Número de expediente
<ref>)``. The chair ("presidenta"/"presidente") can interrupt a speaker; when the
same speaker resumes afterwards, the resumed text belongs to the same speech.
"""

import re

from thefuzz import fuzz


# A speaker heading. Two shapes occur in the Diario:
#   - parliamentarians: "El señor PÉREZ GARCÍA:" / "La señora LÓPEZ MORENO:"
#   - government/role:   "La señora MINISTRA DE X, Y Y Z (Saiz Delgado):"
# Group 2 is the name/role; group 3 the optional trailing parenthetical (the
# surname, for government speakers).
#
# Crucially the name/role is matched as UPPERCASE-only (no lowercase letters):
# the Diario prints real speaker names/roles in caps, while a speech that merely
# *mentions* a colleague ("...ese es el señor Tellado, que...") is mixed case. The
# uppercase constraint is what tells the two apart — without it, such a mention
# plus any downstream colon (e.g. a page-header footer "...- D C S D :") forms a
# bogus heading that truncates the speech. MUST be matched WITHOUT re.IGNORECASE,
# or the uppercase character classes would match lowercase too and reopen the bug.
SPEAKER_PATTERN = r"(El señor|La señora)\s+([A-ZÁÀÉÈÍÏÓÒÚÜÇÑ][^:()a-zß-ÿ]*?)(\s*\([^)]*\))?:"

# The chair interrupting a speaker.
INTERRUPTER_PATTERNS = (
    r"(La señora presidenta|El señor presidente):",
    r"(La señora VICEPRESIDENTA|El señor VICEPRESIDENTE)\s+?([a-zá-ú\s+]*[()])*:",
)

# Boilerplate to strip from extracted speech text (stage directions, page
# headers/footers, footnotes). Encodes the Diario layout; tune as it changes.
REMOVABLE_PATTERNS = (
    r"[(]rumores[)]\.?",
    r"[(]aplausos[)]\.?",
    r"\d+ En aplicación del punto Tercero\.7 del Acuerdo de la Mesa del Congreso "
    r"de los Diputados relativo al régimen lingüístico de los debates en los "
    r"órganos parlamentarios\. ",
    r"\d+(\s\d+)*\s-\s?[A-Z\s]+\s-\s\d+(\s\d+)*\s-\s[A-Z\s]+\s[A-Z]+\s[A-Z]+\s?:"
    r"\s?.*?DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS.*?Pág\.\s?\d+\s",
)


def parse_speaker(orador):
    """Split an API ``orador`` into ``(speaker, group, surname)``.

    Two shapes: parliamentarians carry a "(Grupo)" suffix ("Apellido, Nombre
    (Grupo)"); government members do not ("Apellido, Nombre"), in which case
    ``group`` is ``None``. Returns ``(None, None, None)`` only for an empty value."""
    if not orador:
        return None, None, None
    match = re.match(r"(.*?)\s*\((.*)\)\s*$", orador)
    if match:
        speaker, group = match.group(1).strip(), match.group(2).strip()
    else:
        speaker, group = orador.strip(), None
    surname = speaker.split(",")[0].strip()
    return speaker, group, surname


def speaker_surname_upper(orador):
    """The upper-cased surname used to match (and typo-fix) headings in the PDF."""
    name = orador.split("(")[0].strip()
    return name.split(",")[0].strip().upper()


def build_speaker_regex(orador):
    """A regex matching this speaker's heading, built from their surname.

    ``[^:]*?`` between the courtesy title and the surname absorbs any role title
    ("MINISTRA DE …, …") for government speakers, while the optional parens around
    the surname cover both "El señor SURNAME:" and "… (Surname):" forms."""
    surname = orador.split("(")[0].strip().split(",")[0]
    if not surname:
        return None
    surname = re.sub(r"\s+", r"\\s+", surname)
    return rf"(?<!\()(La señora|El señor)\s+[^:]*?\(?{surname}\)?:"


def normalize_session_text(raw, reference):
    """Trim the PDF text to the initiative's debate (the last ``(Número de
    expediente <ref>)`` anchor), collapse whitespace and unify dashes."""
    pattern = rf"\s\(Número\s+de\s+expediente\s{re.escape(reference)}\)\.\s"
    matches = list(re.finditer(pattern, raw))
    if matches:
        raw = raw[matches[-1].end():]
    raw = re.sub(r"\s+", " ", raw)
    return raw.replace("‑", "-").replace("–", "-").replace("—", "-")


def fix_speaker_typos(text, surnames, threshold=80):
    """Repair OCR/transcription typos in speaker surnames against the known list."""
    for match in re.finditer(SPEAKER_PATTERN, text):
        if is_interrupter(match.group(0)):
            continue
        for surname in surnames:
            similarity = fuzz.ratio(match.group(2), surname)
            if threshold < similarity < 100:
                text = text.replace(match.group(2), surname)
                break
    return text


def is_interrupter(heading):
    return any(
        re.fullmatch(pattern, heading, flags=re.IGNORECASE)
        for pattern in INTERRUPTER_PATTERNS
    )


def clean_speech(text):
    for pattern in REMOVABLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


class SpeechSegmenter:
    """Walks a session's text in document order, yielding each speaker's speech.

    A monotonic cursor advances forward only — interventions are processed in
    document order — and is left *at* the next speaker's heading after each call
    (not past it), so the following speaker can still be found.
    """

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def next_speech(self, speaker_regex):
        """The current speaker's speech: their heading up to the next speaker's,
        stitched across chair interruptions when the same speaker resumes. Returns
        ``None`` if the speaker's heading is not found from the cursor onward."""
        if not speaker_regex:
            return None

        start = re.search(speaker_regex, self._text[self._pos:], flags=re.IGNORECASE)
        if not start:
            return None

        speech, next_pos = self._collect(self._pos + start.end(), speaker_regex)
        self._pos = next_pos
        return clean_speech(speech)

    def _collect(self, pos, speaker_regex):
        """Accumulate the speaker's text from ``pos``, returning ``(speech,
        next_pos)`` where ``next_pos`` starts the next speaker's heading (or end of
        text). Chair interruptions are skipped; the speaker's resumed text is kept."""
        speech = ""
        while True:
            heading = re.search(SPEAKER_PATTERN, self._text[pos:])
            if not heading:
                return speech + self._text[pos:], len(self._text)

            speech += self._text[pos:pos + heading.start()]
            heading_pos = pos + heading.start()
            after_heading = pos + heading.end()

            if is_interrupter(heading.group(0)):
                resume = re.search(
                    SPEAKER_PATTERN, self._text[after_heading:])
                if resume and re.fullmatch(
                        speaker_regex, resume.group(0), flags=re.IGNORECASE):
                    # Same speaker resumes: skip the chair's heading + text and the
                    # resumed heading, then keep collecting.
                    pos = after_heading + resume.end()
                    continue
                # Interrupted and not resumed: speech ends; the next intervention
                # resumes from the interrupter's heading.
                return speech, heading_pos

            # A different speaker: speech ends; next intervention starts here.
            return speech, heading_pos
