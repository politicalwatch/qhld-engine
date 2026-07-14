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
# Newlines are excluded throughout: the normalized text keeps paragraph breaks as
# "\n\n" and a real heading never spans one.
SPEAKER_PATTERN = (r"(El señor|La señora) "
                   r"([A-ZÁÀÉÈÍÏÓÒÚÜÇÑ][^:()a-zß-ÿ\n]*?)( ?\([^)\n]*\))?:")

# The chair interrupting a speaker.
INTERRUPTER_PATTERNS = (
    r"(La señora presidenta|El señor presidente):",
    r"(La señora VICEPRESIDENTA|El señor VICEPRESIDENTE)\s+?([a-zá-ú\s+]*[()])*:",
)

# Boilerplate to strip from extracted speech text (stage directions, footnotes).
# Encodes the Diario layout; tune as it changes.
REMOVABLE_PATTERNS = (
    r"[(]rumores[)]\.?",
    r"[(]aplausos[)]\.?",
    r"\d+ En aplicación del punto Tercero\.7 del Acuerdo de la Mesa del Congreso "
    r"de los Diputados relativo al régimen lingüístico de los debates en los "
    r"órganos parlamentarios\.\s?",
)

# The margin apparatus pdfminer emits wherever a page break falls (possibly
# mid-sentence): the "cve: DSCD-…" code printed vertically (one character per
# line), then the running header down to the page number. Removed from the raw
# text before paragraphs are reconstructed, so a page turn never fabricates a
# paragraph break.
PAGE_JUNK_PATTERN = re.compile(
    r"(?:\n[^\n]{0,3})*"
    r"\n[^\S\n]*DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
    r"(?:[^\n]*\n)+?"
    r"[^\S\n]*Pág\.\s?\d+[^\n]*\s*"
)

# A line break marks a paragraph boundary only when the line ends a sentence:
# in the pdfminer output wrapped lines keep a trailing space while
# paragraph-final lines end flush at their punctuation, and the next paragraph
# opens with an uppercase/opening character.
PARAGRAPH_END_CHARS = ".!?…»:\"”)"
# Digits deliberately excluded: "art.\n5" is a wrapped citation, not a new
# paragraph, and speech paragraphs opening with a bare number are rare.
PARAGRAPH_START_PATTERN = re.compile(r"[A-ZÁÀÉÈÍÏÓÒÚÜÇÑ¿¡«\"“—-]")


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
    the surname cover both "El señor SURNAME:" and "… (Surname):" forms. Hyphens
    and spaces are interchangeable: compound surnames get printed with the hyphen
    dropped or with a space after it when wrapped across lines.

    Long compound surnames also get *shortened* on a speaker's later turns
    ("ÁLVAREZ DE TOLEDO PERALTA-RAMOS" resumes as "ÁLVAREZ DE TOLEDO"), so the
    surname minus its last word is matched as an alternative — inside the same
    regex, not as a retry, so whichever form occurs first from the cursor wins
    (a retry would let the full form latch onto a later turn first). The colon
    required right after the surname keeps the short form from matching the
    long one's prefix."""
    surname = orador.split("(")[0].strip().split(",")[0]
    if not surname:
        return None
    variants = [surname]
    words = surname.split()[:-1]
    while words and words[-1].islower():  # drop dangling connectors ("de", "i")
        words.pop()
    shortened = " ".join(words)
    if len(re.split(r"[-\s]+", shortened)) >= 2:  # one word alone is too ambiguous
        variants.append(shortened)
    variants = [re.sub(r"[-\s]+", r"[- ]+", v) for v in variants]
    alternatives = "|".join(rf"\(?{v}\)?" for v in variants)
    # The optional parenthetical before the colon covers the inverted government
    # form "DÍAZ PÉREZ (vicepresidenta segunda y ministra de …):". Newlines are
    # excluded so a heading match never spans a paragraph break.
    return (rf"(?<!\()(La señora|El señor) [^:\n]*?(?:{alternatives})"
            rf"( ?\([^)\n]*\))?:")


def build_role_regex(role):
    """A regex matching a government speaker's heading from their office
    (``cargo_orador``), e.g. "Ministro de X" -> "El señor MINISTRO DE X (Y):".

    Fallback for interventions whose API ``orador`` names someone other than
    the person who actually took the floor (seen with interpelaciones answered
    on the Government's behalf), where no surname-based heading exists."""
    if not role:
        return None
    role = re.sub(r"\s+", " ", role.strip().upper())
    return rf"(La señora|El señor) {re.escape(role)} ?\([^)\n]*\):"


def flatten_paragraphs(raw):
    """Collapse the raw PDF text to single-spaced paragraphs joined by ``\\n\\n``.

    In the pdfminer output a paragraph break and a mere line wrap both come out
    as newlines — and how many (one, or a blank line) varies by document vintage:
    some Diarios interleave blank lines mid-sentence, even inside a speaker
    heading. What *is* reliable is how the previous line ends: wrapped lines
    keep a trailing space (justified text), while a paragraph-final line ends
    flush at its sentence punctuation, with the next paragraph opening
    upper-case. So every line boundary, blank or not, is judged by that rule.

    Page-boundary junk is stripped first (see ``PAGE_JUNK_PATTERN``) so a page
    turn joins back per the same rules instead of fabricating a break."""
    text = PAGE_JUNK_PATTERN.sub("\n", raw)
    paragraphs = []
    current = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if current and _ends_paragraph(current[-1], line):
            paragraphs.append(current)
            current = []
        current.append(line)
    if current:
        paragraphs.append(current)
    # [^\S\n] = any whitespace but a newline, so non-breaking spaces and tabs
    # collapse along with plain spaces.
    return "\n\n".join(
        re.sub(r"[^\S\n]+", " ", " ".join(p)).strip() for p in paragraphs)


def _ends_paragraph(prev, line):
    """Whether the bare newline between ``prev`` and ``line`` is a paragraph
    boundary rather than a line wrap."""
    if prev != prev.rstrip():
        return False
    return (prev[-1] in PARAGRAPH_END_CHARS
            and bool(PARAGRAPH_START_PATTERN.match(line.lstrip())))


def normalize_session_text(raw, reference, speaker_regexes=()):
    """Collapse whitespace to single-spaced ``\\n\\n``-separated paragraphs,
    unify dashes and trim the PDF text to the initiative's debate.

    The Diario prints ``(Número de expediente <ref>)`` in several places: the
    opening summary, the debate's own section heading, and — for initiative
    types that get voted — again in the vote announcement at the end of the
    sitting, occasionally with extra zero-padding in the number. Which
    occurrence starts the debate cannot be told from position alone, so each
    candidate is trial-segmented against the interventions' ``speaker_regexes``
    and the one yielding the most speeches wins. Ties go to the latest
    position: a window that starts at the summary spans the whole sitting and
    can latch onto a similar heading in someone else's debate.

    Without ``speaker_regexes`` the last occurrence is used."""
    flat = flatten_paragraphs(raw)
    flat = flat.replace("‑", "-").replace("–", "-").replace("—", "-")
    candidates = _anchor_candidates(flat, reference)
    if not candidates:
        return flat
    if not speaker_regexes:
        return flat[candidates[-1]:]
    best = max(
        candidates,
        key=lambda start: (_segmentable(flat[start:], speaker_regexes), start),
    )
    return flat[best:]


def _anchor_candidates(text, reference):
    """End positions of every ``(Número de expediente <ref>)`` occurrence,
    tolerating zero-padding differences in the printed number."""
    try:
        initiative_type, number = reference.split("/")
        printed = rf"{re.escape(initiative_type)}/0*{int(number)}"
    except ValueError:
        printed = re.escape(reference)
    pattern = rf"\(Número\s+de\s+expediente\s+{printed}\)\.?\s*"
    return [match.end() for match in re.finditer(pattern, text)]


def _segmentable(text, speaker_regexes):
    """How many of the debate's interventions, taken in order, can be segmented
    out of ``text`` — the trial score used to pick the debate's anchor."""
    segmenter = SpeechSegmenter(text)
    regexes = list(speaker_regexes)
    return sum(
        1 for regex, upcoming in zip(regexes, regexes[1:] + [None])
        if segmenter.next_speech(regex, upcoming) is not None
    )


def fix_speaker_typos(text, surnames, threshold=80):
    """Repair OCR/transcription typos in speaker surnames against the known list.

    The replacement is bounded to whole uppercase words: a typo that is a prefix
    of the correct surname ("RODRÍGUEZ SALA" for "RODRÍGUEZ SALAS") must not
    rewrite the correct occurrences too, or they stop matching their heading."""
    for match in re.finditer(SPEAKER_PATTERN, text):
        if is_interrupter(match.group(0)):
            continue
        for surname in surnames:
            similarity = fuzz.ratio(match.group(2), surname)
            if threshold < similarity < 100:
                text = re.sub(
                    rf"(?<![A-ZÁÀÉÈÍÏÓÒÚÜÇÑ]){re.escape(match.group(2))}"
                    rf"(?![A-ZÁÀÉÈÍÏÓÒÚÜÇÑ])",
                    surname, text)
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
    # Tidy what the removals (and interruption stitching) leave behind: doubled
    # spaces, spaces hugging a paragraph break, paragraphs emptied entirely.
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
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

    def next_speech(self, speaker_regex, upcoming_regex=None):
        """The current speaker's speech: their heading up to the next speaker's,
        stitched across chair interruptions when the same speaker resumes. Returns
        ``None`` if the speaker's heading is not found from the cursor onward.

        ``upcoming_regex`` is the *next* intervention's speaker. When a speaker
        holds two consecutive interventions, the PDF shows the same
        heading/chair/heading shape as a mere interruption; the lookahead is what
        tells them apart, so the resumed text is left for the next intervention
        instead of being stitched into this one."""
        if not speaker_regex:
            return None

        start = re.search(speaker_regex, self._text[self._pos:], flags=re.IGNORECASE)
        if not start:
            return None

        speech, next_pos = self._collect(
            self._pos + start.end(), speaker_regex, upcoming_regex)
        self._pos = next_pos
        return clean_speech(speech)

    def _collect(self, pos, speaker_regex, upcoming_regex):
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
                if (resume
                        and re.fullmatch(
                            speaker_regex, resume.group(0), flags=re.IGNORECASE)
                        and not (upcoming_regex and re.fullmatch(
                            upcoming_regex, resume.group(0), flags=re.IGNORECASE))):
                    # Same speaker resumes: skip the chair's heading + text and the
                    # resumed heading, then keep collecting.
                    pos = after_heading + resume.end()
                    continue
                # Interrupted and not resumed (or the resumed heading is the next
                # intervention): speech ends; the next intervention resumes from
                # the interrupter's heading.
                return speech, heading_pos

            # A different speaker: speech ends; next intervention starts here.
            return speech, heading_pos
