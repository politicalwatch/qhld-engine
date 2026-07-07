"""Resolve raw NER person-spans to canonical people — pure, no I/O.

NER (``NerPort``) yields person
spans verbatim from a speech's Spanish text ("Sánchez", "el señor Sánchez",
"Pedro Sánchez"); this module normalizes them and fuzzy-matches each against a
person catalog (sitting deputies plus non-deputies — government ministers, the
King, regional presidents, foreign leaders), collapsing the many surface forms of
one person into a single ``Mention`` with an occurrence count.

The catalog is a flat list of ``PersonEntry`` — deputies and non-deputies scored
in ONE pass. A few non-deputies share a surname with a deputy ("Clavijo" = the
Canarias president vs the deputy Gamarra Ruiz-Clavijo's second surname); such
entries carry ``overrides_deputy`` so that, when tied with a deputy, they win.

Matching reuses the ``thefuzz.token_set_ratio`` + high-threshold trick the query
``EntityResolver`` relies on: ``token_set_ratio`` scores a subset match ~100
("sánchez" ⊆ "sánchez pérez-castejón, pedro") while unrelated names stay low, so
a surname alone resolves but noise does not. Because bare surnames collide across
deputies ("García"), an **ambiguity guard** drops any span whose top score is
shared by two or more deputies — favouring a missed mention over a wrong one.

Kept pure (takes a prebuilt index, returns ``Mention`` objects) so it is unit-
testable offline with no Mongo, mirroring ``domain.speeches.segmentation``.
"""

import re
from collections import Counter
from dataclasses import dataclass

from thefuzz import fuzz

from tipi_data.models.speech import Mention

# Courtesy honorifics and articles spaCy often folds into a PER span. Stripped
# before matching so "el señor Sánchez" resolves like "Sánchez". Kept deliberately
# small: role words (ministro/presidente…) are left in — token_set_ratio tolerates
# the extra token, whereas over-stripping risks matching a bare title to a name.
_HONORIFICS = {
    "el", "la", "los", "las", "un", "una", "al",
    "sr", "sra", "srs", "sras", "señor", "señora", "señores", "señoras",
    "don", "doña", "su", "señoria", "señorias", "señoría", "señorías",
    "excelentisimo", "excelentisima", "excelentísimo", "excelentísima",
    "ilustrisimo", "ilustrisima", "ilustrísimo", "ilustrísima",
}
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_MIN_LEN = 3

# Common words spaCy sometimes tags as a person because they are also borne as a
# surname (typically sentence-initial "Bueno, …"). They never name a real person.
# (Famous non-deputies who share a surname with a deputy — Aznar, Suárez, Clavijo —
# are no longer dropped here: they now resolve to their own catalog entry, which
# wins the tie via ``overrides_deputy``.)
COMMON_WORD_SURNAMES = frozenset({"bueno"})


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens, split on whitespace and hyphens, keeping those long
    enough to discriminate (so "de"/"la" are ignored)."""
    return {t for t in re.split(r"[\s-]+", text.lower()) if len(t) >= _MIN_LEN}


def _first_surname_tokens(name: str) -> set[str]:
    """The tokens of a deputy's FIRST surname (the first whitespace element before the
    comma; hyphenated compounds like "Grande-Marlaska" split into both parts) — the
    core of their identity, used to tell a real mention from a homonym."""
    surname_part = name.partition(",")[0]
    first = surname_part.split()[0] if surname_part.split() else ""
    return _tokens(first)


# Offices a sitting deputy would not simultaneously hold: when the text introduces a
# person by one of these, the person named is not the deputy who happens to share the
# surname. Scanned per speech so the exclusion is scoped to that speech (a cue anywhere
# in it disqualifies the surname throughout — e.g. "expresidente Aznar" ⇒ every "Aznar").
_CONTEXT_CUE_PATTERNS = (
    # "<Apellido>, (actual) magistrado" / "magistrado(a) <Apellido>"
    r"([a-zá-úñ]+),?\s+(?:actual\s+)?magistrad[oa]",
    r"magistrad[oa]\s+(?:del?\s+\w+\s+)*([a-zá-úñ]+)",
    # "(el) juez/jueza <Apellido>", "(el) fiscal (general) <Apellido>"
    r"jueza?\s+([a-zá-úñ]+)",
    r"fiscal(?:\s+general)?\s+([a-zá-úñ]+)",
    # NB: no "expresidente(a) del Gobierno <Apellido>" cue — its only job was to keep
    # an ex-PM (Aznar, Zapatero) from resolving to a colliding deputy; the ex-PMs now
    # live in the person catalog and resolve there instead of being dropped.
)
# Franco is uniquely the dictator whenever the speech invokes the dictatorship; the
# deputy surnamed Franco is meant only absent that framing (hence a cue, not a denylist).
_DICTATORSHIP_CUE = re.compile(r"dictadur|dictador|franquism|r[ée]gimen\s+de\s+franco")


def context_excluded_surnames(text: str) -> frozenset[str]:
    """Surnames the speech's own wording marks as non-deputies (a magistrate, judge,
    prosecutor, former head of government, or Franco-the-dictator). Speech-scoped."""
    low = (text or "").lower()
    excluded = set()
    if _DICTATORSHIP_CUE.search(low):
        excluded.add("franco")
    for pattern in _CONTEXT_CUE_PATTERNS:
        for match in re.finditer(pattern, low):
            excluded.add(match.group(1))
    return frozenset(excluded)


@dataclass(frozen=True)
class PersonEntry:
    """One person's canonical identity plus the lowercased keys a span is scored
    against (full "apellido, nombre", "nombre apellido", the bare surname, and any
    explicit aliases). ``person_type`` is ``"deputy"`` for catalog deputies, else the
    non-deputy kind. ``overrides_deputy`` marks a non-deputy who should win a tie
    against a deputy sharing the surname (e.g. "Clavijo")."""
    person_id: str
    person_type: str
    name: str
    keys: tuple[str, ...]
    overrides_deputy: bool = False


def _name_keys(name: str) -> set[str]:
    """Match keys derived from an "Apellido(s), Nombre" name: the whole string, the
    bare first surname group, and the "Nombre Apellido" order."""
    keys = {name.lower(), name.split(",")[0].strip().lower()}
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 2 and parts[0] and parts[1]:
        keys.add(f"{parts[1]} {parts[0]}".lower())
    return {k for k in keys if k}


def make_person_entry(person_id, person_type, name, aliases=(), overrides_deputy=False):
    """Build a non-deputy ``PersonEntry``. Keys come from the canonical ``name`` plus
    any ``aliases`` (nicknames, bare surname, role phrases like "su majestad"), each
    run through ``normalize_span`` so they match under the same normalization the
    corpus spans get."""
    keys = _name_keys(name)
    for alias in aliases:
        norm = normalize_span(alias)
        if norm:
            keys.add(norm)
    return PersonEntry(
        person_id=person_id, person_type=person_type, name=name,
        keys=tuple(sorted(keys)), overrides_deputy=overrides_deputy)


def build_deputy_index(deputies) -> list[PersonEntry]:
    """Build match entries from ``Deputy`` records (``name`` = 'Apellido, Nombre',
    ``get_fullname()`` = 'Nombre Apellido'). Deputies without a name are skipped."""
    index = []
    for deputy in deputies:
        name = getattr(deputy, "name", None)
        if not name:
            continue
        keys = {name.lower(), name.split(",")[0].strip().lower()}
        try:
            keys.add(deputy.get_fullname().lower())
        except (AttributeError, IndexError):
            pass
        index.append(PersonEntry(
            person_id=deputy.id, person_type="deputy", name=name,
            keys=tuple(k for k in keys if k), overrides_deputy=False))
    return index


def build_person_index(deputies, extra=()) -> list[PersonEntry]:
    """The full match index: every deputy plus ``extra`` non-deputy ``PersonEntry``
    rows (curated catalog + corpus-bootstrapped speakers, assembled at the
    application layer). Scored together in one pass by the resolver."""
    return build_deputy_index(deputies) + list(extra)


def build_surname_gazetteer(deputies) -> list[str]:
    """Distinctive first-surname surfaces (each borne by exactly one deputy) to seed an
    NER gazetteer, so the model also tags the uncommon/compound surnames it otherwise
    misses. Hyphenated compounds contribute each part ("Grande-Marlaska" → "Grande",
    "Marlaska"). Surnames shared by several deputies are left out: the base model
    usually catches common ones, and they would only add ambiguous spans the resolver
    drops anyway. Original casing is kept (names are Title-case in the Diario text)."""
    counts: Counter[str] = Counter()
    surface: dict[str, str] = {}
    for deputy in deputies:
        name = getattr(deputy, "name", None)
        surname_part = name.partition(",")[0] if name else ""
        if not surname_part.split():
            continue
        for token in re.split(r"[-\s]+", surname_part.split()[0]):
            key = token.lower()
            if len(key) >= _MIN_LEN:
                counts[key] += 1
                surface.setdefault(key, token)
    return sorted(surface[key] for key, count in counts.items() if count == 1)


def normalize_span(span: str) -> str:
    """Lowercase, drop punctuation and courtesy honorifics/articles. Returns the
    residual name, or "" when nothing usable remains (e.g. "Su Señoría")."""
    cleaned = _PUNCT_RE.sub(" ", span.lower())
    tokens = [t for t in cleaned.split() if t and t not in _HONORIFICS]
    residual = " ".join(tokens)
    return residual if len(residual) >= _MIN_LEN else ""


def _break_tie(norm: str, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Narrow equally-scored people (``token_set_ratio`` gives a bare surname 100
    against everyone who carries it anywhere) to the one the span actually names:

    1. Prefer deputies whose FIRST surname the span matches — a surname resolves to
       whoever bears it first, not to someone who has it as a second surname or given
       name ("Bravo" → Juan Bravo, not Aitor Esteban Bravo).
    2. If several still qualify, prefer the closest exact-order match
       (``token_sort_ratio``), which separates a partial multi-token overlap from the
       real full surname ("Sánchez Pérez-Castejón" → Pedro, not Sánchez Pérez, César).

    Returns the surviving candidates; a still-tied result (e.g. two deputies sharing a
    first surname) is left for the caller to drop as genuinely ambiguous."""
    span_tokens = _tokens(norm)
    first = [e for e in tied if span_tokens & _first_surname_tokens(e.name)]
    candidates = first if first else tied
    if len(candidates) == 1:
        return candidates
    # Only a fuller, multi-token reference can separate several deputies who share a
    # first surname; a bare surname borne by many stays ambiguous (caller drops it).
    if len(span_tokens) < 2:
        return candidates
    scored = [(max(fuzz.token_sort_ratio(norm, key) for key in e.keys), e)
              for e in candidates]
    top = max(score for score, _ in scored)
    return [e for score, e in scored if score == top]


def resolve_person(name: str, index: list[PersonEntry], threshold: int) -> PersonEntry | None:
    """Resolve a free-text person name (as typed in a search query, e.g. "Zapatero",
    "María Jesús Montero", "Ayuso") to a catalog person, or ``None`` if it does not clear
    the threshold or stays ambiguous. Runs the span through the SAME normalization + fuzzy
    match + ambiguity guard used to tag the corpus, so a query resolves consistently with
    what was indexed."""
    norm = normalize_span(name)
    if not norm:
        return None
    return _resolve_one(norm, index, threshold)


def _resolve_one(norm: str, index: list[PersonEntry], threshold: int):
    """Best-scoring person for a normalized span, or ``None`` when nothing clears the
    threshold or the top score stays shared after tie-breaking (ambiguous surname)."""
    best_score = 0
    tied: list[PersonEntry] = []
    for entry in index:
        score = max(fuzz.token_set_ratio(norm, key) for key in entry.keys)
        if score > best_score:
            best_score, tied = score, [entry]
        elif score == best_score and best_score > 0:
            tied.append(entry)
    if best_score < threshold:
        return None
    if len(tied) > 1:
        # An override only applies when the span names the override's OWN first surname —
        # not when it merely shares a secondary token (the ex-PM "Aznar López" must not
        # hijack a bare "López" tie via his second surname).
        span_tokens = _tokens(norm)
        named_override = any(
            e.overrides_deputy and (_first_surname_tokens(e.name) & span_tokens)
            for e in tied)
        if named_override:
            # A famous non-deputy tied with the deputy who merely shares the surname
            # wins: "Clavijo" is the Canarias president, not the deputy Gamarra
            # Ruiz-Clavijo. But only for a bare-surname reference — the ORDER-sensitive
            # score separates a fuller deputy match ("Gamarra Ruiz-Clavijo" → the deputy)
            # from a plain surname ("Clavijo" → the president).
            tied = _prefer_overrides(norm, tied)
        elif any(e.person_type == "deputy" for e in tied):
            # Deputies are the primary referents in the chamber: a non-deputy (a
            # bootstrapped minister, a non-override curated figure) never blocks or
            # steals a deputy resolution on a shared-surname tie ("Rego" → the deputy
            # Rego Candamil, not the minister Sira Rego). This keeps the deputy metric
            # identical to the deputies-only baseline.
            tied = [e for e in tied if e.person_type == "deputy"]
    if len(tied) > 1:
        tied = _break_tie(norm, tied)
    return tied[0] if len(tied) == 1 else None


def _prefer_overrides(norm: str, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Resolve an override-vs-deputy tie by order-sensitive match: keep the entries the
    span matches most exactly (``token_sort_ratio``), then, within that group, keep the
    override(s) if any. So "Clavijo"/"Aznar" (bare surname) go to the non-deputy, while
    "Gamarra Ruiz-Clavijo" (the deputy's own full name) stays with the deputy."""
    scored = [(max(fuzz.token_sort_ratio(norm, key) for key in e.keys), e) for e in tied]
    top = max(score for score, _ in scored)
    best = [e for score, e in scored if score == top]
    overrides = [e for e in best if e.overrides_deputy]
    return overrides if overrides else best


def _is_excluded(norm: str, entry: PersonEntry, excluded: frozenset[str]) -> bool:
    """Whether a resolved span actually names a flagged non-deputy rather than the
    deputy it fuzzy-matched. Two cases:

    - *referent-homonym* — the deputy's OWN first surname is flagged (the ex-PM Aznar
      vs the deputy Aznar Teruel): the surname coincides, so drop it.
    - *mismatch* — the span carries a flagged token but resolved via a secondary
      surname (the Canarias president "Clavijo" fuzzy-matching Gamarra Ruiz-Clavijo):
      drop only when the deputy's first surname is absent from the span, so a genuine
      full-name mention ("Gamarra Ruiz-Clavijo") that merely contains the token survives.
    """
    if not excluded:
        return False
    span_tokens = _tokens(norm)
    if not (span_tokens & excluded):
        return False
    first = _first_surname_tokens(entry.name)
    if first & excluded:
        return True
    return not (span_tokens & first)


def resolve_mentions(
    spans, index: list[PersonEntry], threshold: int,
    excluded_surnames: frozenset[str] = frozenset()) -> list[Mention]:
    """Collapse raw NER ``spans`` (duplicates preserved) into canonical ``Mention``s.

    Each span is normalized then resolved (cached per normalized form). Occurrences
    that resolve to the same person are merged: ``count`` totals them and
    ``surface_forms`` keeps the distinct raw spans seen. ``excluded_surnames`` drops
    spans that name a flagged non-deputy homonym of a DEPUTY (see ``_is_excluded``);
    it never touches a resolved non-deputy. Returns mentions ordered by descending
    count then name."""
    cache: dict[str, PersonEntry | None] = {}
    by_person: dict[str, dict] = {}
    for span in spans:
        norm = normalize_span(span)
        if not norm:
            continue
        if norm not in cache:
            cache[norm] = _resolve_one(norm, index, threshold)
        entry = cache[norm]
        if entry is None:
            continue
        # The homonym denylist / speech-scoped cues exist only to stop a famous
        # non-deputy being mistaken for a deputy — so they gate deputy resolutions
        # only. A resolved non-deputy is exactly who we want and is never excluded.
        if entry.person_type == "deputy" and _is_excluded(norm, entry, excluded_surnames):
            continue
        acc = by_person.setdefault(
            entry.person_id,
            {"name": entry.name, "person_type": entry.person_type,
             "surface_forms": set(), "count": 0})
        acc["surface_forms"].add(span.strip())
        acc["count"] += 1

    mentions = [
        Mention(
            person_id=person_id,
            person_type=acc["person_type"],
            name=acc["name"],
            surface_forms=sorted(acc["surface_forms"]),
            count=acc["count"])
        for person_id, acc in by_person.items()
    ]
    mentions.sort(key=lambda m: (-m.count, m.name))
    return mentions
