"""Resolve raw NER person-spans to canonical deputies — pure, no I/O.

NER (``NerPort``) yields person
spans verbatim from a speech's Spanish text ("Sánchez", "el señor Sánchez",
"Pedro Sánchez"); this module normalizes them and fuzzy-matches each against the
deputies catalog, collapsing the many surface forms of one person into a single
``Mention`` with an occurrence count.

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

# Surnames that fuzzy-match a real deputy but, in this national-Congress corpus, denote
# a famous non-deputy who happens to share the surname. Every entry MUST collide with a
# deputy — otherwise it is dead weight: a surname with no deputy homonym (a foreign
# leader, a non-colliding ex-president) matches nothing and is already dropped by the
# match threshold, and a surname several deputies share is dropped by the ambiguity
# guard. So only DISTINCTIVE, actually-colliding surnames earn a place here.
NON_DEPUTY_SURNAMES = frozenset({
    "aznar",     # José María Aznar (ex-PM) vs the deputy Aznar Teruel
    "suárez",    # Adolfo Suárez (ex-PM) vs the deputy Rodríguez Suárez (2nd surname)
    "clavijo",   # Fernando Clavijo (presidente de Canarias) vs Gamarra Ruiz-Clavijo (2nd surname)
})

# Common words spaCy sometimes tags as a person because they are also borne as a
# surname (typically sentence-initial "Bueno, …"). They never name the deputy.
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
    # "expresidente/-a (del Gobierno) <Apellido>"
    r"expresident[ae]s?\s+(?:del\s+gobierno\s+)?([a-zá-úñ]+)",
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
class DeputyEntry:
    """One deputy's canonical identity plus the lowercased keys a span is scored
    against (full "apellido, nombre", "nombre apellido", and the bare surname)."""
    deputy_id: str
    name: str
    keys: tuple[str, ...]


def build_deputy_index(deputies) -> list[DeputyEntry]:
    """Build the match index from ``Deputy`` records (``name`` = 'Apellido, Nombre',
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
        index.append(DeputyEntry(
            deputy_id=deputy.id, name=name, keys=tuple(k for k in keys if k)))
    return index


def normalize_span(span: str) -> str:
    """Lowercase, drop punctuation and courtesy honorifics/articles. Returns the
    residual name, or "" when nothing usable remains (e.g. "Su Señoría")."""
    cleaned = _PUNCT_RE.sub(" ", span.lower())
    tokens = [t for t in cleaned.split() if t and t not in _HONORIFICS]
    residual = " ".join(tokens)
    return residual if len(residual) >= _MIN_LEN else ""


def _resolve_one(norm: str, index: list[DeputyEntry], threshold: int):
    """Best-scoring deputy for a normalized span, or ``None`` when nothing clears
    the threshold or the top score is shared (ambiguous surname)."""
    best_score, best, tie = 0, None, False
    for entry in index:
        score = max(fuzz.token_set_ratio(norm, key) for key in entry.keys)
        if score > best_score:
            best_score, best, tie = score, entry, False
        elif score == best_score and best is not None and entry.deputy_id != best.deputy_id:
            tie = True
    if best is None or best_score < threshold or tie:
        return None
    return best


def _is_excluded(norm: str, entry: DeputyEntry, excluded: frozenset[str]) -> bool:
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
    spans, index: list[DeputyEntry], threshold: int,
    excluded_surnames: frozenset[str] = frozenset()) -> list[Mention]:
    """Collapse raw NER ``spans`` (duplicates preserved) into canonical ``Mention``s.

    Each span is normalized then resolved (cached per normalized form). Occurrences
    that resolve to the same deputy are merged: ``count`` totals them and
    ``surface_forms`` keeps the distinct raw spans seen. ``excluded_surnames`` drops
    spans that name a flagged non-deputy (see ``_is_excluded``). Returns mentions
    ordered by descending count then name."""
    cache: dict[str, DeputyEntry | None] = {}
    by_deputy: dict[str, dict] = {}
    for span in spans:
        norm = normalize_span(span)
        if not norm:
            continue
        if norm not in cache:
            cache[norm] = _resolve_one(norm, index, threshold)
        entry = cache[norm]
        if entry is None:
            continue
        if _is_excluded(norm, entry, excluded_surnames):
            continue
        acc = by_deputy.setdefault(
            entry.deputy_id,
            {"name": entry.name, "surface_forms": set(), "count": 0})
        acc["surface_forms"].add(span.strip())
        acc["count"] += 1

    mentions = [
        Mention(
            deputy_id=deputy_id,
            name=acc["name"],
            surface_forms=sorted(acc["surface_forms"]),
            count=acc["count"])
        for deputy_id, acc in by_deputy.items()
    ]
    mentions.sort(key=lambda m: (-m.count, m.name))
    return mentions
