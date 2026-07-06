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
    "el", "la", "los", "las", "un", "una",
    "sr", "sra", "srs", "sras", "señor", "señora", "señores", "señoras",
    "don", "doña", "su", "señoria", "señorias", "señoría", "señorías",
    "excelentisimo", "excelentisima", "excelentísimo", "excelentísima",
    "ilustrisimo", "ilustrisima", "ilustrísimo", "ilustrísima",
}
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_MIN_LEN = 3


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


def resolve_mentions(spans, index: list[DeputyEntry], threshold: int) -> list[Mention]:
    """Collapse raw NER ``spans`` (duplicates preserved) into canonical ``Mention``s.

    Each span is normalized then resolved (cached per normalized form). Occurrences
    that resolve to the same deputy are merged: ``count`` totals them and
    ``surface_forms`` keeps the distinct raw spans seen. Returns mentions ordered
    by descending count then name."""
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
