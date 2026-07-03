"""Entity resolution for Query understanding: turn a ``ParsedQuery`` into concrete Qdrant
payload filters.

The LLM extracts what the user *said* (a name, a title, a party, ISO dates); this
service maps that onto what the index actually *stores*:
- speaker name  -> fuzzy match against the corpus ``speaker`` values ("Apellido,
  Nombre"), so it works for deputies AND non-deputies (ministers etc.) alike.
- speaker title -> fuzzy match against the corpus ``role`` values (full office titles).
- group/party   -> the payload ``group`` code (== ``ParliamentaryGroup.shortname``),
  resolved from an alias map over group short/long names and party names.
- ISO dates     -> a numeric ``date`` range ({"gte"/"lte": YYYYMMDD}).

Corpus values are read via an injected ``distinct(key)`` callable (wrapping
``VectorStorePort.distinct_values`` on the target collection), so the resolver is
trivially testable with a stub. Each resolution is recorded in ``notes`` so the
CLI can show what was understood (and what could not be matched).
"""

from dataclasses import dataclass, field

from thefuzz import fuzz, process

from qhld_engine.domain.ports.query_parser import ParsedQuery

# token_set_ratio scores a subset match ~100 ("María Jesús Montero" ⊆ "Montero
# Cuadrado, María Jesús", or a surname-only "Montero") while an unrelated name
# stays low (~25), so a high threshold is both forgiving and precise.
_SPEAKER_THRESHOLD = 90
_ROLE_THRESHOLD = 70      # "ministra de economía" ⊆ the full official title
_GROUP_THRESHOLD = 80


@dataclass
class Resolution:
    """The store-ready filters plus a human-readable trace of how each field
    resolved (or why it did not)."""
    filters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class EntityResolver:
    def __init__(self, distinct, groups):
        """``distinct`` is ``callable(key) -> set`` over the target collection's
        payload; ``groups`` is the list of ``ParliamentaryGroup`` records."""
        self._distinct = distinct
        self._group_aliases = _build_group_aliases(groups)

    def resolve(self, parsed: ParsedQuery) -> Resolution:
        result = Resolution()
        if parsed.speaker:
            self._resolve_fuzzy(
                result, "speaker", parsed.speaker, "speaker",
                fuzz.token_set_ratio, _SPEAKER_THRESHOLD)
        if parsed.speaker_title:
            self._resolve_fuzzy(
                result, "role", parsed.speaker_title, "role",
                fuzz.token_set_ratio, _ROLE_THRESHOLD)
        if parsed.group_or_party:
            self._resolve_group(result, parsed.group_or_party)
        self._resolve_dates(result, parsed)
        for passthrough in ("lang", "legislature"):
            value = getattr(parsed, passthrough)
            if value:
                result.filters[passthrough] = value
        return result

    def _resolve_fuzzy(self, result, payload_key, raw, distinct_key, scorer, threshold):
        choices = [v for v in self._distinct(distinct_key) if v]
        match = process.extractOne(raw, choices, scorer=scorer) if choices else None
        if match and match[1] >= threshold:
            result.filters[payload_key] = match[0]
            result.notes.append(f"{payload_key}: '{raw}' → '{match[0]}' ({match[1]})")
        else:
            best = f" (best '{match[0]}' {match[1]})" if match else ""
            result.notes.append(f"{payload_key}: '{raw}' unresolved{best} — not filtered")

    def _resolve_group(self, result, raw):
        shortname = self._match_group(raw)
        if shortname:
            result.filters["group"] = shortname
            result.notes.append(f"group: '{raw}' → '{shortname}'")
        else:
            result.notes.append(f"group: '{raw}' unresolved — not filtered")

    def _match_group(self, raw):
        key = raw.strip().lower()
        if key in self._group_aliases:
            return self._group_aliases[key]
        match = process.extractOne(
            key, list(self._group_aliases), scorer=fuzz.token_set_ratio)
        if match and match[1] >= _GROUP_THRESHOLD:
            return self._group_aliases[match[0]]
        return None

    def _resolve_dates(self, result, parsed):
        bounds = {}
        if parsed.date_from:
            bounds["gte"] = _iso_to_int(parsed.date_from)
        if parsed.date_to:
            bounds["lte"] = _iso_to_int(parsed.date_to)
        bounds = {k: v for k, v in bounds.items() if v is not None}
        if bounds:
            result.filters["date"] = bounds
            result.notes.append(f"date: {bounds}")


def _build_group_aliases(groups) -> dict:
    """Map lowercased short/long/party names to the payload ``group`` code
    (``shortname``). Single-party groups win over the multi-party Mixto group on a
    party-name conflict (e.g. 'PSOE' -> GS, not GMx)."""
    aliases: dict[str, str] = {}
    for group in sorted(groups or [], key=lambda g: len(getattr(g, "parties", None) or [])):
        shortname = getattr(group, "shortname", None)
        if not shortname:
            continue
        candidates = [shortname, getattr(group, "name", None), *(getattr(group, "parties", None) or [])]
        for alias in candidates:
            if alias:
                aliases.setdefault(alias.lower(), shortname)
    return aliases


def _iso_to_int(iso: str) -> int | None:
    """'2025-04-03' -> 20250403. Returns None on a malformed value."""
    try:
        return int(iso.replace("-", ""))
    except (ValueError, AttributeError):
        return None
