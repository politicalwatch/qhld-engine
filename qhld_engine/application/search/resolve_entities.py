"""Entity resolution for Query understanding: turn a ``ParsedQuery`` into concrete Qdrant
payload filters.

The LLM extracts what the user *said* (names, titles, parties, ISO dates); this
service maps that onto what the index actually *stores*:
- speaker names -> fuzzy match against the corpus ``speaker`` values ("Apellido,
  Nombre"), so it works for deputies AND non-deputies (ministers etc.) alike.
- speaker title -> fuzzy match against the corpus ``role`` values (full office titles).
- groups/parties -> the payload ``group`` code (== ``ParliamentaryGroup.shortname``),
  resolved from an alias map over group short/long names, party names and a curated
  alias file, plus a token-normalized form that strips the generic words users swap
  freely ("grupo socialista" / "partido socialista" / "los socialistas"). A curated
  ideological/bloc category ("izquierda", "independentistas") expands to every group
  labelled with it — the parser passes categories through verbatim; the labels are
  editorial data in the alias file, not a model judgment.
- mentioned persons -> person ids (deputies, or non-deputies such as ministers, the
  King, regional presidents or foreign leaders), matched against the SAME person
  catalog that tags the corpus, then filtered on the payload ``mentions`` list.
- ISO dates     -> a numeric ``date`` range ({"gte"/"lte": YYYYMMDD}).

When several values resolve for one field, the filter value becomes a list (the
store treats it as any-of); mentioned persons instead honour the parsed
``mentions_mode`` — ``{"all": [ids]}`` requires every person to be mentioned,
a plain list accepts any of them.

Corpus values are read via an injected ``distinct(key)`` callable (wrapping
``VectorStorePort.distinct_values`` on the target collection), so the resolver is
trivially testable with a stub. Each resolution is recorded in ``notes`` so the
CLI can show what was understood (and what could not be matched).
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from thefuzz import fuzz, process

from qhld_engine.application.speeches.persons_catalog import load_person_index
from qhld_engine.domain.ports.query_parser import ParsedQuery
from qhld_engine.domain.speeches.mentions import resolve_person

# token_set_ratio scores a subset match ~100 ("María Jesús Montero" ⊆ "Montero
# Cuadrado, María Jesús", or a surname-only "Montero") while an unrelated name
# stays low (~25), so a high threshold is both forgiving and precise.
_SPEAKER_THRESHOLD = 90
_ROLE_THRESHOLD = 70      # "ministra de economía" ⊆ the full official title
_GROUP_THRESHOLD = 80

# Map the many ways a language can be named (or mis-coded by an LLM: "Gallego",
# "cat") to the ISO code stored in the payload ``lang``. Payload uses es/ca/gl/eu.
_LANG_ALIASES = {
    "es": "es", "spa": "es", "castellano": "es", "español": "es", "espanol": "es",
    "ca": "ca", "cat": "ca", "catalán": "ca", "catalan": "ca", "català": "ca",
    "gl": "gl", "gal": "gl", "gallego": "gl", "galego": "gl",
    "eu": "eu", "eus": "eu", "euskera": "eu", "euskara": "eu", "vasco": "eu", "vascuence": "eu",
}

GROUP_ALIASES_FILE = Path(__file__).parent / "group_aliases.json"

# Words that carry no identity within a group/party name — the scaffolding users
# swap freely ("grupo socialista" / "partido socialista" / "los socialistas").
# Unaccented singular forms, matched after accent stripping and plural folding.
_GENERIC_GROUP_TOKENS = {
    "grupo", "parlamentario", "parlamentaria", "partido", "politico", "politica",
    "bloque", "el", "la", "los", "las", "de", "del", "per",
}


def load_curated_group_aliases(path=GROUP_ALIASES_FILE):
    """Read the curated group-alias records (a JSON array of {code, aliases})."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Resolution:
    """The store-ready filters plus a human-readable trace of how each field
    resolved (or why it did not)."""
    filters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class EntityResolver:
    def __init__(self, distinct, groups, deputies=None, mention_threshold=90,
                 curated=None, nondeputy_speakers=None, curated_aliases=None):
        """``distinct`` is ``callable(key) -> set`` over the target collection's
        payload; ``groups`` is the list of ``ParliamentaryGroup`` records. ``deputies``
        (the ``Deputy`` catalog) enables resolving mentioned persons; when given, the
        full person index (deputies + curated non-deputies + bootstrapped speakers) is
        built with the SAME assembler used to tag the corpus, so a query resolves to the
        same ids that were indexed. ``curated``/``nondeputy_speakers``/``curated_aliases``
        may be injected (tests); otherwise they are read from the data files /
        ``Speeches``. Omit ``deputies`` => mentioned-person queries are left unfiltered."""
        self._distinct = distinct
        if curated_aliases is None:
            curated_aliases = load_curated_group_aliases()
        (self._group_aliases, self._group_aliases_normalized,
         self._group_categories) = _build_group_aliases(groups, curated_aliases)
        self._person_index = (
            load_person_index(deputies, mention_threshold,
                              curated=curated, nondeputy_speakers=nondeputy_speakers)
            if deputies else [])
        self._mention_threshold = mention_threshold

    def resolve(self, parsed: ParsedQuery) -> Resolution:
        result = Resolution()
        if parsed.speakers:
            self._resolve_speakers(result, parsed.speakers)
        if parsed.speaker_title:
            self._resolve_role(result, parsed.speaker_title)
        if parsed.mentioned_persons:
            self._resolve_mentions(result, parsed.mentioned_persons, parsed.mentions_mode)
        if parsed.groups_or_parties:
            self._resolve_groups(result, parsed.groups_or_parties)
        self._resolve_dates(result, parsed)
        if parsed.lang:
            self._resolve_lang(result, parsed.lang)
        if parsed.legislature:
            result.filters["legislature"] = parsed.legislature
        return result

    def _resolve_speakers(self, result, raws):
        choices = [v for v in self._distinct("speaker") if v]
        matched = []
        for raw in raws:
            value = self._fuzzy_match(
                result, "speaker", raw, choices, _SPEAKER_THRESHOLD)
            if value and value not in matched:
                matched.append(value)
        _set_filter(result, "speaker", matched)

    def _resolve_role(self, result, raw):
        choices = [v for v in self._distinct("role") if v]
        value = self._fuzzy_match(result, "role", raw, choices, _ROLE_THRESHOLD)
        if value:
            result.filters["role"] = value

    def _resolve_mentions(self, result, raws, mode):
        ids = []
        for raw in raws:
            entry = (resolve_person(raw, self._person_index, self._mention_threshold)
                     if self._person_index else None)
            if entry:
                result.notes.append(
                    f"mentions: '{raw}' → '{entry.name}' ({entry.person_type})")
                if entry.person_id not in ids:
                    ids.append(entry.person_id)
            else:
                result.notes.append(f"mentions: '{raw}' unresolved — not filtered")
        if not ids:
            return
        if len(ids) == 1:
            result.filters["mentions"] = ids[0]
        elif mode == "any":
            result.filters["mentions"] = sorted(ids)
        else:
            result.filters["mentions"] = {"all": sorted(ids)}

    def _resolve_lang(self, result, raw):
        code = _LANG_ALIASES.get(raw.strip().lower())
        if code:
            result.filters["lang"] = code
            if code != raw:
                result.notes.append(f"lang: '{raw}' → '{code}'")
        else:
            result.notes.append(f"lang: '{raw}' unresolved — not filtered")

    @staticmethod
    def _fuzzy_match(result, payload_key, raw, choices, threshold):
        """Best fuzzy match for one raw value, traced in ``notes``; None below
        ``threshold``."""
        match = process.extractOne(
            raw, choices, scorer=fuzz.token_set_ratio) if choices else None
        if match and match[1] >= threshold:
            result.notes.append(f"{payload_key}: '{raw}' → '{match[0]}' ({match[1]})")
            return match[0]
        best = f" (best '{match[0]}' {match[1]})" if match else ""
        result.notes.append(f"{payload_key}: '{raw}' unresolved{best} — not filtered")
        return None

    def _resolve_groups(self, result, raws):
        matched = []
        for raw in raws:
            codes = self._group_categories.get(_normalize_group_key(raw))
            if codes:
                result.notes.append(f"group: '{raw}' → {', '.join(codes)} (category)")
            else:
                shortname = self._match_group(raw)
                if not shortname:
                    result.notes.append(f"group: '{raw}' unresolved — not filtered")
                    continue
                result.notes.append(f"group: '{raw}' → '{shortname}'")
                codes = [shortname]
            matched.extend(code for code in codes if code not in matched)
        _set_filter(result, "group", matched)

    def _match_group(self, raw):
        key = raw.strip().lower()
        if key in self._group_aliases:
            return self._group_aliases[key]
        normalized = _normalize_group_key(raw)
        if not normalized:
            return None
        if normalized in self._group_aliases_normalized:
            return self._group_aliases_normalized[normalized]
        match = process.extractOne(
            normalized, list(self._group_aliases_normalized),
            scorer=fuzz.token_set_ratio)
        if match and match[1] >= _GROUP_THRESHOLD:
            return self._group_aliases_normalized[match[0]]
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


def _set_filter(result, key, matched):
    """A single resolved value stays a scalar (exact match); several become a
    list (the store treats it as any-of)."""
    if matched:
        result.filters[key] = matched[0] if len(matched) == 1 else sorted(matched)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _fold_plural(token: str) -> str:
    """'populares' -> 'popular', 'socialistas' -> 'socialista'. Applied to aliases
    and queries alike, so any over-stripping stays symmetric and still matches."""
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _normalize_group_key(text: str) -> str:
    """Reduce a group/party name to its distinctive tokens: lowercase, unaccent,
    fold plurals, drop generic words. 'los socialistas' and 'Grupo Parlamentario
    Socialista' both reduce to 'socialista'; 'los partidos de izquierda' to
    'izquierda'."""
    tokens = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    folded = (_fold_plural(t) for t in tokens)
    return " ".join(t for t in folded if t not in _GENERIC_GROUP_TOKENS)


def _build_group_aliases(groups, curated=None) -> tuple[dict, dict, dict]:
    """Three maps for group resolution: lowercased short/long/party/curated names
    verbatim -> code (``shortname``), their token-normalized forms -> code, and
    normalized curated category ('izquierda', 'independentista') -> every code
    labelled with it. Single-party groups win over the multi-party Mixto group on
    a party-name conflict (e.g. 'PSOE' -> GS, not GMx). Curated aliases and
    categories only apply to codes present in the current catalog."""
    curated_by_code = {row["code"]: row for row in (curated or [])}
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    categories: dict[str, list[str]] = {}
    for group in sorted(groups or [], key=lambda g: len(getattr(g, "parties", None) or [])):
        shortname = getattr(group, "shortname", None)
        if not shortname:
            continue
        row = curated_by_code.get(shortname, {})
        candidates = [shortname, getattr(group, "name", None),
                      *(getattr(group, "parties", None) or []),
                      *row.get("aliases", [])]
        for alias in candidates:
            if not alias:
                continue
            exact.setdefault(alias.lower(), shortname)
            key = _normalize_group_key(alias)
            if key:
                normalized.setdefault(key, shortname)
        for category in row.get("categories", []):
            key = _normalize_group_key(category)
            if key and shortname not in categories.setdefault(key, []):
                categories[key].append(shortname)
    return exact, normalized, categories


def _iso_to_int(iso: str) -> int | None:
    """'2025-04-03' -> 20250403. Returns None on a malformed value."""
    try:
        return int(iso.replace("-", ""))
    except (ValueError, AttributeError):
        return None
