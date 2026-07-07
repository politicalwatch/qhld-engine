"""Unit tests for entity resolution — pure, with stubbed corpus + groups."""

from types import SimpleNamespace

import pytest

from qhld_engine.application.search.resolve_entities import EntityResolver
from qhld_engine.domain.ports.query_parser import ParsedQuery

pytestmark = pytest.mark.unit


CORPUS = {
    "speaker": {"Abascal Conde, Santiago", "Montero Cuadrado, María Jesús", "Aagesen Muñoz, Sara", None},
    "role": {"Diputado", "Ministra de Hacienda", "Ministra de Economía, Comercio y Empresa"},
}

GROUPS = [
    SimpleNamespace(name="Grupo Parlamentario Socialista", shortname="GS", parties=["PSOE"]),
    SimpleNamespace(name="Grupo Parlamentario Popular", shortname="GP", parties=["PP"]),
    SimpleNamespace(name="Grupo Parlamentario Mixto", shortname="GMx",
                    parties=["PODEMOS", "UPN", "BNG", "CCa", "PSOE"]),
]


class _FakeDeputy:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


DEPUTIES = [
    _FakeDeputy("dep-montero", "Montero Cuadrado, María Jesús"),
    _FakeDeputy("dep-abascal", "Abascal Conde, Santiago"),
]


@pytest.fixture
def resolver():
    # curated/nondeputy_speakers injected empty → the person index is deputies-only and
    # no Mongo/data-file I/O happens in these unit tests.
    return EntityResolver(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, deputies=DEPUTIES,
        curated=[], nondeputy_speakers=[])


def test_resolves_speaker_name_with_token_reordering(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speaker="Santiago Abascal"))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"


def test_unresolvable_speaker_is_not_filtered(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speaker="Fulano de Tal"))
    assert "speaker" not in r.filters
    assert any("unresolved" in note for note in r.notes)


def test_resolves_title_to_role(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speaker_title="ministra de economía"))
    assert r.filters["role"] == "Ministra de Economía, Comercio y Empresa"


def test_resolves_party_name_to_group_code(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="x", group_or_party="PSOE")).filters["group"] == "GS"
    assert resolver.resolve(ParsedQuery(semantic_query="x", group_or_party="PP")).filters["group"] == "GP"


def test_resolves_group_long_name(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", group_or_party="Grupo Socialista"))
    assert r.filters["group"] == "GS"


def test_shared_party_prefers_single_party_group_over_mixto(resolver):
    # 'PSOE' is listed under both GS and the catch-all GMx; GS (fewer parties) wins.
    assert resolver.resolve(ParsedQuery(semantic_query="x", group_or_party="PSOE")).filters["group"] == "GS"


def test_iso_dates_become_numeric_range(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", date_from="2025-04-03", date_to="2025-07-03"))
    assert r.filters["date"] == {"gte": 20250403, "lte": 20250703}


def test_open_ended_date_range(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", date_from="2025-01-01"))
    assert r.filters["date"] == {"gte": 20250101}


def test_lang_and_legislature_pass_through(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", lang="gl", legislature="15"))
    assert r.filters["lang"] == "gl"
    assert r.filters["legislature"] == "15"


def test_lang_names_and_variants_normalize_to_iso_code(resolver):
    # LLMs often emit the language name or an off-code ("Gallego", "cat").
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="Gallego")).filters["lang"] == "gl"
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="cat")).filters["lang"] == "ca"
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="euskera")).filters["lang"] == "eu"


def test_unknown_lang_is_not_filtered(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", lang="klingon"))
    assert "lang" not in r.filters


def test_no_filters_when_nothing_extracted(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="financiación")).filters == {}


def test_mentioned_person_resolves_to_deputy_id(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_person="Montero"))
    assert r.filters["mentions"] == "dep-montero"
    assert any("mentions:" in note for note in r.notes)


def test_unresolvable_mentioned_person_is_not_filtered(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_person="Winston Churchill"))
    assert "mentions" not in r.filters
    assert any("unresolved" in note for note in r.notes)


def test_mentioned_person_ignored_without_deputies_catalog():
    # A resolver built without the catalog cannot resolve a mention → no filter.
    resolver = EntityResolver(distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS)
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_person="Montero"))
    assert "mentions" not in r.filters


def test_mentioned_non_deputy_resolves_via_curated():
    # A curated non-deputy (Ayuso) resolves to her person id and is filtered.
    resolver = EntityResolver(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, deputies=DEPUTIES,
        curated=[{"person_id": "isabel-diaz-ayuso", "person_type": "regional_president",
                  "name": "Díaz Ayuso, Isabel", "aliases": ["Ayuso", "Díaz Ayuso"]}],
        nondeputy_speakers=[])
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_person="Ayuso"))
    assert r.filters["mentions"] == "isabel-diaz-ayuso"
    assert any("regional_president" in note for note in r.notes)


def test_mentioned_bootstrapped_minister_resolves():
    # A non-deputy speaker bootstrapped from the corpus (a minister) is resolvable too.
    resolver = EntityResolver(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, deputies=DEPUTIES,
        curated=[],
        nondeputy_speakers=[{"speaker": "Aagesen Muñoz, Sara",
                             "role": "Vicepresidenta Tercera y Ministra"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_person="Aagesen"))
    assert r.filters["mentions"] == "aagesen-munoz-sara"
    assert any("minister" in note for note in r.notes)
