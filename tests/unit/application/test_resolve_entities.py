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


@pytest.fixture
def resolver():
    return EntityResolver(distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS)


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


def test_no_filters_when_nothing_extracted(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="financiación")).filters == {}
