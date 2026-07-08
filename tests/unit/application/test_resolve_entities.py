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
    SimpleNamespace(name="Grupo Parlamentario Republicano", shortname="GR", parties=["ERC"]),
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


def _resolver(**overrides):
    # curated/nondeputy_speakers/curated_aliases injected → no Mongo/data-file I/O
    # happens in these unit tests.
    defaults = dict(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, deputies=DEPUTIES,
        curated=[], nondeputy_speakers=[], curated_aliases=[])
    return EntityResolver(**{**defaults, **overrides})


@pytest.fixture
def resolver():
    return _resolver()


def test_resolves_speaker_name_with_token_reordering(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Santiago Abascal"]))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"


def test_unresolvable_speaker_is_not_filtered(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Fulano de Tal"]))
    assert "speaker" not in r.filters
    assert any("unresolved" in note for note in r.notes)


def test_multiple_speakers_resolve_to_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", speakers=["Santiago Abascal", "María Jesús Montero"]))
    assert r.filters["speaker"] == [
        "Abascal Conde, Santiago", "Montero Cuadrado, María Jesús"]


def test_partially_resolved_speakers_keep_the_resolved_one(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", speakers=["Santiago Abascal", "Fulano de Tal"]))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"
    assert any("unresolved" in note for note in r.notes)


def test_resolves_title_to_role(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speaker_title="ministra de economía"))
    assert r.filters["role"] == "Ministra de Economía, Comercio y Empresa"


def test_resolves_party_name_to_group_code(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PSOE"])).filters["group"] == "GS"
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PP"])).filters["group"] == "GP"


def test_resolves_group_long_name(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["Grupo Socialista"]))
    assert r.filters["group"] == "GS"


def test_resolves_colloquial_group_names(resolver):
    # The phrasings a user swaps freely: party-word scaffolding and plural demonyms.
    for raw, code in [
        ("Partido Socialista", "GS"),
        ("los socialistas", "GS"),
        ("socialistas", "GS"),
        ("Partido Socialista Obrero Español", "GS"),
        ("los populares", "GP"),
        ("partido popular", "GP"),
        ("los republicanos", "GR"),
    ]:
        r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=[raw]))
        assert r.filters.get("group") == code, f"'{raw}' → {r.filters.get('group')}"


def test_curated_alias_resolves_when_code_is_in_catalog():
    resolver = _resolver(curated_aliases=[{"code": "GR", "aliases": ["Esquerra"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["esquerra"]))
    assert r.filters["group"] == "GR"


def test_curated_alias_for_absent_group_is_ignored():
    resolver = _resolver(curated_aliases=[{"code": "GCUP", "aliases": ["Esquerra"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["esquerra"]))
    assert "group" not in r.filters


def test_multiple_groups_resolve_to_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", groups_or_parties=["Grupo Socialista", "Grupo Popular"]))
    assert r.filters["group"] == ["GP", "GS"]


def test_shared_party_prefers_single_party_group_over_mixto(resolver):
    # 'PSOE' is listed under both GS and the catch-all GMx; GS (fewer parties) wins —
    # for the verbatim alias and for the normalized colloquial forms alike.
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PSOE"])).filters["group"] == "GS"
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["socialistas"])).filters["group"] == "GS"


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
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_persons=["Montero"]))
    assert r.filters["mentions"] == "dep-montero"
    assert any("mentions:" in note for note in r.notes)


def test_unresolvable_mentioned_person_is_not_filtered(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Winston Churchill"]))
    assert "mentions" not in r.filters
    assert any("unresolved" in note for note in r.notes)


def test_multiple_mentions_default_to_requiring_all(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"]))
    assert r.filters["mentions"] == {"all": ["dep-abascal", "dep-montero"]}


def test_multiple_mentions_any_mode_becomes_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"], mentions_mode="any"))
    assert r.filters["mentions"] == ["dep-abascal", "dep-montero"]


def test_partially_resolved_mentions_keep_the_resolved_one(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Winston Churchill"]))
    assert r.filters["mentions"] == "dep-montero"


def test_mentioned_person_ignored_without_deputies_catalog():
    # A resolver built without the catalog cannot resolve a mention → no filter.
    resolver = EntityResolver(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, curated_aliases=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Montero"]))
    assert "mentions" not in r.filters


def test_mentioned_non_deputy_resolves_via_curated():
    # A curated non-deputy (Ayuso) resolves to her person id and is filtered.
    resolver = _resolver(
        curated=[{"person_id": "isabel-diaz-ayuso", "person_type": "regional_president",
                  "name": "Díaz Ayuso, Isabel", "aliases": ["Ayuso", "Díaz Ayuso"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_persons=["Ayuso"]))
    assert r.filters["mentions"] == "isabel-diaz-ayuso"
    assert any("regional_president" in note for note in r.notes)


def test_mentioned_bootstrapped_minister_resolves():
    # A non-deputy speaker bootstrapped from the corpus (a minister) is resolvable too.
    resolver = _resolver(
        nondeputy_speakers=[{"speaker": "Aagesen Muñoz, Sara",
                             "role": "Vicepresidenta Tercera y Ministra"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Aagesen"]))
    assert r.filters["mentions"] == "aagesen-munoz-sara"
    assert any("minister" in note for note in r.notes)
