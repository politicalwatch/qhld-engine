"""Unit tests for NaturalSearchSpeeches — stubbed parser/resolver/search."""

from datetime import date

import pytest

from qhld_engine.application.search.natural_search import NaturalSearchSpeeches
from qhld_engine.application.search.resolve_entities import Resolution, UnresolvedEntity
from qhld_engine.domain.ports.query_parser import ParsedQuery
from qhld_engine.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


class _StubParser:
    def __init__(self, parsed):
        self.parsed = parsed

    def parse(self, query, today):
        self.query = query
        self.today = today
        return self.parsed


class _StubResolver:
    def __init__(self, resolution):
        self.resolution = resolution

    def resolve(self, parsed):
        return self.resolution


class _SpySearch:
    def __init__(self):
        self.calls = []

    def search(self, query, k=10, filters=None):
        self.calls.append(("search", query, k, filters))
        return ["hit"]

    def search_grouped(self, query, page_size=10, highlights=3, filters=None):
        self.calls.append(("grouped", query, page_size, highlights, filters))
        return ["group"]


def _service(parsed, resolution):
    return NaturalSearchSpeeches(
        settings=Settings(_env_file=None),
        parser=_StubParser(parsed),
        search=_SpySearch(),
        resolver=_StubResolver(resolution))


def test_searches_residual_topic_with_resolved_filters():
    parsed = ParsedQuery(semantic_query="financiación autonómica", speakers=["Montero"])
    resolution = Resolution(filters={"speaker": "Montero Cuadrado, María Jesús",
                                     "date": {"gte": 20240703}})
    service = _service(parsed, resolution)
    result = service.execute("intervenciones de Montero sobre financiación autonómica del último año",
                             today=date(2025, 7, 3), k=5)
    kind, query, k, filters = service.search.calls[0]
    assert kind == "search"
    assert query == "financiación autonómica"        # topic only, not the full NL query
    assert k == 5
    assert filters == {"speaker": "Montero Cuadrado, María Jesús", "date": {"gte": 20240703}}
    assert result.hits == ["hit"]


def test_mentioned_person_filter_is_forwarded_to_search():
    parsed = ParsedQuery(semantic_query="vivienda", mentioned_persons=["Zapatero"])
    resolution = Resolution(filters={"mentions": "dep-zapatero"})
    service = _service(parsed, resolution)
    service.execute("intervenciones sobre vivienda que mencionen a Zapatero",
                    today=date(2025, 7, 3))
    _, query, _, filters = service.search.calls[0]
    assert query == "vivienda"
    assert filters == {"mentions": "dep-zapatero"}


def test_grouped_routes_to_search_grouped():
    service = _service(ParsedQuery(semantic_query="vivienda"), Resolution())
    service.execute("vivienda", today=date(2025, 7, 3), k=8, grouped=True, highlights=4)
    kind, query, page_size, highlights, filters = service.search.calls[0]
    assert kind == "grouped"
    assert (query, page_size, highlights) == ("vivienda", 8, 4)


def test_no_filters_passes_none():
    service = _service(ParsedQuery(semantic_query="sanidad pública"), Resolution())
    service.execute("sanidad pública", today=date(2025, 7, 3))
    assert service.search.calls[0][3] is None


def test_pure_filter_query_falls_back_to_full_text():
    # No topic extracted (semantic_query empty) but a group filter present.
    parsed = ParsedQuery(semantic_query="", groups_or_parties=["PSOE"])
    resolution = Resolution(filters={"group": "GS"})
    service = _service(parsed, resolution)
    service.execute("intervenciones del PSOE", today=date(2025, 7, 3))
    _, query, _, filters = service.search.calls[0]
    assert query == "intervenciones del PSOE"   # fallback so there's a vector to rank by
    assert filters == {"group": "GS"}


def test_blocked_resolution_skips_retrieval():
    # An unsatisfiable filter (person not in the catalog) must yield zero hits,
    # not hits that silently ignore the filter.
    parsed = ParsedQuery(semantic_query="vivienda", mentioned_persons=["Santiago Segura"])
    resolution = Resolution(unresolved=[
        UnresolvedEntity("mentions", "Santiago Segura", blocking=True)])
    service = _service(parsed, resolution)
    result = service.execute("vivienda que mencionen a Santiago Segura",
                             today=date(2025, 7, 3))
    assert result.hits == []
    assert service.search.calls == []
    assert result.resolution.blocked


def test_blocked_resolution_skips_grouped_retrieval_too():
    resolution = Resolution(unresolved=[
        UnresolvedEntity("speaker", "Fulano de Tal", blocking=True)])
    service = _service(ParsedQuery(semantic_query="x"), resolution)
    result = service.execute("x", today=date(2025, 7, 3), grouped=True)
    assert result.hits == []
    assert service.search.calls == []


def test_nonblocking_unresolved_still_searches():
    # A member dropped from an any-of list is a warning, not a dead end.
    resolution = Resolution(
        filters={"speaker": "Abascal Conde, Santiago"},
        unresolved=[UnresolvedEntity("speaker", "Fulano de Tal", blocking=False)])
    service = _service(ParsedQuery(semantic_query="pensiones"), resolution)
    result = service.execute("pensiones", today=date(2025, 7, 3))
    assert result.hits == ["hit"]
    assert service.search.calls[0][3] == {"speaker": "Abascal Conde, Santiago"}


def test_today_is_forwarded_to_parser():
    parser = _StubParser(ParsedQuery(semantic_query="x"))
    service = NaturalSearchSpeeches(
        settings=Settings(_env_file=None), parser=parser,
        search=_SpySearch(), resolver=_StubResolver(Resolution()))
    service.execute("x", today=date(2024, 1, 15))
    assert parser.today == date(2024, 1, 15)
