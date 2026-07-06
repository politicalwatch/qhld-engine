"""Application service for natural-language search over speeches.

Orchestrates parse -> resolve -> filtered retrieve, then delegates to the existing
``SearchSpeeches`` on the *residual* semantic query (topic only), applying the
resolved structured filters. Plain and injectable — parser, resolver and search
are all defaulted from env for production but overridable in tests. The LangGraph
wrapper (and LangSmith tracing) is deferred; this near-linear flow is a plain
service for now.

``today`` is passed in (from the CLI edge), never read from a wall-clock here, so
relative-date resolution stays deterministic and testable.
"""

from dataclasses import dataclass, field

from qhld_engine.application.search.resolve_entities import EntityResolver, Resolution
from qhld_engine.domain.ports.query_parser import ParsedQuery
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_engine.infrastructure.queryparsing.factory import create_query_parser_from_env
from qhld_engine.infrastructure.vectorstore.naming import collection_name


@dataclass
class NaturalResult:
    parsed: ParsedQuery
    resolution: Resolution
    semantic_query: str
    hits: list = field(default_factory=list)
    grouped: bool = False


class NaturalSearchSpeeches:
    def __init__(self, settings=None, parser=None, search=None, resolver=None):
        self.settings = settings or get_settings()
        self.parser = parser or create_query_parser_from_env(self.settings)
        self.search = search or self._default_search()
        self._resolver = resolver

    def _default_search(self):
        from qhld_engine.application.search.search_speeches import SearchSpeeches

        return SearchSpeeches(settings=self.settings)

    def _resolver_from_corpus(self) -> EntityResolver:
        """Build a resolver bound to the target (per-model) collection's payload:
        distinct speaker/role/group values come from that collection, group aliases
        from the ParliamentaryGroups repo, and the deputies catalog resolves a
        mentioned person to a deputy id (matching the payload ``mentions`` list)."""
        from tipi_data.repositories.deputies import Deputies
        from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

        dim = len(self.search.embedder.embed_query("probe"))
        collection = collection_name(self.settings, dim)
        return EntityResolver(
            distinct=lambda key: self.search.store.distinct_values(collection, key),
            groups=ParliamentaryGroups.get_all(),
            deputies=Deputies.get_all(),
            mention_threshold=self.settings.mention_match_threshold)

    def resolver(self) -> EntityResolver:
        if self._resolver is None:
            self._resolver = self._resolver_from_corpus()
        return self._resolver

    def execute(self, query, today, k=10, grouped=False, highlights=3) -> NaturalResult:
        parsed = self.parser.parse(query, today)
        resolution = self.resolver().resolve(parsed)
        filters = resolution.filters or None
        # Search the topic only. If the query was pure-filter (no topic), fall back
        # to the full text so there is still a vector to rank the filtered set by.
        semantic = parsed.semantic_query.strip() if parsed.semantic_query else ""
        semantic = semantic or query
        if grouped:
            hits = self.search.search_grouped(
                semantic, page_size=k, highlights=highlights, filters=filters)
        else:
            hits = self.search.search(semantic, k=k, filters=filters)
        return NaturalResult(
            parsed=parsed, resolution=resolution, semantic_query=semantic,
            hits=hits, grouped=grouped)
