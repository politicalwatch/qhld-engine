"""Benchmark service for the LLM-vs-rule-based query-parser comparison (thesis).

For each parser it runs the frozen parse query set, times ``parse``, resolves the
extracted fields through the *shared* ``EntityResolver`` (so the metric isolates
parsing, not resolution), and returns rows the pure ``parse_scoring`` module then
aggregates. Runs live (needs Qdrant for the corpus distinct values + Mongo for
the group aliases + an LLM for the "llm" parser), so run it on the host per the
standing benchmark-on-host preference:

    QDRANT_HOST=localhost OLLAMA_BASE_URL=http://localhost:11434 \\
    EMBEDDING_MODEL=bge-m3:567m \\
    QUERY_PARSER_LLM_PROVIDER=ollama QUERY_PARSER_LLM_MODEL=gpt-oss:20b \\
    uv run --group eval qhld eval parse --parsers llm,rule_based
"""

import json
import os
import time
from datetime import date

DEFAULT_QUERYSET = os.path.join(os.path.dirname(__file__), "parse_queryset.json")


def load_queryset(path=DEFAULT_QUERYSET):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _parse_date(iso):
    year, month, day = (int(part) for part in iso.split("-"))
    return date(year, month, day)


class RunParseBenchmark:
    def __init__(self, queryset_path=DEFAULT_QUERYSET, settings=None):
        from qhld_engine.infrastructure.config.settings import get_settings

        data = load_queryset(queryset_path)
        self.today = _parse_date(data["today"])
        self.queries = data["queries"]
        self.settings = settings or get_settings()
        self._resolver = None

    def run(self, parser_name):
        """Return a scored row per query for one parser ('llm' / 'rule_based')."""
        parser = self._parser(parser_name)
        resolver = self._resolver_obj()
        rows = []
        for entry in self.queries:
            start = time.perf_counter()
            parsed = parser.parse(entry["query"], self.today)
            latency = time.perf_counter() - start
            resolution = resolver.resolve(parsed)
            rows.append({
                **entry,
                "pred_filters": resolution.filters,
                "pred_topic": parsed.semantic_query,
                "notes": resolution.notes,
                "latency": latency,
            })
        return rows

    def _parser(self, name):
        from qhld_engine.infrastructure.queryparsing.factory import create_query_parser_from_env

        return create_query_parser_from_env(
            self.settings.model_copy(update={"query_parser_provider": name}))

    def _resolver_obj(self):
        """The shared resolver, bound to the target collection's corpus values and
        the ParliamentaryGroups alias source (built once, reused across parsers)."""
        if self._resolver is None:
            from qhld_engine.application.search.resolve_entities import EntityResolver
            from qhld_engine.infrastructure.embeddings.factory import create_embedder_from_env
            from qhld_engine.infrastructure.vectorstore.factory import create_vector_store_from_env
            from qhld_engine.infrastructure.vectorstore.naming import collection_name
            from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

            embedder = create_embedder_from_env(self.settings)
            store = create_vector_store_from_env(self.settings)
            collection = collection_name(self.settings, len(embedder.embed_query("probe")))
            self._resolver = EntityResolver(
                distinct=lambda key: store.distinct_values(collection, key),
                groups=ParliamentaryGroups.get_all())
        return self._resolver
