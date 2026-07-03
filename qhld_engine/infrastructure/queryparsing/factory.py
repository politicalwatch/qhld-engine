"""Query-parser factory — mirrors the llm/embeddings/reranker registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.query_parser_provider``. "llm" uses structured
output from a chat model (the product path); a rule-based baseline (spaCy +
dateparser) registers here later for the thesis comparison.
"""

from qhld_engine.domain.ports.query_parser import QueryParserPort
from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_query_parser_from_env(settings: Settings | None = None) -> QueryParserPort:
    from qhld_engine.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.query_parser_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown query parser provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration.
from qhld_engine.infrastructure.queryparsing import llm  # noqa: E402, F401
