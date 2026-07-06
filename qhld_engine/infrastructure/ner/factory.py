"""NER factory — mirrors the queryparsing/llm/embeddings/reranker registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.ner_provider`` (default "spacy").
"""

from qhld_engine.domain.ports.ner import NerPort
from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_ner_from_env(settings: Settings | None = None) -> NerPort:
    from qhld_engine.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.ner_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown NER provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration. The spaCy import is lazy inside the adapter,
# so this stays cheap even before the model wheel is installed.
from qhld_engine.infrastructure.ner import spacy  # noqa: E402, F401
