"""Reranker factory — mirrors the embeddings/vectorstore registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.reranker_provider``. "noop" leaves the bi-encoder
order untouched (the clean baseline); "cross_encoder" serves a local
sentence-transformers cross-encoder.
"""

from qhld_engine.domain.ports.reranker import RerankerPort
from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_reranker_from_env(settings: Settings | None = None) -> RerankerPort:
    from qhld_engine.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.reranker_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown reranker provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration.
from qhld_engine.infrastructure.reranker import cross_encoder, noop  # noqa: E402, F401
