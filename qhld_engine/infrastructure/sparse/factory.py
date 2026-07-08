"""Sparse-embedder factory — mirrors the reranker/embeddings registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.sparse_provider``. There is no noop adapter:
"none"/empty means sparse retrieval is disabled and callers simply don't build an
embedder, so a misspelled provider fails loudly here instead of silently degrading
to dense-only search.
"""

from qhld_engine.domain.ports.sparse_embeddings import SparseEmbedderPort
from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_sparse_embedder_from_env(settings: Settings | None = None) -> SparseEmbedderPort:
    from qhld_engine.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.sparse_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown sparse provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration.
from qhld_engine.infrastructure.sparse import bm25  # noqa: E402, F401
