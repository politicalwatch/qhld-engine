from langchain_core.embeddings import Embeddings

from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def create_embedder_from_env(settings: Settings | None = None) -> Embeddings:
    from qhld_engine.infrastructure.config.settings import get_settings
    s = settings or get_settings()
    provider = s.embedding_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


from qhld_engine.infrastructure.embeddings import openai, google, ollama  # noqa: E402, F401
