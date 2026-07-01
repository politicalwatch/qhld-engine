from qhld_engine.domain.ports.vector_store import VectorStorePort
from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def create_vector_store_from_env(settings: Settings | None = None) -> VectorStorePort:
    from qhld_engine.infrastructure.config.settings import get_settings
    s = settings or get_settings()
    # Only Qdrant is wired today; the registry mirrors the embeddings/llm factories
    # so a second backend is a one-file add.
    provider = "qdrant"
    return _PROVIDERS[provider](s)


from qhld_engine.infrastructure.vectorstore import qdrant  # noqa: E402, F401
