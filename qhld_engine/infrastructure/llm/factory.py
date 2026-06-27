from langchain_core.language_models.chat_models import BaseChatModel

from qhld_engine.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def create_llm_from_env(settings: Settings | None = None) -> BaseChatModel:
    from qhld_engine.infrastructure.config.settings import get_settings
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Import providers to trigger registration
from qhld_engine.infrastructure.llm import (  # noqa: E402, F401
    anthropic,
    google,
    mistral,
    ollama,
    openai,
)
