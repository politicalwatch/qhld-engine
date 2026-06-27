from langchain_ollama import ChatOllama

from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


@_register("ollama")
def create(settings: Settings) -> ChatOllama:
    kwargs = {"model": settings.llm_model, "base_url": settings.ollama_base_url}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatOllama(**kwargs)
