from langchain_mistralai import ChatMistralAI

from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


@_register("mistral")
def create(settings: Settings) -> ChatMistralAI:
    kwargs = {"model": settings.llm_model, "api_key": settings.mistral_api_key}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatMistralAI(**kwargs)
