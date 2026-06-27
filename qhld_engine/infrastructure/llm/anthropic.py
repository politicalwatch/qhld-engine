from langchain_anthropic import ChatAnthropic

from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


@_register("anthropic")
def create(settings: Settings) -> ChatAnthropic:
    kwargs = {"model": settings.llm_model, "api_key": settings.anthropic_api_key}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatAnthropic(**kwargs)
