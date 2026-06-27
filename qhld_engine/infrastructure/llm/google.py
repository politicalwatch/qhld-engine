from langchain_google_genai import ChatGoogleGenerativeAI

from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


@_register("google")
def create(settings: Settings) -> ChatGoogleGenerativeAI:
    kwargs = {"model": settings.llm_model, "google_api_key": settings.google_api_key}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatGoogleGenerativeAI(**kwargs)
