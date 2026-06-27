from langchain_openai import OpenAIEmbeddings

from qhld_engine.infrastructure.config.settings import Settings
from .factory import _register


@_register("openai")
def create(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
