from langchain_google_genai import GoogleGenerativeAIEmbeddings

from qhld_engine.infrastructure.config.settings import Settings
from .factory import _register


@_register("google")
def create(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.google_api_key,
    )
