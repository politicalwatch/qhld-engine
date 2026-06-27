from typing import Protocol


class EmbedderProtocol(Protocol):
    """Port for computing embeddings.

    Implementations live in infrastructure/embeddings/. Langchain's
    Embeddings base class satisfies this interface.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
