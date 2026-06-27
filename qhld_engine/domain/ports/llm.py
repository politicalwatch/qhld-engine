from typing import Protocol


class LLMProvider(Protocol):
    """Port for text generation. Implementations live in infrastructure/llm/.

    In practice, langchain's BaseChatModel satisfies this interface — the
    infrastructure factory returns a BaseChatModel instance. Application
    services type-hint with this Protocol so they stay decoupled from the
    concrete provider.
    """

    def invoke(self, input: str | list) -> object: ...
    async def ainvoke(self, input: str | list) -> object: ...
