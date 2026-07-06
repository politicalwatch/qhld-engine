from langchain_anthropic import ChatAnthropic

from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register

# Reasoning-tier models (Claude 5 family + Opus 4.7/4.8) removed the sampling
# parameters: sending `temperature`/`top_p`/`top_k` returns a 400
# ("temperature is deprecated for this model"). Determinism is steered by prompt
# and effort instead. Haiku 4.5, Sonnet 4.6, Opus 4.6 and older still accept them.
_NO_SAMPLING_PARAMS = frozenset({
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
})


@_register("anthropic")
def create(settings: Settings) -> ChatAnthropic:
    kwargs = {"model": settings.llm_model, "api_key": settings.anthropic_api_key}
    if settings.llm_temperature is not None and settings.llm_model not in _NO_SAMPLING_PARAMS:
        kwargs["temperature"] = settings.llm_temperature
    return ChatAnthropic(**kwargs)
