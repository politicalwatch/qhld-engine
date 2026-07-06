"""Unit tests for the LLM provider registry + factory — no network.

Ported from vinculante and trimmed to the surface qhld-engine kept: the registry
and the single ``create_llm_from_env``. The recording-provider tests pin the
wiring (factory passes the right Settings to the right provider); the real-build
tests are the value-add — they exercise the actual adapters and confirm the
langchain provider packages are installed and construct offline with dummy keys.
"""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import qhld_engine.infrastructure.llm.factory as factory_module
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.llm.factory import create_llm_from_env

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    # qhld Settings has no required fields; _env_file=None keeps the repo .env out.
    return Settings(_env_file=None, **overrides)


class _RecordingLLM:
    """Fake provider that records the Settings it received."""

    received: Settings | None = None

    def __init__(self, settings: Settings) -> None:
        _RecordingLLM.received = settings


def test_create_llm_from_env_uses_llm_provider(monkeypatch):
    monkeypatch.setitem(factory_module._PROVIDERS, "anthropic", _RecordingLLM)
    s = _settings(llm_provider="anthropic", llm_model="claude-y", llm_temperature=0.5)
    create_llm_from_env(s)
    assert _RecordingLLM.received.llm_provider == "anthropic"
    assert _RecordingLLM.received.llm_model == "claude-y"
    assert _RecordingLLM.received.llm_temperature == pytest.approx(0.5)


def test_create_llm_unknown_provider_raises():
    s = _settings(llm_provider="unknown_xyz")
    with pytest.raises(ValueError, match="unknown_xyz"):
        create_llm_from_env(s)


def test_all_llm_providers_registered():
    assert {"anthropic", "openai", "ollama", "google", "mistral"} <= set(
        factory_module._PROVIDERS
    )


def test_anthropic_omits_temperature_for_reasoning_models():
    # Claude 5 family / Opus 4.7-4.8 reject sampling params with a 400 — the
    # adapter must not send temperature for them.
    llm = create_llm_from_env(
        _settings(llm_provider="anthropic", llm_model="claude-sonnet-5",
                  llm_temperature=0.0, anthropic_api_key="x"))
    assert llm.temperature is None


def test_anthropic_keeps_temperature_for_older_models():
    llm = create_llm_from_env(
        _settings(llm_provider="anthropic", llm_model="claude-haiku-4-5",
                  llm_temperature=0.0, anthropic_api_key="x"))
    assert llm.temperature == pytest.approx(0.0)


@pytest.mark.parametrize(
    "provider, expected_cls",
    [
        ("anthropic", ChatAnthropic),
        ("openai", ChatOpenAI),
        ("ollama", ChatOllama),
        ("google", ChatGoogleGenerativeAI),
        ("mistral", ChatMistralAI),
    ],
)
def test_each_provider_builds_real_chatmodel(provider, expected_cls):
    # Dummy keys are enough: langchain chat models don't hit the network at
    # construction. This proves the copied adapters import and wire correctly.
    s = _settings(
        llm_provider=provider,
        anthropic_api_key="x",
        openai_api_key="x",
        google_api_key="x",
        mistral_api_key="x",
    )
    assert isinstance(create_llm_from_env(s), expected_cls)
