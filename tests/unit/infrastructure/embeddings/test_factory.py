"""Unit tests for the embeddings provider registry + factory — no network.

Ported from vinculante; same shape as the LLM factory tests.
"""

import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

import qhld_engine.infrastructure.embeddings.factory as factory_module
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.embeddings.factory import create_embedder_from_env

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _RecordingEmbedder:
    received: Settings | None = None

    def __init__(self, settings: Settings) -> None:
        _RecordingEmbedder.received = settings


def test_create_embedder_uses_embedding_provider(monkeypatch):
    monkeypatch.setitem(factory_module._PROVIDERS, "google", _RecordingEmbedder)
    s = _settings(embedding_provider="google", embedding_model="embedding-001")
    create_embedder_from_env(s)
    assert _RecordingEmbedder.received.embedding_provider == "google"
    assert _RecordingEmbedder.received.embedding_model == "embedding-001"


def test_create_embedder_unknown_provider_raises():
    s = _settings(embedding_provider="unknown_xyz")
    with pytest.raises(ValueError, match="unknown_xyz"):
        create_embedder_from_env(s)


def test_all_embedding_providers_registered():
    assert {"openai", "google", "ollama"} <= set(factory_module._PROVIDERS)


@pytest.mark.parametrize(
    "provider, expected_cls",
    [
        ("openai", OpenAIEmbeddings),
        ("google", GoogleGenerativeAIEmbeddings),
        ("ollama", OllamaEmbeddings),
    ],
)
def test_each_provider_builds_real_embedder(provider, expected_cls):
    s = _settings(
        embedding_provider=provider,
        openai_api_key="x",
        google_api_key="x",
    )
    assert isinstance(create_embedder_from_env(s), expected_cls)
