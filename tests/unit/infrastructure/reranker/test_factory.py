"""Offline tests for the reranker factory registry."""

import pytest

from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.reranker.cross_encoder import CrossEncoderReranker
from qhld_engine.infrastructure.reranker.factory import create_reranker_from_env
from qhld_engine.infrastructure.reranker.noop import NoOpReranker

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_noop_provider_builds_noop_reranker():
    reranker = create_reranker_from_env(_settings(reranker_provider="noop"))
    assert isinstance(reranker, NoOpReranker)


def test_cross_encoder_provider_builds_cross_encoder():
    settings = _settings(
        reranker_provider="cross_encoder",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_top_n=25,
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model == "BAAI/bge-reranker-v2-m3"
    assert reranker._top_n == 25       # constructing it does not load the model


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        create_reranker_from_env(_settings(reranker_provider="bogus"))
