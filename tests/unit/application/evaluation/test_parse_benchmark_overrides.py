"""Offline tests: RunParseBenchmark threads --models overrides into the settings copy."""

import pytest

from qhld_engine.application.evaluation.parse_benchmark import RunParseBenchmark
from qhld_engine.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def captured(monkeypatch):
    """Capture the Settings the parser factory is built from (no live parser)."""
    seen = {}

    def fake_create(settings):
        seen["settings"] = settings
        return object()

    monkeypatch.setattr(
        "qhld_engine.infrastructure.queryparsing.factory.create_query_parser_from_env",
        fake_create)
    return seen


def _runner():
    return RunParseBenchmark(settings=Settings(_env_file=None))


def test_llm_override_sets_query_parser_llm_provider_and_model(captured):
    _runner()._parser("llm", "openai", "gpt-5.4-nano-2026-03-17")
    passed = captured["settings"]
    assert passed.query_parser_provider == "llm"
    assert passed.query_parser_llm_provider == "openai"
    assert passed.query_parser_llm_model == "gpt-5.4-nano-2026-03-17"


def test_no_override_leaves_parser_llm_settings_default(captured):
    _runner()._parser("rule_based")
    passed = captured["settings"]
    assert passed.query_parser_provider == "rule_based"
    assert passed.query_parser_llm_provider == ""   # untouched fallback
    assert passed.query_parser_llm_model == ""
