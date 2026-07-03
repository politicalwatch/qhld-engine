"""Unit tests for the LLM query parser — offline, with a fake chat model."""

from datetime import date

import pytest

from qhld_engine.domain.ports.query_parser import ParsedQuery
from qhld_engine.infrastructure.config.settings import Settings
from qhld_engine.infrastructure.queryparsing.factory import create_query_parser_from_env
from qhld_engine.infrastructure.queryparsing.llm import LLMQueryParser

pytestmark = pytest.mark.unit


class _FakeStructured:
    def __init__(self, captured, result):
        self._captured = captured
        self._result = result

    def invoke(self, messages):
        self._captured["messages"] = messages
        return self._result


class _FakeChat:
    def __init__(self, captured, result):
        self._captured = captured
        self._result = result

    def with_structured_output(self, schema):
        self._captured["schema"] = schema
        return _FakeStructured(self._captured, self._result)


@pytest.fixture
def fake_llm(monkeypatch):
    captured = {}
    result = ParsedQuery(semantic_query="financiación autonómica", speaker="Montero")

    def fake_create(settings):
        captured["settings"] = settings
        return _FakeChat(captured, result)

    monkeypatch.setattr(
        "qhld_engine.infrastructure.llm.factory.create_llm_from_env", fake_create)
    return captured


def test_factory_builds_llm_parser():
    parser = create_query_parser_from_env(Settings(_env_file=None, query_parser_provider="llm"))
    assert isinstance(parser, LLMQueryParser)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown query parser provider"):
        create_query_parser_from_env(Settings(_env_file=None, query_parser_provider="bogus"))


def test_parse_binds_schema_and_returns_structured_result(fake_llm):
    parser = LLMQueryParser(Settings(_env_file=None))
    result = parser.parse("intervenciones de Montero sobre financiación", date(2025, 7, 3))
    assert fake_llm["schema"] is ParsedQuery
    assert result.semantic_query == "financiación autonómica"
    assert result.speaker == "Montero"


def test_parse_injects_today_into_system_prompt(fake_llm):
    parser = LLMQueryParser(Settings(_env_file=None))
    parser.parse("algo del último año", date(2025, 7, 3))
    system, human = fake_llm["messages"]
    assert "2025-07-03" in system.content
    assert human.content == "algo del último año"


def test_decoupled_parser_llm_settings_override_main_llm(fake_llm):
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic", llm_model="claude-sonnet-4-6",
        query_parser_llm_provider="ollama", query_parser_llm_model="qwen2.5")
    LLMQueryParser(settings).parse("hola", date(2025, 7, 3))
    passed = fake_llm["settings"]
    assert passed.llm_provider == "ollama"
    assert passed.llm_model == "qwen2.5"


def test_empty_parser_llm_settings_fall_back_to_main_llm(fake_llm):
    settings = Settings(
        _env_file=None, llm_provider="anthropic", llm_model="claude-sonnet-4-6")
    LLMQueryParser(settings).parse("hola", date(2025, 7, 3))
    passed = fake_llm["settings"]
    assert passed.llm_provider == "anthropic"
    assert passed.llm_model == "claude-sonnet-4-6"
