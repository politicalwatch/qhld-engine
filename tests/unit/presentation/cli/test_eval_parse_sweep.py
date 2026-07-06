"""Offline tests for the `qhld eval parse --models` sweep helpers."""

import pytest
import typer

from qhld_engine.presentation.cli.evaluation import _median, _parse_models

pytestmark = pytest.mark.unit


def test_parse_models_splits_on_first_colon_keeping_ollama_tag():
    specs = _parse_models(
        "ollama:gpt-oss:20b, openai:gpt-5.4-nano-2026-03-17, anthropic:claude-sonnet-5")
    assert specs == [
        ("ollama", "gpt-oss:20b", "ollama:gpt-oss:20b"),
        ("openai", "gpt-5.4-nano-2026-03-17", "openai:gpt-5.4-nano-2026-03-17"),
        ("anthropic", "claude-sonnet-5", "anthropic:claude-sonnet-5"),
    ]


def test_parse_models_rejects_spec_without_provider():
    with pytest.raises(typer.BadParameter):
        _parse_models("gpt-oss:20b, bare-model")


def test_median_odd_and_even():
    assert _median([0.3, 0.1, 0.2]) == 0.2          # odd -> middle
    assert _median([0.2, 0.4]) == 0.3               # even -> mean of the two middles
