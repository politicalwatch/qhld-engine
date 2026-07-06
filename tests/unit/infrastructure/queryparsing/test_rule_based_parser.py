"""Unit tests for the rule-based (spaCy + dateparser) query parser.

Skipped when spaCy or dateparser is not installed.
"""

from datetime import date
from importlib.util import find_spec

import pytest

pytestmark = pytest.mark.unit

if find_spec("spacy") is None or find_spec("dateparser") is None:
    pytest.skip("spaCy/dateparser not installed", allow_module_level=True)

from qhld_engine.infrastructure.queryparsing.rule_based import RuleBasedQueryParser

TODAY = date(2026, 7, 3)


@pytest.fixture(scope="module")
def parser():
    return RuleBasedQueryParser()


def test_person_entity_becomes_speaker(parser):
    parsed = parser.parse("intervenciones de María Jesús Montero sobre vivienda", TODAY)
    assert parsed.speaker == "María Jesús Montero"
    assert "montero" not in parsed.semantic_query.lower()


def test_org_entity_becomes_group(parser):
    parsed = parser.parse("intervenciones del PSOE sobre vivienda", TODAY)
    assert parsed.group_or_party == "PSOE"


def test_title_regex_and_language(parser):
    parsed = parser.parse("qué ha dicho la ministra de defensa sobre el rearme en catalán", TODAY)
    assert parsed.speaker_title == "ministra de defensa"
    assert parsed.lang == "ca"
    assert "rearme" in parsed.semantic_query.lower()


def test_relative_date_range(parser):
    parsed = parser.parse("intervenciones de Pedro Sánchez en los últimos tres meses", TODAY)
    assert parsed.date_from == "2026-04-03"
    assert parsed.date_to == "2026-07-03"
