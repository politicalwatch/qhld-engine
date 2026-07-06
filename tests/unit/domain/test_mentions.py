"""Unit tests for pure mention resolution — no I/O, no spaCy.

Feeds raw NER-style spans + a fake deputy catalog through the resolver and asserts
the canonicalization, dedupe/count, honorific stripping and ambiguity guard.
"""

import pytest

from qhld_engine.domain.speeches.mentions import (
    build_deputy_index, normalize_span, resolve_mentions,
)

pytestmark = pytest.mark.unit


class FakeDeputy:
    """Duck-types the bits of tipi_data ``Deputy`` the index needs."""

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


PEDRO = FakeDeputy("d1", "Sánchez Pérez-Castejón, Pedro")
MONTERO = FakeDeputy("d2", "Montero Cuadrado, María Jesús")
GARCIA_A = FakeDeputy("d3", "García López, Ana")
GARCIA_J = FakeDeputy("d4", "García Ruiz, Juan")

INDEX = build_deputy_index([PEDRO, MONTERO, GARCIA_A, GARCIA_J])


def _names(mentions):
    return {m.name for m in mentions}


# --- normalization ---------------------------------------------------------

def test_normalize_strips_honorifics_and_articles():
    assert normalize_span("el señor Sánchez") == "sánchez"
    assert normalize_span("doña María Jesús Montero") == "maría jesús montero"


def test_normalize_drops_pure_honorific_and_too_short():
    assert normalize_span("Su Señoría") == ""
    assert normalize_span("el") == ""
    assert normalize_span(",.") == ""


# --- resolution ------------------------------------------------------------

def test_surname_only_resolves_to_canonical_name():
    mentions = resolve_mentions(["Montero"], INDEX, 90)
    assert _names(mentions) == {"Montero Cuadrado, María Jesús"}
    assert mentions[0].deputy_id == "d2"


def test_full_name_and_honorific_forms_merge_into_one_mention():
    spans = ["el señor Sánchez", "Pedro Sánchez", "Sánchez"]
    mentions = resolve_mentions(spans, INDEX, 90)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.name == "Sánchez Pérez-Castejón, Pedro"
    assert m.count == 3
    assert m.surface_forms == ["Pedro Sánchez", "Sánchez", "el señor Sánchez"]


def test_ambiguous_bare_surname_is_dropped():
    # "García" matches two deputies at the same top score → precision-safe drop.
    assert resolve_mentions(["García"], INDEX, 90) == []


def test_ambiguity_resolved_when_given_name_disambiguates():
    mentions = resolve_mentions(["Ana García"], INDEX, 90)
    assert _names(mentions) == {"García López, Ana"}


def test_unknown_person_below_threshold_is_dropped():
    assert resolve_mentions(["Winston Churchill"], INDEX, 90) == []


def test_result_sorted_by_count_desc():
    spans = ["Montero", "Sánchez", "Sánchez"]
    mentions = resolve_mentions(spans, INDEX, 90)
    assert [m.name for m in mentions] == [
        "Sánchez Pérez-Castejón, Pedro", "Montero Cuadrado, María Jesús"]


def test_empty_and_honorific_only_spans_yield_nothing():
    assert resolve_mentions(["Su Señoría", "", "  "], INDEX, 90) == []
