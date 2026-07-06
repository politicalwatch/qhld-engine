"""Unit tests for pure mention resolution — no I/O, no spaCy.

Feeds raw NER-style spans + a fake deputy catalog through the resolver and asserts
the canonicalization, dedupe/count, honorific stripping and ambiguity guard.
"""

import pytest

from qhld_engine.domain.speeches.mentions import (
    COMMON_WORD_SURNAMES,
    NON_DEPUTY_SURNAMES,
    build_deputy_index,
    context_excluded_surnames,
    normalize_span,
    resolve_mentions,
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
    # "al" is the contracted article (a + el), not part of the name.
    assert normalize_span("Al señor López") == "lópez"


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


# --- excluded (non-deputy) surnames ----------------------------------------

AZNAR_DEP = FakeDeputy("e1", "Aznar Teruel, Evarist")
GAMARRA = FakeDeputy("e2", "Gamarra Ruiz-Clavijo, Concepción")
ALBARES = FakeDeputy("e3", "Albares Bueno, José Manuel")
FEIJOO = FakeDeputy("e4", "Núñez Feijóo, Alberto")
INDEX_EXCL = build_deputy_index([AZNAR_DEP, GAMARRA, ALBARES, FEIJOO])


def test_referent_homonym_excluded_on_first_surname():
    # "Aznar" fuzzy-matches the deputy Aznar Teruel but denotes the ex-PM; the deputy's
    # OWN first surname is flagged, so it is dropped.
    assert resolve_mentions(["Aznar"], INDEX_EXCL, 90, frozenset({"aznar"})) == []


def test_mismatch_secondary_surname_excluded():
    # "Clavijo" (the Canarias president) resolves to Gamarra via her SECOND surname —
    # dropped because her first surname (Gamarra) is not in the span.
    assert resolve_mentions(["Clavijo"], INDEX_EXCL, 90, frozenset({"clavijo"})) == []


def test_full_name_containing_flagged_token_survives():
    # A genuine full-name mention that merely contains the flagged token is kept.
    mentions = resolve_mentions(
        ["Gamarra Ruiz-Clavijo"], INDEX_EXCL, 90, frozenset({"clavijo"}))
    assert _names(mentions) == {"Gamarra Ruiz-Clavijo, Concepción"}


def test_common_word_surname_excluded():
    assert resolve_mentions(["Bueno"], INDEX_EXCL, 90, COMMON_WORD_SURNAMES) == []


def test_deputy_known_by_second_surname_survives_denylist():
    # Regression guard: the denylist must not suppress a deputy universally named by
    # their SECOND surname (Feijóo of Núñez Feijóo) — "feijóo" is not flagged.
    mentions = resolve_mentions(["Feijóo"], INDEX_EXCL, 90, NON_DEPUTY_SURNAMES)
    assert _names(mentions) == {"Núñez Feijóo, Alberto"}


def test_no_exclusion_by_default():
    assert _names(resolve_mentions(["Aznar"], INDEX_EXCL, 90)) == {
        "Aznar Teruel, Evarist"}


# --- context-cue exclusion (speech-scoped) ---------------------------------

def test_context_cue_magistrate():
    text = "Soldados del régimen como Macías, actual magistrado del Tribunal Constitucional."
    assert "macías" in context_excluded_surnames(text)


def test_context_cue_dictatorship_flags_franco():
    text = "La ley debe ser la orgánica de Franco, la manera fina de llamarla en la Dictadura."
    assert "franco" in context_excluded_surnames(text)


def test_context_cue_expresidente():
    text = "Lo dijo el expresidente del Gobierno Aznar en aquella ocasión."
    assert "aznar" in context_excluded_surnames(text)


def test_no_context_cue_yields_empty():
    assert context_excluded_surnames("El señor Sánchez habló de vivienda.") == frozenset()
