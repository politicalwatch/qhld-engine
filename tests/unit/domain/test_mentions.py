"""Unit tests for pure mention resolution — no I/O, no spaCy.

Feeds raw NER-style spans + a fake deputy catalog through the resolver and asserts
the canonicalization, dedupe/count, honorific stripping and ambiguity guard.
"""

import pytest

from qhld_engine.domain.speeches.mentions import (
    COMMON_WORD_SURNAMES,
    build_deputy_index,
    build_person_index,
    build_surname_gazetteer,
    context_excluded_surnames,
    make_person_entry,
    normalize_span,
    resolve_mentions,
    resolve_person,
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
    assert mentions[0].person_id == "d2"
    assert mentions[0].person_type == "deputy"


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
    # Regression guard: the exclusion set must not suppress a deputy universally named
    # by their SECOND surname (Feijóo of Núñez Feijóo) — "feijóo" is not flagged.
    mentions = resolve_mentions(
        ["Feijóo"], INDEX_EXCL, 90, frozenset({"aznar", "suárez", "clavijo"}))
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


def test_no_context_cue_yields_empty():
    assert context_excluded_surnames("El señor Sánchez habló de vivienda.") == frozenset()


# --- tie-break (recall) ----------------------------------------------------

JUAN_BRAVO = FakeDeputy("t1", "Bravo Baena, Juan")
AITOR_ESTEBAN = FakeDeputy("t2", "Esteban Bravo, Aitor")
GONZALEZ_PONS = FakeDeputy("t3", "González Pons, Esteban")
RAMOS_ESTEBAN = FakeDeputy("t4", "Ramos Esteban, César Joaquín")
PEDRO_SANCHEZ = FakeDeputy("t5", "Sánchez Pérez-Castejón, Pedro")
CESAR_SANCHEZ = FakeDeputy("t6", "Sánchez Pérez, César")
MUNOZ_IGLESIA = FakeDeputy("t7", "Muñoz de la Iglesia, Ester")
MUNOZ_ABRINES = FakeDeputy("t8", "Muñoz Abrines, Pedro")
INDEX_TIE = build_deputy_index([
    JUAN_BRAVO, AITOR_ESTEBAN, GONZALEZ_PONS, RAMOS_ESTEBAN,
    PEDRO_SANCHEZ, CESAR_SANCHEZ, MUNOZ_IGLESIA, MUNOZ_ABRINES])


def test_tie_broken_toward_first_surname_holder():
    # "Bravo" is Juan Bravo's FIRST surname but Aitor Esteban Bravo's SECOND — resolves
    # to the former instead of dropping as an ambiguous tie.
    assert _names(resolve_mentions(["el señor Bravo"], INDEX_TIE, 90)) == {
        "Bravo Baena, Juan"}


def test_tie_first_surname_beats_given_name_and_second_surname():
    # "Esteban" is Aitor Esteban's FIRST surname, González Pons's GIVEN name and Ramos
    # Esteban's SECOND surname — resolves to the first-surname holder.
    assert _names(resolve_mentions(["Esteban"], INDEX_TIE, 90)) == {
        "Esteban Bravo, Aitor"}


def test_tie_broken_by_exact_token_order():
    # Both share first surname "Sánchez" and tie at token_set_ratio 100; the exact
    # surname order picks Pedro over the shorter "Sánchez Pérez".
    assert _names(resolve_mentions(["Sánchez Pérez-Castejón"], INDEX_TIE, 90)) == {
        "Sánchez Pérez-Castejón, Pedro"}


def test_ambiguous_shared_first_surname_still_drops():
    # Two deputies hold "Muñoz" as their first surname → genuinely ambiguous → dropped.
    assert resolve_mentions(["Muñoz"], INDEX_TIE, 90) == []


# --- resolve_person (query side) -------------------------------------------

def test_resolve_person_surname_resolves_to_deputy():
    entry = resolve_person("Montero", INDEX, 90)
    assert entry is not None
    assert (entry.person_id, entry.name) == ("d2", "Montero Cuadrado, María Jesús")


def test_resolve_person_full_name_resolves():
    entry = resolve_person("Pedro Sánchez", INDEX, 90)
    assert entry.person_id == "d1"


def test_resolve_person_ambiguous_surname_is_none():
    # "García" is borne by two deputies at the same score → ambiguous → not resolved.
    assert resolve_person("García", INDEX, 90) is None


def test_resolve_person_unknown_is_none():
    assert resolve_person("Winston Churchill", INDEX, 90) is None


def test_resolve_person_empty_after_normalize_is_none():
    assert resolve_person("Su Señoría", INDEX, 90) is None


# --- surname gazetteer -----------------------------------------------------

def test_gazetteer_keeps_distinctive_first_surnames_and_compound_parts():
    deputies = [
        FakeDeputy("g1", "Vallugera Balañà, Pilar"),
        FakeDeputy("g2", "Grande-Marlaska Gómez, Fernando"),
        FakeDeputy("g3", "García López, Ana"),
        FakeDeputy("g4", "García Ruiz, Juan"),  # 'García' shared → excluded
    ]
    terms = build_surname_gazetteer(deputies)
    assert "Vallugera" in terms
    assert "Grande" in terms and "Marlaska" in terms  # hyphenated compound split
    assert "García" not in terms  # borne by two deputies → not distinctive
    assert all(t == t for t in terms) and terms == sorted(terms)


# --- non-deputy people (curated catalog + overrides) -----------------------

# A deputy who shares a surname with a famous non-deputy, plus one who doesn't.
GAMARRA = FakeDeputy("gamarra", "Gamarra Ruiz-Clavijo, Concepción")
AZNAR_DEP2 = FakeDeputy("aznar-teruel", "Aznar Teruel, Evarist")
FEIJOO2 = FakeDeputy("nunez-feijoo-alberto", "Núñez Feijóo, Alberto")

CURATED = [
    make_person_entry("fernando-clavijo", "regional_president", "Clavijo Batlle, Fernando",
                      aliases=["Clavijo", "Fernando Clavijo"], overrides_deputy=True),
    make_person_entry("jose-maria-aznar", "former_pm", "Aznar López, José María",
                      aliases=["Aznar", "José María Aznar"], overrides_deputy=True),
    make_person_entry("isabel-diaz-ayuso", "regional_president", "Díaz Ayuso, Isabel",
                      aliases=["Ayuso", "Díaz Ayuso"]),
    make_person_entry("donald-trump", "foreign_leader", "Trump, Donald", aliases=["Trump"]),
    make_person_entry("felipe-vi", "head_of_state", "Felipe VI",
                      aliases=["Felipe VI", "su majestad"]),
]
PERSON_INDEX = build_person_index([GAMARRA, AZNAR_DEP2, FEIJOO2], CURATED)


def _one(span):
    m = resolve_mentions([span], PERSON_INDEX, 90)
    return (m[0].name, m[0].person_type) if m else None


def test_non_deputy_resolves_to_catalog_person():
    assert _one("Ayuso") == ("Díaz Ayuso, Isabel", "regional_president")
    assert _one("Trump") == ("Trump, Donald", "foreign_leader")


def test_override_wins_over_colliding_deputy_on_bare_surname():
    # "Clavijo" is the Canarias president, not the deputy Gamarra Ruiz-Clavijo (2nd
    # surname); "Aznar" is the ex-PM, not the deputy Aznar Teruel.
    assert _one("Clavijo") == ("Clavijo Batlle, Fernando", "regional_president")
    assert _one("Aznar") == ("Aznar López, José María", "former_pm")


def test_deputy_full_name_beats_override():
    # The deputy's OWN full name outranks the surname-sharing override.
    assert _one("Gamarra Ruiz-Clavijo") == ("Gamarra Ruiz-Clavijo, Concepción", "deputy")


def test_deputy_second_surname_still_wins_when_no_override():
    # Feijóo has no override entry and only the deputy matches → deputy, unchanged.
    assert _one("Feijóo") == ("Núñez Feijóo, Alberto", "deputy")


def test_king_matched_by_explicit_alias_not_bare_common_noun():
    assert _one("su majestad") == ("Felipe VI", "head_of_state")
    assert _one("Felipe VI") == ("Felipe VI", "head_of_state")
    # bare "rey" is a common noun (and collides with the real deputy 'Rey de las
    # Heras'), deliberately not an alias → the King is not matched from it.
    assert _one("rey") is None


def test_non_deputy_never_excluded_by_deputy_denylist():
    # The exclusion set only guards deputy resolutions; a resolved non-deputy is kept
    # even if a homonymous surname would be flagged for deputies.
    mentions = resolve_mentions(["Aznar"], PERSON_INDEX, 90, frozenset({"aznar"}))
    assert _names(mentions) == {"Aznar López, José María"}


def test_resolve_person_resolves_non_deputy():
    entry = resolve_person("Ayuso", PERSON_INDEX, 90)
    assert (entry.person_id, entry.person_type) == ("isabel-diaz-ayuso", "regional_president")


def test_override_second_surname_does_not_hijack_ambiguous_tie():
    # 'Aznar López' (override ex-PM) shares his SECOND surname with a bare 'López', which
    # is ambiguous across several deputies. The override must not fire on a secondary
    # token: 'López' stays ambiguous (dropped), while the ex-PM still resolves from 'Aznar'.
    lopez1 = FakeDeputy("lopez-cano", "López Cano, Ignacio")
    lopez2 = FakeDeputy("lopez-alvarez", "López Álvarez, Patxi")
    aznar = make_person_entry("jose-maria-aznar", "former_pm", "Aznar López, José María",
                              aliases=["Aznar"], overrides_deputy=True)
    index = build_person_index([lopez1, lopez2], [aznar])
    assert resolve_mentions(["López"], index, 90) == []
    assert resolve_mentions(["Aznar"], index, 90)[0].person_type == "former_pm"


def test_deputy_wins_tie_over_nonoverride_nondeputy():
    # A bootstrapped minister shares a surname with a deputy ("Rego"): the deputy is the
    # primary referent and wins — a non-override non-deputy never blocks a deputy.
    deputy = FakeDeputy("rego-candamil-nestor", "Rego Candamil, Néstor")
    minister = make_person_entry("sira-rego", "minister", "Rego, Sira Abed")
    index = build_person_index([deputy], [minister])
    m = resolve_mentions(["Rego"], index, 90)
    assert (m[0].name, m[0].person_type) == ("Rego Candamil, Néstor", "deputy")
