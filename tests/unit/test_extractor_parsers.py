"""Unit tests for pure extractor parsing/logic helpers — no HTTP, no DB.

These cover the country extractors' small deterministic helpers: roman-numeral
formatting and status classification for Spain, and final-state / workflow-phase
logic for Paraguay. The DB-touching helpers in the same modules (e.g.
``has_finished``, ``get_current_status``) are exercised by the integration tier.
"""

import pytest

from extractors.spain.utils import int_to_roman
from extractors.spain.initiative_extractors.initiative_status import (
    get_status,
    is_final_status,
)
from extractors.paraguay.initiatives_status import is_final_state
from extractors.paraguay.initiatives_attachments import get_next_phase

pytestmark = pytest.mark.unit


# --- Spain: int_to_roman ------------------------------------------------------

@pytest.mark.parametrize("num,expected", [
    (0, ""),
    (4, "IV"),
    (9, "IX"),
    (14, "XIV"),
    (40, "XL"),
    (1990, "MCMXC"),
    (2024, "MMXXIV"),
])
def test_int_to_roman(num, expected):
    assert int_to_roman(num) == expected


# --- Spain: get_status / is_final_status -------------------------------------

def test_get_status_type_070_always_approved():
    assert get_status(history=[], initiative_type="070") == "Aprobada"


def test_get_status_no_history_is_unknown():
    assert get_status(history=[], initiative_type="001") == "Desconocida"


def test_get_status_matches_status_map_on_latest_history_item():
    # "Convalidado" maps to "Convalidada" (no type include/exclude constraints).
    assert get_status(history=["Convalidado"], initiative_type="001") == "Convalidada"


def test_get_status_unmatched_history_is_unknown():
    assert get_status(history=["nothing matches this"], initiative_type="001") == "Desconocida"


@pytest.mark.parametrize("status,expected", [
    ("Aprobada", True),
    ("En tramitación", False),
    ("Desconocida", False),
])
def test_is_final_status(status, expected):
    assert is_final_status(status) is expected


# --- Paraguay: is_final_state -------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("archivado", True),
    ("Publicado en Gaceta", True),
    ("RETIRADO", True),
    ("En estudio", False),
])
def test_is_final_state(status, expected):
    assert is_final_state(status) is expected


# --- Paraguay: get_next_phase -------------------------------------------------

def test_get_next_phase_walks_the_workflow():
    assert get_next_phase("") == (0, "INICIATIVA")
    assert get_next_phase("INICIATIVA") == (1, "SANCIÓN COMPLETA")
    assert get_next_phase("SANCIÓN COMPLETA") == (2, "LEY")


def test_get_next_phase_past_last_returns_sentinel():
    assert get_next_phase("LEY") == (-1, "")
