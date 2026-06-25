"""Integration regression test for the Paraguay extractor write-path ``_id`` bug.

Paraguay is an almost-legacy extractor, kept in its own file separate from Spain.
The create branch of ``__create_or_update`` builds an ``Initiative`` empty, which
pre-fix raises ``ValidationError`` for the now-required ``_id``. Reaching it needs
a MongoDB lookup, so this uses the throwaway ``mongo_db`` fixture.
"""

from datetime import datetime

import pytest

from tipi_data.models.initiative import Initiative
from tipi_data.repositories.initiatives import Initiatives

from qhld_engine.extractors.paraguay import initiatives as paraguay_initiatives
from qhld_engine.extractors.paraguay.initiatives import InitiativesExtractor

pytestmark = pytest.mark.integration


REMOTE = {
    "idProyecto": 12345,
    "expedienteCamara": "D-12345",
    "acapite": "Proyecto de prueba",
    "tipoProyecto": "Proyecto de ley",
    "descripcionEtapa": "INICIATIVA",
    "descripcionSubEtapa": "Ingreso",
    "estadoProyecto": "En estudio",
    "origenProyecto": "Diputados",
    "appURL": "https://silpy.congreso.gov.py/proyecto/12345",
    "fechaIngresoExpediente": "01/02/2023",
    "iniciativa": "Algún proponente",
}


def test_paraguay_initiatives_creates_new_initiative(mongo_db, monkeypatch):
    monkeypatch.setattr(
        paraguay_initiatives.LegislativePeriod, "get", lambda self: "2023-2028"
    )
    # Stub the network + content-loading step; it is unrelated to the _id write-path
    # under test here.
    monkeypatch.setattr(
        InitiativesExtractor,
        "_InitiativesExtractor__load_more_data",
        lambda self, initiative: None,
    )

    # Name-mangled private method — the empty ``Initiative()`` lives in its except branch.
    InitiativesExtractor()._InitiativesExtractor__create_or_update(REMOTE)

    saved = Initiatives.get("12345")
    assert saved.reference == "D-12345"
    assert saved.title == "Proyecto de prueba"


def test_paraguay_update_preserves_existing_extra(mongo_db, monkeypatch):
    """On the *update* branch, ``extra`` (proponente, ignored_attachments, content
    bookkeeping) accumulated on a prior run must survive. Pre-fix the broken
    ``"extra" not in initiative`` membership check was always True, so ``extra`` was
    re-initialized on every run, dropping ``ignored_attachments``."""
    monkeypatch.setattr(
        paraguay_initiatives.LegislativePeriod, "get", lambda self: "2023-2028"
    )
    monkeypatch.setattr(
        InitiativesExtractor,
        "_InitiativesExtractor__load_more_data",
        lambda self, initiative: None,
    )

    Initiatives.save(
        Initiative(
            id="12345",
            updated=datetime(2023, 2, 1),
            extra={
                "proponente": "Proponente original",
                "ignored_attachments": [99],
                "content_reference": "Dictamen",
                "content_counter": 3,
            },
        )
    )

    InitiativesExtractor()._InitiativesExtractor__create_or_update(REMOTE)

    saved = Initiatives.get("12345")
    assert saved.extra["ignored_attachments"] == [99]
    assert saved.extra["content_reference"] == "Dictamen"
    assert saved.extra["content_counter"] == 3
