"""Integration regression test for the Paraguay extractor write-path ``_id`` bug.

Paraguay is an almost-legacy extractor, kept in its own file separate from Spain.
The create branch of ``__create_or_update`` builds an ``Initiative`` empty, which
pre-fix raises ``ValidationError`` for the now-required ``_id``. Reaching it needs
a MongoDB lookup, so this uses the throwaway ``mongo_db`` fixture.
"""

import pytest

from tipi_data.repositories.initiatives import Initiatives

from qhld_engine.extractors.paraguay import initiatives as paraguay_initiatives
from qhld_engine.extractors.paraguay.initiatives import InitiativesExtractor

pytestmark = pytest.mark.integration


def test_paraguay_initiatives_creates_new_initiative(mongo_db, monkeypatch):
    monkeypatch.setattr(
        paraguay_initiatives.LegislativePeriod, "get", lambda self: "2023-2028"
    )
    # Stub the network + content-loading step. It is unrelated to the _id write-path
    # under test here and has its own latent bug (its __untag sets fields the
    # pydantic Initiative model does not declare); kept out of scope.
    monkeypatch.setattr(
        InitiativesExtractor,
        "_InitiativesExtractor__load_more_data",
        lambda self, initiative: None,
    )

    remote = {
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

    # Name-mangled private method — the empty ``Initiative()`` lives in its except branch.
    InitiativesExtractor()._InitiativesExtractor__create_or_update(remote)

    saved = Initiatives.get("12345")
    assert saved.reference == "D-12345"
    assert saved.title == "Proyecto de prueba"
