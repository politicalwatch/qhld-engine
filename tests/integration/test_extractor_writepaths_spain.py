"""Integration regression tests for the Spain extractor write-path ``_id`` bug.

These Spain write-paths reach the empty-construction bug only with a reachable
MongoDB (repo lookups/saves), so they use the throwaway ``mongo_db`` fixture.
Pre-fix each fails: the model is constructed without the now-required ``_id``
(``GroupsExtractor`` swallows the error so the group simply never persists; the
others raise ``ValidationError``).
"""

import hashlib
import json

import pytest

from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
from tipi_data.repositories.amendments import Amendments

from qhld_engine.extractors.spain.groups import GroupsExtractor
from qhld_engine.extractors.spain.initiative_extractors.initiative_extractor import (
    InitiativeExtractor,
)
from qhld_engine.extractors.spain.initiative_extractors.amendments.totallity_amendments import (
    TotallityAmendments,
)

pytestmark = pytest.mark.integration


# --- Spain GroupsExtractor ----------------------------------------------------

def test_groups_extractor_persists_group(mongo_db, tmp_path):
    groups_file = tmp_path / "groups.json"
    groups_file.write_text(json.dumps([
        {
            "_id": "gps",
            "name": "Grupo Parlamentario Socialista",
            "shortname": "GS",
            "color": "#ff0000",
            "parties": ["PSOE"],
        }
    ]))

    GroupsExtractor().load(str(groups_file))

    group = ParliamentaryGroups.get("gps")
    assert group.name == "Grupo Parlamentario Socialista"
    assert group.parties == ["PSOE"]


# --- Spain InitiativeExtractor: new-initiative branch -------------------------

def test_spain_initiative_extractor_creates_new_initiative(mongo_db):
    from types import SimpleNamespace

    response = SimpleNamespace(
        text="<html><body></body></html>",
        url="https://www.congreso.es/?_iniciativas_id=184/000001",
    )

    extractor = InitiativeExtractor(response, {}, [], {}, [])

    assert extractor.is_a_new_initiative is True
    assert extractor.initiative.id == ""  # placeholder; real id set in extract()


# --- Spain TotallityAmendments ------------------------------------------------

def test_totallity_amendment_is_saved(mongo_db):
    reference = "122/000001"
    bulletin = "BOCG-14-A-1-2"
    content = (
        "Texto previo ENMIENDA NÚM. 1\n"
        "(Grupo Parlamentario Socialista)\n"
        "JUSTIFICACIÓN\n"
        "Texto de la justificación de la enmienda."
    )

    TotallityAmendments(reference, content, bulletin).extract()

    amendments = Amendments.by_reference(reference)
    assert len(amendments) == 1
    expected_id = "{}/{}/{}".format(
        reference, hashlib.md5(bulletin.encode()).hexdigest(), "1"
    )
    assert amendments[0].id == expected_id
