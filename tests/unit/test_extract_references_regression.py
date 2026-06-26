"""Unit regression for the reference-explosion bug — no DB, no network.

The pymongo/pydantic migration made ``Initiatives.get_non_answers_refs()`` return
Pydantic ``Initiative`` models instead of mongoengine documents. ``extract_references``
asks ``if 'reference' not in initiative`` to skip refs-less docs — but Pydantic models
had no ``__contains__``, so ``in`` fell back to ``__iter__`` (which yields
``(name, value)`` pairs) and the check was *always* True. Every existing initiative
was skipped, the per-type DB count came out 0, and the tail loop regenerated
``origin_total`` references **per type, from reference 1** — the whole Congress corpus
(~61k) instead of the daily delta.

This test feeds real ``Initiative`` models through ``extract_references`` and asserts
the already-stored, final-status initiatives are NOT re-queued. It is RED against a
``tipi_data`` without ``__contains__`` and GREEN once the fix is in.
"""

import pytest

from tipi_data.models.initiative import Initiative

from qhld_engine.extractors.spain import initiatives as initiatives_module
from qhld_engine.extractors.spain.initiatives import InitiativesExtractor

pytestmark = pytest.mark.unit

TYPE_CODE = "121"
TYPE_TITLE = "Proyecto de ley"


def _make_extractor(monkeypatch, stored, totals, types):
    """An ``InitiativesExtractor`` with the DB-touching ``__init__`` bypassed and
    the data layer stubbed to controlled values."""
    extractor = InitiativesExtractor.__new__(InitiativesExtractor)
    extractor.all_references = []
    extractor.totals_by_type = {}
    extractor.SAFETY_EXTRACTION_GAP = 3

    monkeypatch.setattr(extractor, "sync_totals", lambda: extractor.totals_by_type.update(totals))
    monkeypatch.setattr(extractor, "get_types", lambda: types)
    monkeypatch.setattr(initiatives_module.Initiatives, "get_non_answers_refs", staticmethod(lambda: stored))
    return extractor


def test_existing_initiatives_are_not_regenerated(monkeypatch):
    # Five consecutive, final-status initiatives already stored; Congress reports
    # the same five. The correct delta is just the safety-gap tail.
    stored = [
        Initiative.model_validate(
            {"_id": f"{TYPE_CODE}-00000{n}", "reference": f"{TYPE_CODE}/00000{n}",
             "initiative_type_alt": TYPE_TITLE, "status": "Concluido (Aprobado)"}
        )
        for n in range(1, 6)
    ]
    types = [{"code": TYPE_CODE, "type": TYPE_TITLE, "group": "g"}]
    extractor = _make_extractor(monkeypatch, stored, {TYPE_TITLE: 5}, types)

    extractor.extract_references()

    # The bug regenerated 121/000001..121/000005 from scratch; the fix must not.
    for n in range(1, 6):
        assert f"{TYPE_CODE}/00000{n}" not in extractor.all_references

    # Only the safety-gap tail (SAFETY_EXTRACTION_GAP - 1 refs) should be queued,
    # never the whole corpus (origin_total + gap - 1 == 7 under the bug).
    assert len(extractor.all_references) == extractor.SAFETY_EXTRACTION_GAP - 1
