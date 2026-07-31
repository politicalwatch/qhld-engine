"""Unit tests for the abort paths of the Spain extractors — no DB, no network.

These steps used to log a problem and return as if they had succeeded, which made the
scheduler mark the task successful, run the rest of the pipeline and report a clean
day. They now raise ``ExtractionError``, so the process exits non-zero.

The line they must hold: an individual item failing is normal and stays tolerated;
producing *nothing at all* is a failure.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from qhld_engine.extractors.errors import ExtractionError
from qhld_engine.extractors.spain.congress_api import (
    CongressError,
    CongressForbiddenError,
)
from qhld_engine.extractors.spain import groups as groups_module
from qhld_engine.extractors.spain import members as members_module
from qhld_engine.extractors.spain.groups import GroupsExtractor
from qhld_engine.extractors.spain.members import MembersExtractor

pytestmark = pytest.mark.unit


# --- MembersExtractor ---------------------------------------------------------

@pytest.fixture
def extractor(monkeypatch):
    """A MembersExtractor with no Mongo read in __init__ and no real API."""
    monkeypatch.setattr(members_module.ParliamentaryGroups, "get_all", lambda: [])
    return MembersExtractor()


@pytest.mark.parametrize("error", [CongressForbiddenError, CongressError])
def test_an_unreadable_deputies_list_fails_the_run(extractor, error):
    def refuse():
        raise error()

    extractor.api = SimpleNamespace(get_deputies=refuse)

    with pytest.raises(ExtractionError):
        extractor.extract()


def _resolved(ok):
    """An already-completed future, as ``async_get`` returns: it does not raise on a
    4xx, the caller inspects ``ok`` — which is how a throttled roster stayed silent."""
    future = Future()
    future.set_result(SimpleNamespace(
        ok=ok, status_code=200 if ok else 403,
        url="https://www.congreso.es/deputy"))
    return future


def _with_deputy_responses(extractor, monkeypatch, responses):
    extracted = []
    monkeypatch.setattr(
        members_module, "DeputyExtractor",
        lambda response, groups: SimpleNamespace(
            extract=lambda: extracted.append(response)))
    futures = iter([_resolved(ok) for ok in responses])
    extractor.api = SimpleNamespace(get_deputy=lambda reference: next(futures))
    extractor.references = [f"ref{i}" for i in range(len(responses))]
    return extracted


def test_a_roster_that_returns_nothing_fails_the_run(extractor, monkeypatch):
    # Every detail request refused: zero deputies written, previously reported as a
    # successful extraction.
    _with_deputy_responses(extractor, monkeypatch, [False, False, False])

    with pytest.raises(ExtractionError):
        extractor.extract_deputies()


def test_one_failed_deputy_does_not_fail_the_run(extractor, monkeypatch):
    extracted = _with_deputy_responses(extractor, monkeypatch, [True, False, True])

    extractor.extract_deputies()  # must not raise

    assert len(extracted) == 2


def test_no_references_is_not_a_failure(extractor, monkeypatch):
    # Nothing to do is not the same as failing to do it.
    _with_deputy_responses(extractor, monkeypatch, [])

    extractor.extract_deputies()


# --- GroupsExtractor ----------------------------------------------------------

def _with_groups(monkeypatch, groups, composition):
    monkeypatch.setattr(groups_module.ParliamentaryGroups, "get_all", lambda: groups)
    monkeypatch.setattr(
        groups_module.ParliamentaryGroups, "get_composition", composition)
    saved = []
    monkeypatch.setattr(groups_module.ParliamentaryGroups, "save", saved.append)
    return saved


def test_composition_failing_for_every_group_fails_the_run(monkeypatch):
    def refuse(shortname):
        raise RuntimeError("mongo is gone")

    _with_groups(monkeypatch, [{"_id": "g1", "shortname": "GS"},
                               {"_id": "g2", "shortname": "GP"}], refuse)

    with pytest.raises(ExtractionError):
        GroupsExtractor().calculate_composition()


def test_composition_failing_for_one_group_does_not_fail_the_run(monkeypatch):
    def refuse_gp(shortname):
        if shortname == "GP":
            raise RuntimeError("bad group")
        return 121

    saved = _with_groups(monkeypatch, [{"_id": "g1", "shortname": "GS"},
                                       {"_id": "g2", "shortname": "GP"}], refuse_gp)

    GroupsExtractor().calculate_composition()  # must not raise

    assert len(saved) == 1


def test_no_groups_at_all_is_not_a_failure(monkeypatch):
    _with_groups(monkeypatch, [], lambda shortname: 0)

    GroupsExtractor().calculate_composition()


def test_a_missing_groups_file_fails_the_run():
    with pytest.raises(ExtractionError):
        GroupsExtractor().load("/nonexistent/groups.json")
