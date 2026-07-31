"""Unit tests for the post-extraction freshness bookkeeping — no Mongo, no Redis.

Two things must hold for every dataset the engine rewrites: the refresh is stamped
(the backend serves it on ``GET /``) and exactly the cache keys that went stale are
deleted — never a whole-database flush, since Redis DB 8 holds other caches.

The third requirement is that neither half can break an extraction run, so the
failure paths are pinned too.
"""

import pytest

from qhld_engine.application import freshness
from qhld_engine.application.freshness import (
    DEPUTIES,
    INITIATIVES,
    PARLIAMENTARY_GROUPS,
    mark_refreshed,
)
from qhld_engine.extractors import extractor as extractor_module
from qhld_engine.extractors.extractor import ExtractorTask

pytestmark = pytest.mark.unit


@pytest.fixture
def spy(monkeypatch):
    """Record what would have been stamped and deleted."""
    calls = {"touched": [], "deleted": []}
    monkeypatch.setattr(freshness.DatasetUpdates, "touch", calls["touched"].append)
    monkeypatch.setattr(
        freshness.cache, "delete",
        lambda *keys: calls["deleted"].extend(keys) or len(keys))
    return calls


# --- what each dataset invalidates -------------------------------------------

def test_deputies_stamps_and_drops_both_deputy_keys(spy):
    mark_refreshed(DEPUTIES)

    assert spy["touched"] == ["deputies"]
    assert sorted(spy["deleted"]) == ["deputies", "deputies-compact"]


def test_groups_stamps_and_drops_only_the_groups_key(spy):
    mark_refreshed(PARLIAMENTARY_GROUPS)

    assert spy["touched"] == ["parliamentary-groups"]
    assert spy["deleted"] == ["parliamentary-groups"]


def test_initiatives_are_stamped_but_flush_nothing(spy):
    # The backend serves initiatives uncached; Redis must not be touched at all.
    mark_refreshed(INITIATIVES)

    assert spy["touched"] == ["initiatives"]
    assert spy["deleted"] == []


def test_cache_keys_follow_the_settings(monkeypatch, spy):
    # The key names are shared with the backend through env; honour an override.
    monkeypatch.setenv("CACHE_GROUPS", "grupos")

    mark_refreshed(PARLIAMENTARY_GROUPS)

    assert spy["deleted"] == ["grupos"]


def test_unknown_dataset_is_a_programming_error(spy):
    with pytest.raises(KeyError):
        mark_refreshed("speeches")

    # It fails before doing anything, so no half-done bookkeeping.
    assert spy["touched"] == []
    assert spy["deleted"] == []


# --- neither half may fail an extraction run ---------------------------------

def _raise(*args, **kwargs):
    raise ConnectionError("nope")


def test_unreachable_redis_does_not_stop_the_run(monkeypatch, spy):
    monkeypatch.setattr(freshness.cache, "delete", _raise)

    mark_refreshed(DEPUTIES)  # must not raise

    # ...and the refresh is still recorded, so freshness survives a Redis outage.
    assert spy["touched"] == ["deputies"]


def test_unreachable_mongo_does_not_stop_the_run(monkeypatch, spy):
    monkeypatch.setattr(freshness.DatasetUpdates, "touch", _raise)

    mark_refreshed(DEPUTIES)  # must not raise

    # ...and the stale cache is still dropped, which is the more visible half.
    assert sorted(spy["deleted"]) == ["deputies", "deputies-compact"]


# --- the hooks on the extraction paths ---------------------------------------

@pytest.fixture
def refreshed(monkeypatch):
    marked = []
    monkeypatch.setattr(extractor_module, "mark_refreshed", marked.append)
    return marked


class _Recorder:
    """Stands in for an extractor; records the methods the task calls."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append(name)


class _Task(ExtractorTask):
    """The real task with the real methods, minus the ``__init__`` that loads a
    country module and connects to Mongo."""

    def __init__(self, **extractors):
        self.__dict__.update(extractors)


def test_members_marks_deputies_only(refreshed):
    task = _Task(members_extractor=_Recorder())

    task.members()

    assert task.members_extractor.calls == ["extract"]
    # Not the groups: their stored composition is only recomputed by
    # calculate_composition_groups(), so the cached groups response is still valid.
    assert refreshed == [DEPUTIES]


def test_calculate_composition_groups_marks_groups(refreshed):
    task = _Task(groups_extractor=_Recorder())

    task.calculate_composition_groups()

    assert task.groups_extractor.calls == ["calculate_composition"]
    assert refreshed == [PARLIAMENTARY_GROUPS]


def test_load_groups_marks_groups(refreshed):
    # GroupsExtractor.load() recalculates the composition itself, so this path
    # changes the groups too.
    task = _Task(groups_extractor=_Recorder())

    task.load_groups("groups.json")

    assert task.groups_extractor.calls == ["load"]
    assert refreshed == [PARLIAMENTARY_GROUPS]


def test_initiatives_marks_initiatives(refreshed):
    task = _Task(initiatives_extractor=_Recorder())

    task.initiatives()

    assert refreshed == [INITIATIVES]


def test_run_marks_every_dataset_once(refreshed):
    task = _Task(members_extractor=_Recorder(), groups_extractor=_Recorder(),
                 initiatives_extractor=_Recorder())

    task.run()

    # run() goes through the same methods, so no dataset is stamped twice.
    assert refreshed == [DEPUTIES, PARLIAMENTARY_GROUPS, INITIATIVES]
