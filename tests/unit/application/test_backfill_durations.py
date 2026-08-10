"""Unit tests for the clip-length backfill — no network, no Mongo.

Its siblings (mentions, entities) are untested plain loops; this one earns tests
because it probes concurrently and because its failure path decides whether an
unreachable video costs one field or the whole run.
"""

import pytest

from qhld_engine.application.speeches import backfill_durations as mod

pytestmark = pytest.mark.unit


class FakeSpeech:
    def __init__(self, id, video_link="http://v/1.mp4", duration=None):
        self.id = id
        self.video_link = video_link
        self.duration = duration


def _stub_repository(monkeypatch, speeches, saved, by_reference=None):
    monkeypatch.setattr(mod.Speeches, "all", staticmethod(lambda: list(speeches)))
    monkeypatch.setattr(mod.Speeches, "by_references",
                        staticmethod(lambda refs: list(by_reference or [])))
    monkeypatch.setattr(mod.Speeches, "save",
                        staticmethod(lambda speech: saved.append(speech)))


def test_measures_and_persists_every_unmeasured_speech(monkeypatch):
    speeches = [FakeSpeech("a"), FakeSpeech("b")]
    saved = []
    _stub_repository(monkeypatch, speeches, saved)

    mod.BackfillDurations(probe=lambda link: 352.0, workers=2).execute()

    assert [s.duration for s in speeches] == [352.0, 352.0]
    assert [s.id for s in saved] == ["a", "b"]


def test_incremental_skips_speeches_that_already_have_one(monkeypatch):
    speeches = [FakeSpeech("a", duration=100.0), FakeSpeech("b")]
    saved, probed = [], []
    _stub_repository(monkeypatch, speeches, saved)

    mod.BackfillDurations(
        probe=lambda link: probed.append(link) or 352.0, workers=2).execute()

    assert [s.id for s in saved] == ["b"]
    assert len(probed) == 1


def test_all_reprobes_everything(monkeypatch):
    speeches = [FakeSpeech("a", duration=100.0), FakeSpeech("b")]
    saved = []
    _stub_repository(monkeypatch, speeches, saved)

    mod.BackfillDurations(probe=lambda link: 352.0, workers=2).execute(
        incremental=False)

    assert [s.id for s in saved] == ["a", "b"]
    assert speeches[0].duration == 352.0


def test_an_unreadable_video_is_skipped_not_fatal(monkeypatch):
    # One bad clip must not abandon the other 3,900.
    speeches = [FakeSpeech("a"), FakeSpeech("b", video_link="http://v/2.mp4"),
                FakeSpeech("c")]
    saved = []
    _stub_repository(monkeypatch, speeches, saved)

    def _probe(link):
        if link.endswith("2.mp4"):
            raise mod.DurationUnavailable("nope")
        return 10.0

    mod.BackfillDurations(probe=_probe, workers=2).execute()

    assert [s.id for s in saved] == ["a", "c"]
    assert speeches[1].duration is None


def test_a_speech_without_a_video_is_never_probed(monkeypatch):
    speeches = [FakeSpeech("a", video_link=None), FakeSpeech("b")]
    saved, probed = [], []
    _stub_repository(monkeypatch, speeches, saved)

    mod.BackfillDurations(
        probe=lambda link: probed.append(link) or 10.0, workers=2).execute()

    assert probed == ["http://v/1.mp4"]
    assert [s.id for s in saved] == ["b"]


def test_a_reference_targets_only_its_speeches(monkeypatch):
    targeted = [FakeSpeech("x", duration=99.0)]
    saved = []
    _stub_repository(monkeypatch, [FakeSpeech("a")], saved, by_reference=targeted)

    mod.BackfillDurations(probe=lambda link: 352.0, workers=2).execute(
        ["210/000151"])

    # a reference is always re-probed, stored duration or not
    assert [s.id for s in saved] == ["x"]
    assert targeted[0].duration == 352.0
