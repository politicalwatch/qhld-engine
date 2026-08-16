"""Unit tests for the corpus-wide subtitle backfill — no network, no model, no Mongo.

Like its ``BackfillDurations`` sibling this earns tests because it downloads
concurrently and because its failure path decides whether one unreachable video costs
an intervention or the twenty hours the rest of the run has already spent. The two
properties worth pinning beyond that are the prefetch window (a pool handed the whole
corpus would hold every clip in memory at once) and the incremental filter (what makes
an interrupted run cheap to resume).
"""

import pytest

from qhld_engine.application.speeches import align_corpus as mod

pytestmark = pytest.mark.unit


class FakeBlock:
    def __init__(self, lang="es", original=True, text="hola que tal"):
        self.lang = lang
        self.original = original
        self.text = text


class FakeSpeech:
    def __init__(self, id, date=20260101, blocks=None, video_link="http://v/1.mp4",
                 duration=300.0):
        self.id = id
        self.date = date
        self.speech = blocks if blocks is not None else [FakeBlock()]
        self.video_link = video_link
        self.duration = duration


class FakeAlign:
    """Stands in for ``AlignSpeech``: records what it was asked to time."""

    def __init__(self, fail=()):
        self.calls = []
        self._fail = set(fail)

    def execute(self, speech_id, force=False, persist=True, samples=None):
        if speech_id in self._fail:
            raise mod.NotAlignable(f"{speech_id} has no alignable words")
        self.calls.append((speech_id, force, persist, samples))
        return []


def _stub(monkeypatch, speeches, aligned=(), by_reference=None):
    """Put the corpus behind the repositories, and say which blocks already have cues.

    ``aligned`` is the set of speech ids whose every block is already timed, which is
    the only thing the incremental filter asks about.
    """
    monkeypatch.setattr(mod.Speeches, "all", staticmethod(lambda: list(speeches)),
                        raising=False)
    monkeypatch.setattr(mod.Speeches, "by_references",
                        staticmethod(lambda refs: list(by_reference or [])),
                        raising=False)
    monkeypatch.setattr(
        mod.SpeechAlignments, "summaries",
        staticmethod(lambda speech_id, blocks:
                     list(blocks) if speech_id in set(aligned) else []),
        raising=False)


def _service(speeches, decode=None, align=None, workers=2):
    return mod.AlignCorpus(align=align or FakeAlign(),
                           decode=decode or (lambda link: f"pcm:{link}"),
                           workers=workers)


# ---- what gets aligned ----

def test_aligns_every_speech_with_a_pending_block(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b")])
    align = FakeAlign()

    _service([], align=align).execute()

    assert [call[0] for call in align.calls] == ["a", "b"]


def test_incremental_skips_speeches_already_fully_aligned(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b")], aligned=["a"])
    align, downloaded = FakeAlign(), []

    mod.AlignCorpus(align=align,
                    decode=lambda link: downloaded.append(link) or "pcm",
                    workers=2).execute()

    assert [call[0] for call in align.calls] == ["b"]
    # The point of skipping: an already-aligned speech never costs its video.
    assert len(downloaded) == 1


def test_all_realigns_everything(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b")], aligned=["a", "b"])
    align = FakeAlign()

    _service([], align=align).execute(incremental=False)

    assert [call[0] for call in align.calls] == ["a", "b"]
    assert all(call[1] is True for call in align.calls), "should force a redo"


def test_a_reference_still_only_times_missing_blocks(monkeypatch):
    # Deliberately unlike BackfillDurations, where narrowing implies a redo: re-probing
    # a duration costs a header request, re-aligning costs the whole video.
    targeted = [FakeSpeech("x")]
    _stub(monkeypatch, [FakeSpeech("a")], aligned=["x"], by_reference=targeted)
    align = FakeAlign()

    _service([], align=align).execute(["210/000151"])

    assert align.calls == []


def test_limit_stops_after_that_many_speeches(monkeypatch):
    _stub(monkeypatch, [FakeSpeech(str(i), date=20260101 + i) for i in range(5)])
    align = FakeAlign()

    _service([], align=align).execute(limit=2)

    assert len(align.calls) == 2


# ---- ordering, so an interrupted run has done the useful half ----

def test_newest_speeches_are_aligned_first(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("old", date=20240101),
                        FakeSpeech("new", date=20260601),
                        FakeSpeech("mid", date=20250301)])
    align = FakeAlign()

    _service([], align=align).execute()

    assert [call[0] for call in align.calls] == ["new", "mid", "old"]


# ---- failure must never sink the run ----

def test_a_speech_whose_video_fails_is_skipped_not_fatal(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b", video_link="http://v/2.mp4"),
                        FakeSpeech("c")])
    align = FakeAlign()

    def _decode(link):
        if link.endswith("2.mp4"):
            raise mod.AudioDecodeError("nope")
        return "pcm"

    mod.AlignCorpus(align=align, decode=_decode, workers=2).execute()

    assert [call[0] for call in align.calls] == ["a", "c"]


def test_a_speech_that_cannot_be_aligned_is_skipped_not_fatal(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b"), FakeSpeech("c")])
    align = FakeAlign(fail=["b"])

    _service([], align=align).execute()

    assert [call[0] for call in align.calls] == ["a", "c"]


def test_failed_speeches_are_listed_not_merely_counted(monkeypatch, caplog):
    # On a run this long, a warning logged fifteen hours ago is not auditable.
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("bad")])
    align = FakeAlign(fail=["bad"])

    with caplog.at_level("WARNING"):
        _service([], align=align).execute()

    assert "bad" in caplog.text


def test_a_speech_without_a_video_is_never_downloaded(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a", video_link=None), FakeSpeech("b")])
    downloaded = []

    mod.AlignCorpus(align=FakeAlign(),
                    decode=lambda link: downloaded.append(link) or "pcm",
                    workers=2).execute()

    assert downloaded == ["http://v/1.mp4"]


def test_a_speech_whose_blocks_hold_no_words_is_not_pending_forever(monkeypatch):
    # It can never gain a track, so counting it as missing would re-download its video
    # on every run for the rest of time.
    silent = FakeSpeech("quiet", blocks=[FakeBlock(text="(Aplausos.)")])
    _stub(monkeypatch, [silent])
    align = FakeAlign()

    _service([], align=align).execute()

    assert align.calls == []


# ---- the prefetch window is a memory bound ----

def test_prefetch_never_holds_more_clips_than_the_window(monkeypatch):
    # A clip is tens of megabytes and the corpus is 3,974 of them, so "submit them all"
    # would try to hold the whole corpus in RAM.
    _stub(monkeypatch, [FakeSpeech(str(i), date=20260101 + i) for i in range(40)])
    in_flight, high_water = [], []

    def _decode(link):
        in_flight.append(link)
        high_water.append(len(in_flight))
        return "pcm"

    class Consuming(FakeAlign):
        def execute(self, speech_id, force=False, persist=True, samples=None):
            if in_flight:
                in_flight.pop(0)
            return super().execute(speech_id, force, persist, samples)

    align = Consuming()
    mod.AlignCorpus(align=align, decode=_decode, workers=3).execute()

    assert len(align.calls) == 40
    # workers + 1 in the window, plus the one being aligned. Bounded is the property;
    # the exact number matters only because it must not grow with the corpus.
    assert max(high_water) <= 5, f"held {max(high_water)} clips, bound is 3 + 2"


# ---- dry runs and persistence ----

def test_persist_false_is_passed_through(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a")])
    align = FakeAlign()

    _service([], align=align).execute(persist=False)

    assert align.calls[0][2] is False


def test_prefetched_audio_reaches_the_aligner(monkeypatch):
    # The whole reason the driver downloads at all: AlignSpeech must not fetch again.
    _stub(monkeypatch, [FakeSpeech("a")])
    align = FakeAlign()

    mod.AlignCorpus(align=align, decode=lambda link: "pcm:a", workers=2).execute()

    assert align.calls[0][3] == "pcm:a"


def test_pending_reports_the_work_without_doing_any(monkeypatch):
    _stub(monkeypatch, [FakeSpeech("a"), FakeSpeech("b")], aligned=["a"])
    downloaded = []

    pending = mod.AlignCorpus(
        align=FakeAlign(), decode=lambda link: downloaded.append(link),
        workers=2).pending()

    assert [s.id for s in pending] == ["b"]
    assert downloaded == []
