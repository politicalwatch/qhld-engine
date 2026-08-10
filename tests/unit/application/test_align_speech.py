"""Unit tests for subtitle alignment — no Mongo, no model, no ffmpeg.

The aligner and the audio decoder are injected fakes. The fake aligner times words
at a fixed rate, so the expected cue boundaries are arithmetic and every assertion
is about the composition this service is responsible for: timing every language block
of a speech against one pass over the audio, keeping stage directions out of the
alignment while keeping the offsets in the full stored string, and grouping timed words
into cues.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from qhld_ai.domain.ports.aligner import Alignment, ModelArtifact, WordTiming
from qhld_ai.infrastructure.config.settings import Settings
from tipi_data.models.speech import Speech, SpeechText

from qhld_engine.application.speeches import align_speech as mod
from qhld_engine.application.speeches.align_speech import AlignSpeech, NotAlignable

pytestmark = pytest.mark.unit

WORDS_PER_SECOND = 2.0


def _settings(**overrides):
    return Settings(_env_file=None, aligner_min_score=85.0, **overrides)


class _FakeAligner:
    """Times word ``i`` to the half-second slot ``i``, and records every request it was
    asked to place — so the tests can assert what reached it, in what language, and that
    the audio was only read once."""

    def __init__(self, score=99.0, scores=None):
        # ``scores`` gives a per-language score, for the case where a translated block
        # is expected to be judged differently from the words actually spoken.
        self.score = score
        self.scores = scores or {}
        self.requests = []
        self.passes = 0

    def align_all(self, samples, sample_rate, requests):
        self.passes += 1
        self.requests.extend(requests)
        return [self._one(request) for request in requests]

    def _one(self, request):
        timings = [
            WordTiming(start=index / WORDS_PER_SECOND,
                       end=(index + 0.9) / WORDS_PER_SECOND)
            for index in range(len(request.words))
        ]
        return Alignment(words=timings,
                         score=self.scores.get(request.lang, self.score),
                         model=ModelArtifact(id="fake", revision="r1",
                                             sha256="0" * 64))

    @property
    def words(self):
        """The words of the first request — the single-block case most tests use."""
        return list(self.requests[0].words) if self.requests else None

    def words_for(self, lang):
        return next((list(r.words) for r in self.requests if r.lang == lang), None)


def _decoder(seconds=10.0):
    return lambda source: np.zeros(int(16000 * seconds), dtype=np.float32)


def _speech(blocks=None, video_link="http://v/1.mp4"):
    return Speech(
        _id="sp-1",
        video_id="726567",
        video_link=video_link,
        speech=blocks if blocks is not None else [
            SpeechText(lang="es", text="Muchas gracias. Segunda frase aquí.",
                       original=True)],
    )


def _stored(lang):
    """Stands in for an alignment already in Mongo. Only its language is read — the
    service returns it untouched rather than looking inside it."""
    return SimpleNamespace(lang=lang)


def _bilingual():
    return [
        SpeechText(lang="gl", text="Grazas, señora presidenta.", original=True),
        SpeechText(lang="es", text="Gracias, señora presidenta.", original=False),
    ]


@pytest.fixture(autouse=True)
def _repositories(monkeypatch):
    """Keep the service off Mongo; each test overrides what it needs."""
    saved = []
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(), raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id, lang: False,
                        raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "save", lambda record: saved.append(record),
                        raising=False)
    return saved


def _service(aligner=None, seconds=10.0, **settings):
    return AlignSpeech(settings=_settings(**settings),
                       aligner=aligner or _FakeAligner(), decode=_decoder(seconds))


# ---- the happy path ---------------------------------------------------------

def test_stores_cues_with_offsets_into_the_stored_text(_repositories):
    text = "Muchas gracias. Segunda frase aquí."
    records = _service().execute("sp-1")

    assert _repositories == records
    assert len(records) == 1
    record = records[0]
    assert record.id == "sp-1:es"
    assert record.speech_id == "sp-1"
    assert record.lang == "es"
    assert record.block_index == 0
    assert record.original is True
    assert record.cues
    for cue in record.cues:
        assert text[cue.char_start:cue.char_end]
        assert cue.end_ms >= cue.start_ms


def test_records_the_text_fingerprint_so_drift_is_detectable(_repositories):
    record = _service().execute("sp-1")[0]

    assert record.text_length == len("Muchas gracias. Segunda frase aquí.")
    assert len(record.text_sha256) == 64


def test_records_which_artifact_produced_the_timings(_repositories):
    record = _service().execute("sp-1")[0]

    assert record.model_id == "fake"
    assert record.model_revision == "r1"
    assert record.model_sha256 == "0" * 64


def test_audio_seconds_come_from_the_decoded_samples(_repositories):
    record = _service(seconds=12.5).execute("sp-1")[0]

    assert record.audio_seconds == pytest.approx(12.5)


# ---- which blocks get aligned ----------------------------------------------

def test_every_language_block_gets_its_own_track(monkeypatch, _repositories):
    """A co-official speech carries the original and its full Spanish interpretation,
    and both are subtitled: most readers only read Spanish, so timing the original
    alone left them a track they could not read."""
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(_bilingual()),
                        raising=False)
    aligner = _FakeAligner()

    records = _service(aligner).execute("sp-1")

    assert [(r.lang, r.block_index, r.original) for r in records] == [
        ("gl", 0, True), ("es", 1, False)]
    assert [r.id for r in records] == ["sp-1:gl", "sp-1:es"]
    assert aligner.words_for("gl") == ["Grazas,", "señora", "presidenta."]
    assert aligner.words_for("es") == ["Gracias,", "señora", "presidenta."]
    assert _repositories == records


def test_both_tracks_come_from_one_pass_over_the_audio(monkeypatch, _repositories):
    """What makes the second track affordable, and what makes the two agree: the clip
    is read once and both transcripts are placed against that same reading."""
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(_bilingual()),
                        raising=False)
    aligner = _FakeAligner()

    _service(aligner).execute("sp-1")

    assert aligner.passes == 1
    assert len(aligner.requests) == 2


def test_the_aligner_is_told_which_language_each_block_is_in(monkeypatch,
                                                             _repositories):
    # A figure has to be read aloud before it can be matched to audio, and which words
    # that means depends on the language. Sending each block's own language rather than
    # letting the aligner assume Spanish is what keeps a Galician intervention's numbers
    # out of the Spanish reading.
    blocks = [
        SpeechText(lang="gl", text="Grazas. Foron 47 casos no ano 2017.",
                   original=True),
        SpeechText(lang="es", text="Gracias. Fueron 47 casos en el año 2017.",
                   original=False),
    ]
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(blocks),
                        raising=False)
    aligner = _FakeAligner()

    _service(aligner).execute("sp-1")

    assert [r.lang for r in aligner.requests] == ["gl", "es"]


def test_a_speech_with_only_a_translation_block_is_still_subtitled(monkeypatch,
                                                                  _repositories):
    """Where the speaker code-switched, the Diario can mark only a fragment as
    as-delivered — and it used to be the only thing aligned, which scored ~1 out of 100
    because a greeting cannot account for a six-minute clip. Every block is timed now,
    so the block that does cover the audio gets its track."""
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(lang="es", text="Gracias, presidenta.",
                                       original=False)]),
        raising=False)

    records = _service().execute("sp-1")

    assert [(r.lang, r.original) for r in records] == [("es", False)]


def test_a_block_with_nothing_alignable_does_not_cost_its_sibling_a_track(
        monkeypatch, _repositories):
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(lang="gl", text="   ", original=True),
                            SpeechText(lang="es", text="Gracias, presidenta.",
                                       original=False)]),
        raising=False)

    records = _service().execute("sp-1")

    assert [r.lang for r in records] == ["es"]


def test_a_speech_with_no_video_is_not_alignable(monkeypatch):
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(video_link=None),
                        raising=False)

    with pytest.raises(NotAlignable, match="has not published it"):
        _service().execute("sp-1")


def test_a_speech_with_no_words_is_not_alignable(monkeypatch):
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(lang="es", text="   ", original=True)]),
        raising=False)

    with pytest.raises(NotAlignable, match="no alignable words"):
        _service().execute("sp-1")


# ---- stage directions -------------------------------------------------------

def test_stage_directions_are_not_sent_to_the_aligner(monkeypatch, _repositories):
    # Applause is audible but is not this speaker's words, so asking the aligner to
    # find it would be asking it to find text nobody in the clip said.
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(
            lang="es", text="Muchas gracias. (Aplausos). Continúo ahora.",
            original=True)]),
        raising=False)
    aligner = _FakeAligner()

    _service(aligner).execute("sp-1")

    assert aligner.words == ["Muchas", "gracias.", "Continúo", "ahora."]


def test_cue_offsets_still_span_the_stage_directions(monkeypatch, _repositories):
    text = "Muchas gracias. (Aplausos). Continúo ahora."
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(lang="es", text=text, original=True)]),
        raising=False)

    record = _service().execute("sp-1")[0]

    # The offsets index the string readers see and search highlights are located in,
    # annotations included.
    assert all(text[c.char_start:c.char_end] for c in record.cues)
    assert max(c.char_end for c in record.cues) <= len(text)


# ---- the trust gate ---------------------------------------------------------

def test_a_confident_alignment_is_marked_ok(_repositories):
    record = _service(_FakeAligner(score=99.0)).execute("sp-1")[0]

    assert record.score == 99.0
    assert record.verdict == "ok"


def test_a_low_score_is_stored_and_flagged_rather_than_withheld(_repositories):
    # The check is made with the model that produced the timings, so it is
    # pessimistic exactly where that model is weak; discarding would lose alignments
    # measured to be correct.
    record = _service(_FakeAligner(score=42.0)).execute("sp-1")[0]

    assert record.verdict == "low"
    assert record.cues
    assert _repositories == [record]


def test_each_track_carries_its_own_verdict(monkeypatch, _repositories):
    """The score is not comparable between the two: it asks whether the audio says
    these words, and a Spanish rendering of a Galician speech does not, however exactly
    its cues land. So a translated track reading low must not drag the original down,
    nor be withheld for it."""
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(_bilingual()),
                        raising=False)
    aligner = _FakeAligner(scores={"gl": 96.0, "es": 77.0})

    records = _service(aligner).execute("sp-1")

    assert [(r.lang, r.score, r.verdict) for r in records] == [
        ("gl", 96.0, "ok"), ("es", 77.0, "low")]
    assert all(r.cues for r in records)


# ---- re-running -------------------------------------------------------------

def test_an_already_aligned_speech_is_not_realigned(monkeypatch, _repositories):
    existing = _stored("es")
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id, lang: True,
                        raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "get", lambda id, lang: existing,
                        raising=False)
    aligner = _FakeAligner()

    assert _service(aligner).execute("sp-1") == [existing]
    assert aligner.requests == []     # the video was never downloaded
    assert _repositories == []


def test_only_the_missing_language_is_aligned(monkeypatch, _repositories):
    """A speech that gained a second block, or one aligned before the Spanish track
    existed, must not pay to redo the language it already has."""
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(_bilingual()),
                        raising=False)
    existing = _stored("gl")
    monkeypatch.setattr(mod.SpeechAlignments, "exists",
                        lambda id, lang: lang == "gl", raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "get", lambda id, lang: existing,
                        raising=False)
    aligner = _FakeAligner()

    records = _service(aligner).execute("sp-1")

    assert [r.lang for r in aligner.requests] == ["es"]
    assert records[0] is existing
    assert records[1].lang == "es"
    assert [r.lang for r in _repositories] == ["es"]


def test_force_realigns_an_already_aligned_speech(monkeypatch, _repositories):
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id, lang: True,
                        raising=False)
    aligner = _FakeAligner()

    records = _service(aligner).execute("sp-1", force=True)

    assert aligner.requests
    assert _repositories == records


def test_dry_run_aligns_without_storing(_repositories):
    record = _service().execute("sp-1", persist=False)[0]

    assert record.cues
    assert _repositories == []


# ---- cue segmentation -------------------------------------------------------

def test_cues_respect_the_configured_word_budget(monkeypatch, _repositories):
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(
            lang="es", text=" ".join(["palabra"] * 12) + ".", original=True)]),
        raising=False)

    records = _service(subtitle_max_words=4,
                       subtitle_max_chars=10 ** 6).execute("sp-1")

    record = records[0]
    assert len(record.cues) == 3
    # Word i is timed to slot i, so the second cue starts two seconds in.
    assert record.cues[0].start_ms == 0
    assert record.cues[1].start_ms == int(4 / WORDS_PER_SECOND * 1000)


def test_cue_times_come_from_the_first_and_last_word_it_covers(_repositories):
    record = _service(subtitle_max_words=2,
                      subtitle_max_chars=10 ** 6).execute("sp-1")[0]

    first = record.cues[0]
    assert first.start_ms == 0
    assert first.end_ms == int(1.9 / WORDS_PER_SECOND * 1000)
