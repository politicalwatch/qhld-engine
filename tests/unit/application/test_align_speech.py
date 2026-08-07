"""Unit tests for subtitle alignment — no Mongo, no model, no ffmpeg.

The aligner and the audio decoder are injected fakes. The fake aligner times words
at a fixed rate, so the expected cue boundaries are arithmetic and every assertion
is about the composition this service is responsible for: choosing the block the
audio contains, keeping stage directions out of the alignment while keeping the
offsets in the full stored string, and grouping timed words into cues.
"""

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
    """Times word ``i`` to the half-second slot ``i``, and records what it was asked
    to align so the tests can assert the annotations never reached it."""

    def __init__(self, score=99.0):
        self.score = score
        self.words = None
        self.lang = None

    def align(self, samples, sample_rate, words, lang):
        self.words = list(words)
        self.lang = lang
        timings = [
            WordTiming(start=index / WORDS_PER_SECOND,
                       end=(index + 0.9) / WORDS_PER_SECOND)
            for index in range(len(words))
        ]
        return Alignment(words=timings, score=self.score,
                         model=ModelArtifact(id="fake", revision="r1",
                                             sha256="0" * 64))


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


@pytest.fixture(autouse=True)
def _repositories(monkeypatch):
    """Keep the service off Mongo; each test overrides what it needs."""
    saved = []
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(), raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id: False,
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
    record = _service().execute("sp-1")

    assert _repositories == [record]
    assert record.id == "sp-1"
    assert record.lang == "es"
    assert record.block_index == 0
    assert record.cues
    for cue in record.cues:
        assert text[cue.char_start:cue.char_end]
        assert cue.end_ms >= cue.start_ms


def test_records_the_text_fingerprint_so_drift_is_detectable(_repositories):
    record = _service().execute("sp-1")

    assert record.text_length == len("Muchas gracias. Segunda frase aquí.")
    assert len(record.text_sha256) == 64


def test_records_which_artifact_produced_the_timings(_repositories):
    record = _service().execute("sp-1")

    assert record.model_id == "fake"
    assert record.model_revision == "r1"
    assert record.model_sha256 == "0" * 64


def test_audio_seconds_come_from_the_decoded_samples(_repositories):
    record = _service(seconds=12.5).execute("sp-1")

    assert record.audio_seconds == pytest.approx(12.5)


# ---- which block gets aligned ----------------------------------------------

def test_aligns_the_as_delivered_block_not_the_translation(monkeypatch, _repositories):
    # A co-official speech carries the original and its full Spanish interpretation;
    # only the original was spoken, and aligning both would present roughly twice as
    # much text as there is audio.
    blocks = [
        SpeechText(lang="gl", text="Grazas, señora presidenta.", original=True),
        SpeechText(lang="es", text="Gracias, señora presidenta.", original=False),
    ]
    monkeypatch.setattr(mod.Speeches, "get", lambda id: _speech(blocks),
                        raising=False)
    aligner = _FakeAligner()

    record = _service(aligner).execute("sp-1")

    assert record.lang == "gl"
    assert record.block_index == 0
    assert aligner.words == ["Grazas,", "señora", "presidenta."]


def test_the_aligner_is_told_which_language_it_is_listening_to(monkeypatch,
                                                              _repositories):
    # A figure has to be read aloud before it can be matched to audio, and which words
    # that means depends on the language. Sending the speech's own language rather than
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

    assert aligner.lang == "gl"


def test_a_speech_with_no_as_delivered_block_is_not_alignable(monkeypatch):
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(lang="es", text="x", original=False)]),
        raising=False)

    with pytest.raises(NotAlignable, match="no as-delivered text"):
        _service().execute("sp-1")


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

    record = _service().execute("sp-1")

    # The offsets index the string readers see and search highlights are located in,
    # annotations included.
    assert all(text[c.char_start:c.char_end] for c in record.cues)
    assert max(c.char_end for c in record.cues) <= len(text)


# ---- the trust gate ---------------------------------------------------------

def test_a_confident_alignment_is_marked_ok(_repositories):
    record = _service(_FakeAligner(score=99.0)).execute("sp-1")

    assert record.score == 99.0
    assert record.verdict == "ok"


def test_a_low_score_is_stored_and_flagged_rather_than_withheld(_repositories):
    # The check is made with the model that produced the timings, so it is
    # pessimistic exactly where that model is weak; discarding would lose alignments
    # measured to be correct.
    record = _service(_FakeAligner(score=42.0)).execute("sp-1")

    assert record.verdict == "low"
    assert record.cues
    assert _repositories == [record]


# ---- re-running -------------------------------------------------------------

def test_an_already_aligned_speech_is_not_realigned(monkeypatch, _repositories):
    existing = object()
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id: True,
                        raising=False)
    monkeypatch.setattr(mod.SpeechAlignments, "get", lambda id: existing,
                        raising=False)
    aligner = _FakeAligner()

    assert _service(aligner).execute("sp-1") is existing
    assert aligner.words is None      # the video was never downloaded
    assert _repositories == []


def test_force_realigns_an_already_aligned_speech(monkeypatch, _repositories):
    monkeypatch.setattr(mod.SpeechAlignments, "exists", lambda id: True,
                        raising=False)
    aligner = _FakeAligner()

    record = _service(aligner).execute("sp-1", force=True)

    assert aligner.words is not None
    assert _repositories == [record]


def test_dry_run_aligns_without_storing(_repositories):
    record = _service().execute("sp-1", persist=False)

    assert record.cues
    assert _repositories == []


# ---- cue segmentation -------------------------------------------------------

def test_cues_respect_the_configured_word_budget(monkeypatch, _repositories):
    monkeypatch.setattr(
        mod.Speeches, "get",
        lambda id: _speech([SpeechText(
            lang="es", text=" ".join(["palabra"] * 12) + ".", original=True)]),
        raising=False)

    record = _service(subtitle_max_words=4, subtitle_max_chars=10 ** 6).execute("sp-1")

    assert len(record.cues) == 3
    # Word i is timed to slot i, so the second cue starts two seconds in.
    assert record.cues[0].start_ms == 0
    assert record.cues[1].start_ms == int(4 / WORDS_PER_SECOND * 1000)


def test_cue_times_come_from_the_first_and_last_word_it_covers(_repositories):
    record = _service(subtitle_max_words=2, subtitle_max_chars=10 ** 6).execute("sp-1")

    first = record.cues[0]
    assert first.start_ms == 0
    assert first.end_ms == int(1.9 / WORDS_PER_SECOND * 1000)
