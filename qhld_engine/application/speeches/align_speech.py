"""Align one speech's transcript to its own video, producing subtitle cues.

The Congress publishes a separate video cut per intervention, so this never has to
find a speech inside a six-hour sitting: it is one clip against one transcript. And
because the transcript comes from the Diario de Sesiones, no speech recognition is
involved — the stenographers supply the words and the acoustic model supplies only
the timing, which is what makes the resulting subtitles as accurate as the official
record.

What is stored is the cue *numbers* — when each line starts and ends, and which
characters of the speech it covers. The subtitle text itself is sliced out of the
stored transcript when a WebVTT track is rendered, so subtitles cannot drift away
from the text readers see, and the stored document holds no second copy of the
prose.
"""

from hashlib import sha256

from qhld_ai.domain.annotations import annotation_spans
from qhld_ai.domain.subtitles import build_cues, word_spans
from qhld_ai.infrastructure.audio.ffmpeg import decode_pcm, SAMPLE_RATE
from qhld_ai.infrastructure.config.settings import get_settings
from tipi_data.models.speech_alignment import Cue, SpeechAlignment
from tipi_data.repositories.speech_alignments import SpeechAlignments
from tipi_data.repositories.speeches import Speeches

from qhld_engine.logger import get_logger

log = get_logger(__name__)


class NotAlignable(Exception):
    """The speech cannot be aligned at all — no published video, or no text.

    Distinct from a low-confidence alignment: this is "there was nothing to do",
    which must not be recorded as a result.
    """


class AlignSpeech:
    def __init__(self, settings=None, aligner=None, decode=None):
        self.settings = settings or get_settings()
        self.aligner = aligner if aligner is not None else self._aligner_from_settings()
        self.decode = decode or decode_pcm

    def _aligner_from_settings(self):
        from qhld_ai.infrastructure.aligner.factory import create_aligner_from_env

        return create_aligner_from_env(self.settings)

    def execute(self, speech_id, force=False, persist=True):
        """Align the speech and return its alignment.

        ``force`` re-aligns a speech that already has one; without it an existing
        alignment is returned untouched, since re-aligning costs a fresh download of
        the whole video.
        """
        speech = Speeches.get(speech_id)
        if not force and SpeechAlignments.exists(speech.id):
            log.info(f"{speech.id} is already aligned (pass --force to redo it)")
            return SpeechAlignments.get(speech.id)

        block_index, block = self._spoken_block(speech)
        words, spans = self._alignable_words(block.text)
        if not words:
            raise NotAlignable(f"{speech.id} has no alignable words")
        if not speech.video_link:
            raise NotAlignable(
                f"{speech.id} has no video; the Congress has not published it yet")

        log.info(f"aligning {speech.id} ({len(words)} words, {block.lang})")
        samples = self.decode(speech.video_link)
        alignment = self.aligner.align(samples, SAMPLE_RATE, words)

        cues = self._cues(block.text, alignment.words, spans)
        verdict = "ok" if alignment.score >= self.settings.aligner_min_score else "low"
        if verdict == "low":
            log.warning(
                f"{speech.id} aligned at {alignment.score} — below "
                f"{self.settings.aligner_min_score}; stored and flagged for review")

        record = SpeechAlignment(
            _id=speech.id,
            lang=block.lang,
            block_index=block_index,
            cues=cues,
            text_sha256=sha256(block.text.encode("utf-8")).hexdigest(),
            text_length=len(block.text),
            model_id=alignment.model.id if alignment.model else None,
            model_revision=alignment.model.revision if alignment.model else None,
            model_sha256=alignment.model.sha256 if alignment.model else None,
            score=alignment.score,
            verdict=verdict,
            audio_seconds=len(samples) / SAMPLE_RATE,
        )
        if persist:
            SpeechAlignments.save(record)
        return record

    @staticmethod
    def _spoken_block(speech):
        """The block of text the audio actually contains, and its position.

        A co-official-language intervention is published as the original followed by
        its full Spanish interpretation, and only the original was spoken. Aligning
        both would offer roughly twice as much text as there is speech, which is the
        textbook way to make a forced aligner fail badly.
        """
        for index, block in enumerate(speech.speech):
            if block.original:
                return index, block
        raise NotAlignable(f"{speech.id} has no as-delivered text block")

    @staticmethod
    def _alignable_words(text):
        """The words to align, and where each sits in ``text``.

        Stage directions are skipped: applause and an interjection from the floor are
        audible but are not this speaker's words, so feeding them to the aligner
        would ask it to find text nobody in the clip said. The offsets returned are
        still into the full stored string, because that is what the transcript is
        rendered from and what search highlights are located in.
        """
        skip = annotation_spans(text)
        words, spans = [], []
        for word in word_spans(text):
            if any(start <= word.char_start < end for start, end in skip):
                continue
            words.append(word.text)
            spans.append((word.char_start, word.char_end))
        return words, spans

    def _cues(self, text, timings, spans):
        """Group the timed words into subtitle-sized cues.

        Cues are cut from the text, then take the start of their first timed word and
        the end of their last. Cutting on the text rather than on the timings is what
        makes the stenographers' small departures from the words actually spoken
        harmless — they vanish inside a cue whose boundaries land — and it is why a
        cue spanning nothing but a stage direction simply gets no timing and is
        dropped.

        Both sequences are in document order, so one walking index matches them.
        """
        cues = []
        word = 0
        for span in build_cues(text, max_chars=self.settings.subtitle_max_chars,
                               max_words=self.settings.subtitle_max_words):
            while word < len(spans) and spans[word][0] < span.char_start:
                word += 1
            first = word
            while word < len(spans) and spans[word][1] <= span.char_end:
                word += 1
            if first == word:
                continue
            start = timings[first].start
            end = max(timings[word - 1].end, start)
            cues.append(Cue(start_ms=int(start * 1000), end_ms=int(end * 1000),
                            char_start=span.char_start, char_end=span.char_end))
        return cues
