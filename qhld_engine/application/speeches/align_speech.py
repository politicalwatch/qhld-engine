"""Align a speech's transcript to its own video, producing subtitle cues.

The Congress publishes a separate video cut per intervention, so this never has to
find a speech inside a six-hour sitting: it is one clip against one transcript. And
because the transcript comes from the Diario de Sesiones, no speech recognition is
involved — the stenographers supply the words and the acoustic model supplies only
the timing, which is what makes the resulting subtitles as accurate as the official
record.

**Every language block is timed, not only the as-delivered one.** A co-official-language
intervention is published as the original followed by its full Spanish interpretation,
and both go through the same audio, producing one track each. Two reasons, both measured:

- Most readers only read Spanish. Aligning the original alone left a Galician speech
  with a Galician track and nothing else, and the page opens on whichever language tab
  best matches the search — so the reader of the translation got subtitles they could
  not read.
- The as-delivered block is often not the one that covers the clip. Where the speaker
  code-switched, the Diario marks only the co-official *portion* as original while the
  Spanish block holds the whole speech: measured across 827 co-official speeches, 187
  have blocks of very different length, and on those the original-only alignment scored
  1.0-1.5 out of 100 — two Catalan words smeared across a six-minute clip — where the
  Spanish block scored 66.7-77.2. Timing both and letting each carry its own score is
  what stops one bad block from being the only thing on offer.

Timing a translation against audio in another language works because a forced aligner
only has to find a monotone path, and Spanish, Catalan and Galician are close enough
that it finds the right one: median 0 ms against the as-delivered track on shared
anchors. Basque is the exception and drifts; it is stored anyway, with the score that
says so, because for the great majority of Basque interventions the "translation" block
is in fact the Spanish the speaker really spoke.

What is stored is the cue *numbers* — when each line starts and ends, and which
characters of the speech it covers. The subtitle text itself is sliced out of the
stored transcript when a WebVTT track is rendered, so subtitles cannot drift away
from the text readers see, and the stored document holds no second copy of the
prose.
"""

from qhld_ai.domain.annotations import annotation_spans
from qhld_ai.domain.ports.aligner import AlignRequest
from qhld_ai.domain.subtitles import build_cues, text_fingerprint, word_spans
from qhld_ai.infrastructure.audio.pyav import decode_pcm, SAMPLE_RATE
from qhld_ai.infrastructure.config.settings import get_settings
from tipi_data.models.speech_alignment import Cue, SpeechAlignment, track_id
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
        """Align every language block of the speech and return one record each.

        ``force`` re-aligns languages that already have a track; without it those are
        returned untouched and only the missing ones are timed. When nothing is missing
        the video is never fetched at all, which is the expensive part.
        """
        speech = Speeches.get(speech_id)
        blocks = self._alignable_blocks(speech)
        if not blocks:
            raise NotAlignable(f"{speech.id} has no alignable words")

        stored, pending = [], []
        for block_index, block, words, spans in blocks:
            if not force and SpeechAlignments.exists(speech.id, block.lang):
                stored.append(SpeechAlignments.get(speech.id, block.lang))
                continue
            pending.append((block_index, block, words, spans))
        if not pending:
            log.info(f"{speech.id} is already aligned in "
                     f"{', '.join(a.lang for a in stored)} (pass --force to redo it)")
            return stored
        if not speech.video_link:
            raise NotAlignable(
                f"{speech.id} has no video; the Congress has not published it yet")

        log.info(f"aligning {speech.id} in "
                 + ", ".join(f"{block.lang} ({len(words)} words)"
                             for _, block, words, _ in pending))
        samples = self.decode(speech.video_link)
        # One call, so the clip goes through the acoustic model once however many blocks
        # it carries: what a second track costs is the search that places its words.
        alignments = self.aligner.align_all(
            samples, SAMPLE_RATE,
            [AlignRequest(words=words, lang=block.lang)
             for _, block, words, _ in pending])

        records = list(stored)
        for (block_index, block, _, spans), alignment in zip(pending, alignments):
            record = self._record(speech, block_index, block, spans, alignment,
                                  len(samples) / SAMPLE_RATE)
            if persist:
                SpeechAlignments.save(record)
            records.append(record)
        return records

    def _record(self, speech, block_index, block, spans, alignment, audio_seconds):
        cues = self._cues(block.text, alignment.words, spans)
        verdict = "ok" if alignment.score >= self.settings.aligner_min_score else "low"
        if verdict == "low":
            # A translated block is expected to score low and it is not a fault: the
            # check asks whether the audio says these words, and the Spanish rendering
            # of a Galician speech does not, however exactly its cues land. Saying so
            # here keeps the log from reading as a defect on every co-official speech.
            note = ("expected for a translated block, whose words are not the ones "
                    "spoken" if not block.original
                    else f"below {self.settings.aligner_min_score}")
            log.warning(f"{speech.id} aligned at {alignment.score} in {block.lang} — "
                        f"{note}; stored and flagged for review")

        # Taken with the same function the reader re-computes it with, so the drift
        # guard cannot fail through the two ends disagreeing about the hash.
        text_sha256, text_length = text_fingerprint(block.text)
        return SpeechAlignment(
            _id=track_id(speech.id, block.lang),
            speech_id=speech.id,
            lang=block.lang,
            block_index=block_index,
            original=bool(block.original),
            cues=cues,
            text_sha256=text_sha256,
            text_length=text_length,
            model_id=alignment.model.id if alignment.model else None,
            model_revision=alignment.model.revision if alignment.model else None,
            model_sha256=alignment.model.sha256 if alignment.model else None,
            score=alignment.score,
            verdict=verdict,
            audio_seconds=audio_seconds,
        )

    @classmethod
    def _alignable_blocks(cls, speech):
        """Every block with words to time, in document order.

        A block with nothing alignable — a stray heading, an empty translation — is
        dropped rather than raising, so one useless block cannot cost a speech the
        track its sibling would have had.
        """
        blocks = []
        for index, block in enumerate(speech.speech):
            words, spans = cls._alignable_words(block.text)
            if words:
                blocks.append((index, block, words, spans))
        return blocks

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
