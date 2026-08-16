"""Backfill subtitle cues over the whole stored corpus.

``AlignSpeech`` times one intervention; this drives it over all of them. New speeches
are aligned as they arrive, so like its ``BackfillDurations`` sibling this exists to
fill in a corpus that was extracted before the feature did — measured at 3,974 of
3,994 speeches carrying no track at all, monolingual Spanish ones included.

**Downloading and aligning overlap, because they compete for nothing.** Pulling a clip
from the Congress CDN is network-bound and forced alignment is CPU-bound, so the next
clips are fetched while the current one is still in the acoustic model. Measured: one
stream runs at 4.85 s per audio-minute and three at 1.05 — a *superlinear* speedup,
which says the CDN throttles per connection rather than per client. Three streams
therefore outrun the aligner's 2.12 s per audio-minute about twofold, and downloading
stops being visible in the wall clock at all. Align throughput alone sets it: about
20 hours for the corpus's 538 hours of audio.

The prefetch is a sliding window rather than "submit them all", and that is a memory
decision, not a style one. ``decode_pcm`` holds the whole clip in RAM (~16 MB for an
average intervention, ~200 MB for the longest), so a pool handed all 3,974 speeches at
once would try to hold the entire corpus. The window keeps a handful in flight.

Only the download is parallel; alignment and every Mongo write stay on the calling
thread, exactly as ``BackfillDurations`` does it. One aligner, one model, no shared
mutable state.

Incremental by default: a speech whose every block already has a track is skipped
without its video ever being fetched, which is what makes an interrupted run cheap to
resume. Nothing tracks progress but the stored cues themselves — a track that exists
is a block that is done — so a run that dies at hour 15 loses only the clip it was
holding. ``--all`` re-aligns everything, which is what a change to cue segmentation or
to the acoustic model needs.
"""

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from qhld_ai.infrastructure.audio.pyav import AudioDecodeError, decode_pcm

from qhld_engine.application.speeches.align_speech import AlignSpeech, NotAlignable
from qhld_engine.logger import get_logger

from tipi_data.repositories.speech_alignments import SpeechAlignments
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)

# Three streams already run the aligner's supplier about twice as fast as the aligner
# consumes, so more would only buy memory pressure and a heavier footprint on a public
# service we do not own.
DOWNLOAD_WORKERS = 3


class AlignCorpus:
    def __init__(self, align=None, decode=decode_pcm, workers=DOWNLOAD_WORKERS):
        # Built lazily rather than as a default argument: constructing it loads the
        # acoustic model, which a --dry-run must not pay for.
        self._align = align
        self._decode = decode
        self._workers = workers

    @property
    def align(self):
        if self._align is None:
            self._align = AlignSpeech()
        return self._align

    def execute(self, references=None, incremental=True, limit=None, persist=True):
        speeches = self.pending(references, incremental)
        if limit is not None:
            speeches = speeches[:limit]
        # Whole seconds, so the bar's running total cannot drift past its own end
        # through float accumulation over four thousand additions.
        audio = sum(int(s.duration or 0) for s in speeches)
        log.info(f"Aligning {len(speeches)} speeches, {audio / 3600:.1f} h of audio")
        if not speeches:
            return

        aligned = failed = 0
        unreachable = []
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            with tqdm(total=audio, desc="Aligning speeches", unit="s",
                      unit_scale=True) as progress:
                for speech, samples, error in self._with_audio(pool, speeches):
                    if error is None:
                        try:
                            self.align.execute(speech.id, force=not incremental,
                                               persist=persist, samples=samples)
                            aligned += 1
                        except NotAlignable as exc:
                            error = exc
                        except Exception as exc:  # noqa: BLE001
                            # One malformed speech must not cost the other 3,973 and
                            # the twenty hours already spent on them.
                            error = exc
                    if error is not None:
                        log.warning(f"Not aligned: {speech.id}: {error}")
                        unreachable.append(speech.id)
                        failed += 1
                    progress.update(int(speech.duration or 0))

        log.info(f"Done: {aligned} aligned, {failed} could not be")
        if unreachable:
            # Listed, not merely counted: on a run this long a warning logged fifteen
            # hours ago is not something anyone can go back and audit.
            log.warning(f"Speeches left unaligned: {', '.join(unreachable)}")

    def pending(self, references=None, incremental=True):
        """The speeches with at least one block still to time, newest first.

        Newest first because that is where the readers are, so an interrupted run has
        still done the half of the corpus anybody is looking at.

        Unlike its ``BackfillDurations`` sibling, ``references`` does **not** imply a
        redo. Re-probing a duration a second time costs a header request; re-aligning a
        speech costs its whole video and a pass through the acoustic model, so a redo
        has to be asked for by name (``--all``) rather than implied by narrowing the
        scope.
        """
        if references:
            speeches = list(Speeches.by_references(references))
        else:
            speeches = list(Speeches.all())
        speeches = [s for s in speeches if s.video_link and s.speech]
        if incremental:
            total = len(speeches)
            speeches = [s for s in speeches if self._missing(s)]
            log.info(
                f"Incremental: {len(speeches)} of {total} speeches have a block with "
                f"no cues (pass --all to re-align them all)")
        return sorted(speeches, key=lambda s: s.date or 0, reverse=True)

    @staticmethod
    def _missing(speech):
        """Whether any block of ``speech`` still lacks a track.

        Counted over the blocks that could ever *have* one — the same reading
        ``AlignSpeech`` aligns by — rather than over every stored block. A block with
        nothing but a stage direction in it is never timed, so measuring against all
        blocks would leave such a speech permanently pending and re-download its video
        on every run. (No block in the corpus is currently in that state; this is what
        stops one appearing from quietly costing an hour a night.)

        Asked through the repository's batched summary rather than a scan of the
        alignment collection: one round trip per speech, and it needs no index, which
        is the whole reason that collection is keyed the way it is.
        """
        blocks = [(block.lang, bool(block.original))
                  for _, block, _, _ in AlignSpeech._alignable_blocks(speech)]
        if not blocks:
            return False
        return len(SpeechAlignments.summaries(speech.id, blocks)) < len(blocks)

    def _with_audio(self, pool, speeches):
        """Yield ``(speech, samples, error)``, downloading ahead of the consumer.

        A sliding window: one clip is submitted for every clip taken, so memory holds
        ``workers + 2`` at the peak — the ``workers + 1`` in the window, plus the one
        the caller is aligning. ``ThreadPoolExecutor.map`` would instead submit all
        3,974 at once and try to hold the entire corpus.

        The replacement is submitted *before* the result is waited on, which is what
        keeps the pool busy through both the wait and the align that follows it. That
        is worth one clip of memory: at ~16 MB for an average intervention the whole
        window is well under 100 MB, and the alternative is a downloader that idles
        for the two seconds per audio-minute the aligner spends.
        """
        queue = []
        upcoming = iter(speeches)

        def submit_next():
            speech = next(upcoming, None)
            if speech is not None:
                queue.append((speech, pool.submit(self._audio, speech)))

        for _ in range(self._workers + 1):
            submit_next()
        while queue:
            speech, future = queue.pop(0)
            submit_next()
            samples, error = future.result()
            yield speech, samples, error

    def _audio(self, speech):
        """The clip's samples, or the reason there are none.

        Returned rather than raised so that one unreachable video costs a single
        intervention instead of the whole run — and because nothing is stored for it,
        a later run simply picks it up again.
        """
        try:
            return self._decode(speech.video_link), None
        except AudioDecodeError as exc:
            return None, exc
