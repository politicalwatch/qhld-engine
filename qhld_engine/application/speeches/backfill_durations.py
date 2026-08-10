"""Backfill ``Speech.duration`` over already-extracted speeches.

New speeches get their clip length at extract time (``ExtractSpeeches``); this
one-shot service fills in the existing corpus, which was extracted before the field
existed. Only the container header is read, so nothing is downloaded and no text is
touched.

Unlike its ``BackfillMentions``/``BackfillEntities`` siblings this one is **network
bound, not CPU bound** — each speech is a round trip to the Congress CDN that spends
its time waiting. Run serially the corpus takes the better part of an hour; probed
concurrently it takes minutes, so the loop keeps a small pool in flight. Writes stay
on this thread: only the probe is parallel.

Incremental by default (speeches that already carry a duration are skipped; ``--all``
re-probes everything, needed if the Congress republishes a clip). A speech whose video
is unpublished or unreadable keeps ``duration=None`` and so is revisited by a later
run — which is the right behaviour, since videos appear days after the sitting.
"""

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from qhld_ai.infrastructure.audio.pyav import DurationUnavailable, probe_duration

from qhld_engine.logger import get_logger

from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)

# Enough to hide the round trips, small enough to stay a polite neighbour to a
# public service we do not own.
WORKERS = 8
# Probe this many before writing them, so a long run persists progress as it goes
# rather than holding every result until the end.
CHUNK = 200


class BackfillDurations:
    def __init__(self, probe=probe_duration, workers=WORKERS):
        self._probe = probe
        self._workers = workers

    def execute(self, references=None, incremental=True):
        if references:
            speeches = list(Speeches.by_references(references))
        else:
            speeches = list(Speeches.all())
            if incremental:
                total = len(speeches)
                speeches = [s for s in speeches if not s.duration]
                log.info(
                    f"Incremental: {len(speeches)} of {total} speeches unmeasured "
                    f"(pass --all to re-probe the whole corpus)")
        speeches = [s for s in speeches if s.video_link]
        log.info(f"Probing the clip of {len(speeches)} speeches")

        measured = failed = 0
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            with tqdm(total=len(speeches), desc="Probing clips",
                      unit="speech") as progress:
                for start in range(0, len(speeches), CHUNK):
                    chunk = speeches[start:start + CHUNK]
                    for speech, duration in zip(
                            chunk, pool.map(self._duration, chunk)):
                        if duration is None:
                            failed += 1
                        else:
                            speech.duration = duration
                            Speeches.save(speech)
                            measured += 1
                        progress.update(1)
        log.info(f"Done: {measured} measured, {failed} without a readable video")

    def _duration(self, speech):
        try:
            return self._probe(speech.video_link)
        except DurationUnavailable as exc:
            log.warning(f"No duration for {speech.id}: {exc}")
            return None
