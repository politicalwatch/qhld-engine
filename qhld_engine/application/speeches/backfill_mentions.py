"""Backfill ``Speech.mentions`` and ``Speech.interruptions`` over already-extracted
speeches.

New speeches get both at extract time (``ExtractSpeeches``); this one-shot service
re-tags the existing corpus, which was extracted before the fields existed. It
re-uses the stored Spanish text block — no Congress API / Diario PDF refetch — so
it is cheap to re-run.

Follows ``IndexSpeeches``: drain the Mongo cursor into memory before the slow NER
loop (a live cursor held open across NER trips MongoDB's idle-cursor timeout).
Incremental by default: speeches that already carry mentions are skipped; ``--all``
re-tags everything (needed after a threshold or model change — and after the
annotation-stripping change, which reshapes the mentions of every speech that
carries stenographer annotations). A speech whose text genuinely names no deputy
resolves to ``[]`` and so is revisited by a later incremental run — harmless for a
one-shot backfill.
"""

from tqdm import tqdm

from qhld_engine.logger import get_logger
from qhld_engine.application.speeches.mention_tagging import MentionTagger, es_text

from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)


class BackfillMentions:
    def __init__(self, tagger=None):
        self.tagger = tagger or MentionTagger(Deputies.get_all())

    def execute(self, references=None, incremental=True):
        if references:
            speeches = list(Speeches.by_references(references))
        else:
            speeches = list(Speeches.all())
            if incremental:
                total = len(speeches)
                speeches = [s for s in speeches if not s.mentions]
                log.info(
                    f"Incremental: {len(speeches)} of {total} speeches untagged "
                    f"(pass --all to re-tag the whole corpus)")
        log.info(f"Tagging mentions for {len(speeches)} speeches")
        tagged = 0
        for speech in tqdm(speeches, desc="Tagging mentions", unit="speech"):
            text = es_text(speech.speech)
            mentions = self.tagger.tag(text)
            speech.mentions = mentions
            speech.interruptions = self.tagger.tag_interruptions(
                text, speaker=speech.speaker)
            Speeches.save(speech)
            if mentions:
                tagged += 1
        log.info(f"Done: {tagged}/{len(speeches)} speeches had at least one mention")
