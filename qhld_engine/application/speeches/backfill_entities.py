"""Backfill ``Speech.entities`` over already-extracted speeches.

New speeches get entities at extract time (``ExtractSpeeches``); this one-shot
service tags the existing corpus, which was extracted before the field existed.
It re-uses the stored Spanish text block — no Congress API / Diario PDF refetch —
and touches ONLY ``entities``: the mentions/interruptions a previous run stored
stay exactly as they are.

Follows ``BackfillMentions``: drain the Mongo cursor into memory before the slow
NER loop, incremental by default (speeches already carrying entities are
skipped; ``--all`` re-tags everything, needed after a normalization or stoplist
change). A speech whose text genuinely references no entity resolves to ``[]``
and so is revisited by a later incremental run — harmless for a one-shot
backfill.
"""

from tqdm import tqdm

from qhld_engine.logger import get_logger
from qhld_engine.application.speeches.mention_tagging import MentionTagger, es_text

from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)


class BackfillEntities:
    def __init__(self, tagger=None):
        self.tagger = tagger or MentionTagger(Deputies.get_all())

    def execute(self, references=None, incremental=True):
        if references:
            speeches = list(Speeches.by_references(references))
        else:
            speeches = list(Speeches.all())
            if incremental:
                total = len(speeches)
                speeches = [s for s in speeches if not s.entities]
                log.info(
                    f"Incremental: {len(speeches)} of {total} speeches untagged "
                    f"(pass --all to re-tag the whole corpus)")
        log.info(f"Tagging entities for {len(speeches)} speeches")
        tagged = 0
        for speech in tqdm(speeches, desc="Tagging entities", unit="speech"):
            entities = self.tagger.tag_entities(es_text(speech.speech))
            speech.entities = entities
            Speeches.save(speech)
            if entities:
                tagged += 1
        log.info(f"Done: {tagged}/{len(speeches)} speeches had at least one entity")
