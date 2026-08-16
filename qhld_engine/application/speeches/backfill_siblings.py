"""Application service: write ``sibling_texts`` onto an already-indexed corpus.

The indexer attaches the key as it embeds, so only speeches indexed BEFORE that
existed need this. It reads the vectors already stored in Qdrant and writes back
a payload key — no re-embedding, no re-extraction, and no change to any block
text, which is what makes it safe to run beside jobs that fingerprint the text
(subtitle alignment) and cheap enough not to need a re-index.

Only co-official speeches are touched: a monolingual Spanish speech has no
sibling to pair with, and a Spanish passage needs none.
"""

from tqdm import tqdm

from qhld_engine.application.speeches.sibling_texts import SIBLING_KEY, siblings_for
from qhld_engine.logger import get_logger
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

log = get_logger(__name__)

CO_OFFICIAL = ("ca", "gl", "eu")


class BackfillSiblings:
    def __init__(self, settings=None, embedder=None, store=None):
        self.settings = settings or get_settings()
        self.store = store or create_vector_store_from_env(self.settings)
        # One probe embedding, only to read the model dimension the collection
        # name carries — nothing in this service embeds any speech text.
        embedder = embedder or create_embedder_from_env(self.settings)
        self.collection = collection_name(self.settings, len(embedder.embed_query("probe")))

    def execute(self, dry_run=False):
        speech_ids = sorted(self.store.distinct_values(
            self.collection, "speech_id", where={"lang": list(CO_OFFICIAL)}))
        log.info(f"{len(speech_ids)} co-official speeches in {self.collection}")
        written = skipped = 0
        for speech_id in tqdm(speech_ids, desc="Pairing siblings", unit="speech"):
            # Every point of the speech: the Spanish ones are the candidates.
            points = self.store.points_by(self.collection, "speech_id", speech_id)
            payloads = {
                point.id: {SIBLING_KEY: texts}
                for point, texts in zip(points, siblings_for(
                    [(p.payload, p.vector) for p in points]))
                if texts
            }
            if not payloads:
                # A speech whose co-official block has no Spanish counterpart:
                # nothing to pair against, and nothing search can compensate with.
                skipped += 1
                log.debug(f"No Spanish block to pair against for speech {speech_id}")
                continue
            if not dry_run:
                self.store.set_payload(self.collection, payloads)
            written += len(payloads)
        log.info(
            f"{'Would write' if dry_run else 'Wrote'} {SIBLING_KEY} on {written} "
            f"passages across {len(speech_ids) - skipped} speeches "
            f"({skipped} with no Spanish block)")
        return written, skipped
