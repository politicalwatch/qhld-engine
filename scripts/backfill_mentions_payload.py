"""One-off backfill: re-stamp each speech's mention payload onto its already-indexed
points, so the mentions filter reflects what Mongo actually holds.

Mentions are speech-level (like entities), so one ``set_payload`` per speech (selected
by a ``speech_id`` filter) covers every chunk-point. Run AFTER the Mongo-side tagging is
correct (``qhld speeches tag-mentions``); this script only copies, it never tags.

Why it exists: unlike ``entities``, the mention payload was never re-stamped after the
annotation-stripping change, which stopped crediting a speaker with "mentioning" whoever
interjected from the floor. Indexed collections therefore still carry the pre-stripping
result — people who merely interrupted (the presiding officer, hecklers) counted as
mentions — plus they miss everyone the tagger learned to resolve since. Fresh indexing
runs are fine: ``IndexSpeeches._payload`` stamps all three keys itself.

Stamps ``mentions`` (the filterable id list), ``mention_types`` and ``mention_counts``
together, since all three are derived from the same records and would otherwise
disagree. Every speech is stamped — an empty list for mention-less speeches — matching
what ``_payload`` writes on a fresh index. Idempotent.

Usage:
    uv run python scripts/backfill_mentions_payload.py [collection]

Without an argument the target collection is resolved the same way indexing resolves it
(embedder probe -> per-model collection name), which requires the configured embedding
provider to be reachable.
"""

import sys

from qdrant_client import models
from tqdm import tqdm

from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

from tipi_data.repositories.speeches import Speeches


def main():
    settings = get_settings()
    store = create_vector_store_from_env(settings)
    if len(sys.argv) > 1:
        collection = sys.argv[1]
    else:
        dim = len(create_embedder_from_env(settings).embed_query("probe"))
        collection = collection_name(settings, dim)

    speeches = list(Speeches.all())
    print(f"Backfilling mentions for {len(speeches)} speeches into '{collection}'")
    stamped = 0
    for speech in tqdm(speeches, desc="Stamping mentions", unit="speech"):
        # Same derivation as IndexSpeeches._payload, so a backfilled point and a
        # freshly-indexed one are indistinguishable.
        resolved = [m for m in (speech.mentions or []) if m.person_id]
        payload = {
            "mentions": [m.person_id for m in resolved],
            "mention_types": {m.person_id: m.person_type for m in resolved},
            "mention_counts": {m.person_id: m.count for m in resolved},
        }
        store.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=models.Filter(must=[models.FieldCondition(
                key="speech_id", match=models.MatchValue(value=speech.id))]),
            wait=True,
        )
        if resolved:
            stamped += 1
    print(f"Done: {stamped}/{len(speeches)} speeches carry at least one mention")


if __name__ == "__main__":
    main()
