"""One-off backfill: stamp each speech's entity keys onto its already-indexed
points, so the entities filter works without re-embedding the corpus.

Entities are speech-level (like mentions), so one ``set_payload`` per speech
(selected by a ``speech_id`` filter) covers every chunk-point. Run AFTER
``qhld speeches tag-entities`` has populated ``Speech.entities`` in Mongo.
Idempotent — re-running just overwrites the same values. New indexing runs set
the field themselves (see ``IndexSpeeches._payload``), so this script is only
needed once per pre-existing collection.

Every speech gets the key stamped — an empty list for entity-less speeches —
matching what ``_payload`` writes on a fresh index.

Usage:
    uv run python scripts/backfill_entities.py [collection]

Without an argument the target collection is resolved the same way indexing
resolves it (embedder probe -> per-model collection name), which requires the
configured embedding provider to be reachable.
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
    print(f"Backfilling entities for {len(speeches)} speeches into '{collection}'")
    stamped = 0
    for speech in tqdm(speeches, desc="Stamping entities", unit="speech"):
        keys = sorted({e.key for e in (speech.entities or []) if e.key})
        store.client.set_payload(
            collection_name=collection,
            payload={"entities": keys},
            points=models.Filter(must=[models.FieldCondition(
                key="speech_id", match=models.MatchValue(value=speech.id))]),
            wait=True,
        )
        if keys:
            stamped += 1
    print(f"Done: {stamped}/{len(speeches)} speeches carry at least one entity key")


if __name__ == "__main__":
    main()
