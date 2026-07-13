"""One-off backfill: stamp each deputy's constituency onto their already-indexed
points, so the constituency filter works without re-embedding the corpus.

Deputy names match the corpus ``speaker`` payload verbatim, so one
``set_payload`` per deputy (selected by a speaker filter) covers every point.
Idempotent — re-running just overwrites the same values. New indexing runs set
the field themselves (see ``IndexSpeeches._payload``), so this script is only
needed once per pre-existing collection.

Usage:
    uv run python scripts/backfill_constituency.py [collection]

Without an argument the target collection is resolved the same way indexing
resolves it (embedder probe -> per-model collection name), which requires the
configured embedding provider to be reachable.
"""

import sys

from qdrant_client import models

from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

from tipi_data.repositories.deputies import Deputies


def main():
    settings = get_settings()
    store = create_vector_store_from_env(settings)
    if len(sys.argv) > 1:
        collection = sys.argv[1]
    else:
        dim = len(create_embedder_from_env(settings).embed_query("probe"))
        collection = collection_name(settings, dim)

    deputies = [d for d in Deputies.get_all() if d.name and d.constituency]
    print(f"Backfilling constituency for {len(deputies)} deputies into '{collection}'")
    for deputy in deputies:
        result = store.client.set_payload(
            collection_name=collection,
            payload={"constituency": deputy.constituency},
            points=models.Filter(must=[models.FieldCondition(
                key="speaker", match=models.MatchValue(value=deputy.name))]),
            wait=True,
        )
        print(f"  {deputy.name} -> {deputy.constituency} ({result.status})")


if __name__ == "__main__":
    main()
