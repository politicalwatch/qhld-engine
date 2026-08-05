"""One-off migration: fold each curated speaker variant into its canonical spelling.

The source credits some deputies under more than one ``orador`` spelling, and only one
of them matches the deputy catalog. Speeches already stored under the other spelling
have to be renamed once; from now on the extractor writes the canonical name itself
(``ExtractSpeeches._canonical_speaker``), so this is needed once per pre-existing
corpus, not on every run.

Renaming also repairs ``constituency``: the index-time join is a verbatim
``Deputies.name`` -> ``speaker`` lookup, so the orphan spelling never matched it and its
points carry no province. Both keys are set together here.

Payload only. Speech ids key on the RAW ``orador`` (``ExtractSpeeches._content_id``), so
nothing moves document and nothing needs re-embedding.

Run Mongo FIRST, then each collection — a collection renamed while Mongo still holds the
old value would be undone by the next indexing run.

Usage:
    uv run python scripts/backfill_canonical_speaker.py --mongo
    uv run python scripts/backfill_canonical_speaker.py [collection]

Without a collection argument the target is resolved the way indexing resolves it
(embedder probe -> per-model collection name), which needs the configured embedding
provider reachable. Idempotent: re-running matches nothing the second time.
"""

import sys

from qdrant_client import models

from qhld_ai.application.persons_catalog import (
    canonical_speakers,
    load_deputy_profiles,
)
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

from tipi_data import db
from tipi_data.repositories.deputies import Deputies


def _constituencies():
    return {d.name: d.constituency
            for d in Deputies.get_all() if d.name and d.constituency}


def migrate_mongo(renames):
    """Rename the stored speeches, and their derived ``speaker_surname`` with them."""
    for variant, canonical in renames.items():
        result = db.speeches.update_many(
            {"speaker": variant},
            {"$set": {"speaker": canonical,
                      "speaker_surname": canonical.split(",")[0].strip()}})
        print(f"  {variant!r} -> {canonical!r}: {result.modified_count} speeches")


def migrate_collection(renames, collection, store):
    """Rename the indexed points, stamping the constituency the orphan spelling missed."""
    provinces = _constituencies()
    for variant, canonical in renames.items():
        payload = {"speaker": canonical}
        province = provinces.get(canonical)
        if province:
            payload["constituency"] = province
        result = store.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=models.Filter(must=[models.FieldCondition(
                key="speaker", match=models.MatchValue(value=variant))]),
            wait=True,
        )
        print(f"  {variant!r} -> {payload} ({result.status})")


def main():
    renames = canonical_speakers(load_deputy_profiles())
    if not renames:
        print("No curated speaker_variants — nothing to migrate.")
        return
    settings = get_settings()

    if "--mongo" in sys.argv:
        print(f"Renaming {len(renames)} speaker variants in Mongo")
        migrate_mongo(renames)
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        collection = args[0]
    else:
        dim = len(create_embedder_from_env(settings).embed_query("probe"))
        collection = collection_name(settings, dim)
    store = create_vector_store_from_env(settings)
    print(f"Renaming {len(renames)} speaker variants in '{collection}'")
    migrate_collection(renames, collection, store)


if __name__ == "__main__":
    main()
