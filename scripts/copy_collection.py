"""One-off migration: copy an indexed collection into a new one that differs only
in storage config — today, to give dev a TurboQuant-4bit twin of the serving
collection without re-embedding the corpus.

Why copy instead of re-index: the points already hold everything a collection
needs (dense vector, sparse vector, full payload), so moving them costs nothing
but I/O, while re-indexing would re-embed all 31k passages. Why copy instead of
``update_collection``: compression is baked in at creation, so a collection built
by ``ensure_collection`` under the target settings has *exactly* the shape a fresh
production index run will have — per-vector quantization plus ``on_disk`` originals
— whereas patching an existing collection in place only approximates it.

The target is created through the adapter's own ``ensure_collection``, so it also
picks up all payload indexes for free. Compression comes from the environment
(``QDRANT_QUANTIZATION``), which is also what puts the suffix on the derived name.

The source is only ever read, so the collection being copied cannot be damaged.
Idempotent: points keep their ids, so re-running overwrites rather than duplicates,
and an interrupted run can simply be repeated.

Usage:
    QDRANT_QUANTIZATION=tq4 uv run python scripts/copy_collection.py <source> [target]

Without a target the name is derived the way every service derives it, using the
source's own vector dimension — so no embedding provider needs to be reachable.
"""

import sys

from qdrant_client import models
from tqdm import tqdm

from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

_BATCH = 256

# The adapter's names for the two vectors a hybrid point carries.
_DENSE, _SPARSE = "dense", "sparse"


def _shape(client, name):
    """The source's vector dimension and whether it is hybrid, read off the
    collection itself rather than from an embedder probe."""
    params = client.get_collection(name).config.params
    vectors = params.vectors
    # A hybrid collection names its dense vector; a dense-only one does not.
    dim = vectors[_DENSE].size if isinstance(vectors, dict) else vectors.size
    return dim, bool(params.sparse_vectors)


def _vector(point):
    """Carry the point's vectors across untouched. ``scroll`` returns a plain list
    for a single unnamed vector and a dict for named ones, which is exactly what
    ``upsert`` expects back."""
    return point.vector


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    source = sys.argv[1]
    settings = get_settings()
    store = create_vector_store_from_env(settings)
    client = store.client

    if not client.collection_exists(source):
        sys.exit(f"Source collection '{source}' does not exist")
    dim, sparse = _shape(client, source)
    target = sys.argv[2] if len(sys.argv) > 2 else collection_name(settings, dim)
    if target == source:
        sys.exit(
            f"Source and target are both '{source}' — set QDRANT_QUANTIZATION (or pass "
            f"a target) so the derived name differs, or this would be a no-op")

    total = client.get_collection(source).points_count
    print(f"Copying {total} points: '{source}' -> '{target}' (dim {dim}, "
          f"sparse={sparse}, quantization={settings.qdrant_quantization})")

    store.ensure_collection(target, dim, sparse=sparse)

    offset, copied = None, 0
    with tqdm(total=total, desc="Copying points", unit="pt") as bar:
        while True:
            points, offset = client.scroll(
                collection_name=source, limit=_BATCH, offset=offset,
                with_vectors=True, with_payload=True)
            if not points:
                break
            client.upsert(
                collection_name=target,
                points=[
                    models.PointStruct(id=p.id, vector=_vector(p), payload=p.payload)
                    for p in points
                ],
                wait=True,
            )
            copied += len(points)
            bar.update(len(points))
            if offset is None:
                break

    # Read the count back from the server rather than trusting the loop: an
    # indexing hiccup would show up here and nowhere else.
    landed = client.get_collection(target).points_count
    print(f"Done: copied {copied}, target now holds {landed} points "
          f"({'matches' if landed == total else 'DOES NOT MATCH'} the source's {total})")


if __name__ == "__main__":
    main()
