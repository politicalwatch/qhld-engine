"""Collection naming for speech embeddings.

Pure string logic (kept beside the adapter for cohesion). Collections are named
per embedding model + dimension so that indexing different models — the TFM's
benchmark A/B — writes to separate collections instead of clobbering each other.
An explicit ``qdrant_collection`` setting overrides the derived name.
"""

import re

from qhld_engine.infrastructure.config.settings import Settings


def collection_name(settings: Settings, dim: int) -> str:
    if settings.qdrant_collection:
        return settings.qdrant_collection
    provider = settings.embedding_provider.lower()
    model = re.sub(r"[^a-z0-9]+", "_", settings.embedding_model.lower()).strip("_")
    name = f"speeches__{provider}__{model}__{dim}"
    # Hybrid collections carry an extra sparse (lexical) vector per point, so
    # they get their own name — dense-only collections stay untouched.
    sparse = (settings.sparse_provider or "").lower()
    if sparse and sparse != "none":
        name += f"__{sparse}"
    return name
