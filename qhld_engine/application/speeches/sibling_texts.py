"""Attach each co-official passage's Spanish interpretation to its payload.

Search reranks a passage against its Spanish sibling as well as itself, because a
cross-encoder scores a language-mismatched pair as junk however relevant it is —
without this a language-filtered query returned nothing at all. The pairing is
cosine over the embeddings indexing already produced, so it costs no inference;
the domain rule lives in ``qhld_ai.domain.sibling_pairing``.

Used from two places, which is why it is its own module: the indexer writes the
key as it embeds, and the backfill writes it onto an already-indexed corpus.
"""

from qhld_ai.domain.sibling_pairing import nearest_texts

# Payload key read by SearchSpeeches._rerank_against_siblings.
SIBLING_KEY = "sibling_texts"


def siblings_for(items):
    """The sibling texts of every chunk of ONE speech, parallel to ``items``.

    ``items`` are ``(payload, vector)`` pairs covering the whole speech — both
    blocks, since the Spanish ones are the candidates. Spanish chunks get an
    empty list (they are already in the query's language and need no sibling),
    and so does every chunk of a monolingual speech, which has no candidates.
    """
    spanish = [(payload.get("text") or "", vector)
               for payload, vector in items if payload.get("lang") == "es"]
    return [
        [] if payload.get("lang") == "es" else nearest_texts(vector, spanish)
        for payload, vector in items
    ]


def attach_siblings(points):
    """Add ``sibling_texts`` in place to the co-official points of one speech."""
    items = [(point.payload, point.vector) for point in points]
    for point, texts in zip(points, siblings_for(items)):
        if texts:
            point.payload[SIBLING_KEY] = texts
