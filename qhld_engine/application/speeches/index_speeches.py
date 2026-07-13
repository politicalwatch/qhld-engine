"""Application service: embed stored speeches and index them in the vector store
for semantic search.

Reads ``Speech`` documents from Mongo (the qhld-data ``Speeches`` repo), splits each
per-language block into passages, embeds them via the configured embedder, and
upserts one point per passage into Qdrant. It is the search-side sibling of
``ExtractSpeeches`` (which produces the speeches this consumes).

Both language blocks of a co-official speech (the original and its Spanish
interpretation) are indexed as separate points, tagged with ``lang``/``original`` in
the payload, so es queries hit the Spanish block and native-language queries hit the
original.

Indexing is idempotent: point ids are deterministic (``uuid5`` of
``speech_id:block:chunk``) and every speech's existing points are deleted before its
fresh ones are written, so re-runs and chunk-count changes never leave orphans.
"""

from uuid import NAMESPACE_DNS, uuid5

from tqdm import tqdm

from qhld_engine.logger import get_logger
from qhld_ai.domain.chunking import chunk_text
from qhld_ai.domain.ports.vector_store import VectorPoint
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)

# Stable namespace for deterministic point ids.
_NS = uuid5(NAMESPACE_DNS, "speeches.qhld.politicalwatch.es")


class IndexSpeeches:
    def __init__(self, settings=None, embedder=None, store=None, sparse_embedder=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embedder_from_env(self.settings)
        self.store = store or create_vector_store_from_env(self.settings)
        self.sparse_embedder = (
            sparse_embedder if sparse_embedder is not None
            else self._sparse_from_settings()
        )
        self.dim = len(self.embedder.embed_query("probe"))
        self.collection = collection_name(self.settings, self.dim)
        self._constituencies = None
        if self.sparse_embedder is None:
            self.store.ensure_collection(self.collection, self.dim)
        else:
            self.store.ensure_collection(self.collection, self.dim, sparse=True)

    def _sparse_from_settings(self):
        """Build the configured sparse embedder, or ``None`` for the "none"/unset
        default so dense-only indexing stays byte-identical."""
        provider = (self.settings.sparse_provider or "").lower()
        if not provider or provider == "none":
            return None
        from qhld_ai.infrastructure.sparse.factory import create_sparse_embedder_from_env

        return create_sparse_embedder_from_env(self.settings)

    def execute(self, references=None, incremental=True):
        # Drain the Mongo cursor into memory before the slow embed/upsert loop.
        # Iterating a live cursor while embedding each speech keeps it open for the
        # whole run and trips MongoDB's idle-cursor timeout (CursorNotFound) on a
        # full-corpus index. The drain itself is fast (find + validate, no embedding).
        if references:
            # A targeted re-index is always forced: the caller named these specific
            # references, typically because their text was just re-extracted.
            speeches = list(Speeches.by_references(references))
        else:
            speeches = list(Speeches.all())
            if incremental:
                indexed = self.store.distinct_values(self.collection, "speech_id")
                total = len(speeches)
                speeches = [s for s in speeches if s.id not in indexed]
                log.info(
                    f"Incremental: {len(speeches)} of {total} speeches not yet in "
                    f"{self.collection} (pass --all to re-index the whole corpus)")
        log.info(f"Indexing {len(speeches)} speeches into {self.collection}")
        for speech in tqdm(speeches, desc="Indexing speeches", unit="speech"):
            self._index_speech(speech)

    def _index_speech(self, speech):
        chunks = self._chunks(speech)
        # Delete first so a re-index with fewer chunks leaves no orphans.
        self.store.delete_by(self.collection, "speech_id", speech.id)
        if not chunks:
            log.debug(f"No text to index for speech {speech.id}")
            return
        texts = [text for _, _, text in chunks]
        vectors = self.embedder.embed_documents(texts)
        sparse_vectors = (
            self.sparse_embedder.embed_documents(texts)
            if self.sparse_embedder is not None
            else [None] * len(chunks)
        )
        points = [
            VectorPoint(id=point_id, vector=vector, payload=payload, sparse=sparse)
            for (point_id, payload, _text), vector, sparse in zip(
                chunks, vectors, sparse_vectors)
        ]
        self.store.upsert(self.collection, points)
        log.debug(f"Indexed {len(points)} passages for speech {speech.id}")

    def _chunks(self, speech):
        """The (point_id, payload, text) triples for every passage of every block."""
        result = []
        for block_index, block in enumerate(speech.speech):
            texts = chunk_text(
                block.text,
                self.settings.speech_chunk_chars,
                self.settings.speech_chunk_overlap,
            )
            for chunk_index, text in enumerate(texts):
                point_id = str(uuid5(_NS, f"{speech.id}:{block_index}:{chunk_index}"))
                payload = self._payload(speech, block, block_index, chunk_index, text)
                result.append((point_id, payload, text))
        return result

    def _constituency_map(self):
        """Speaker name -> province of election, from the deputy catalog. Deputy
        names match the corpus ``speaker`` values verbatim, so a plain dict lookup
        is the whole join; non-deputy speakers (ministers, witnesses) simply miss."""
        if self._constituencies is None:
            self._constituencies = {
                d.name: d.constituency
                for d in Deputies.get_all() if d.name and d.constituency}
        return self._constituencies

    def _payload(self, speech, block, block_index, chunk_index, text):
        # People named within the speech, resolved at extraction time (deputies plus
        # non-deputies — ministers, the King, regional presidents, foreign leaders).
        # Speech-level, so the same lists ride on every passage-point. `mentions` is the
        # filterable list of person ids (a deputy's id is its old deputy slug, so deputy
        # filters are unchanged); `mention_types` maps id→kind for faceting;
        # `mention_counts` (unused for now) is kept so a future relevance boost by
        # mention frequency needs no second full re-index.
        resolved = [m for m in (speech.mentions or []) if m.person_id]
        mentions = [m.person_id for m in resolved]
        mention_types = {m.person_id: m.person_type for m in resolved}
        mention_counts = {m.person_id: m.count for m in resolved}
        payload = {
            "speech_id": speech.id,
            "session_id": speech.session_id,
            "references": speech.references,
            "legislature": speech.legislature,
            "group": speech.group,
            "speaker": speech.speaker,
            "speaker_surname": speech.speaker_surname,
            "role": speech.role,
            "order": speech.order,
            "date": speech.date,
            "session_name": speech.session_name,
            "mentions": mentions,
            "mention_types": mention_types,
            "mention_counts": mention_counts,
            "lang": block.lang,
            "original": block.original,
            "block_index": block_index,
            "chunk_index": chunk_index,
            "text": text,
        }
        # Key absent (not null) for non-deputy speakers, so a constituency filter
        # never matches them and faceting sees only real provinces.
        constituency = self._constituency_map().get(speech.speaker)
        if constituency:
            payload["constituency"] = constituency
        return payload
