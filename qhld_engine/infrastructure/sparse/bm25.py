"""BM25 sparse embedder via fastembed (``Qdrant/bm25``).

Emits term-weight vectors: document weights carry BM25 term-frequency
saturation, query weights are plain term presence; the inverse-document-
frequency half of BM25 is applied server-side by the vector store (the
collection's sparse vectors are configured with an IDF modifier), so weights
here are corpus-independent. Tokenization uses the configured language's
stemmer and stopword list, which must be identical at index and query time.

The fastembed model is loaded lazily on first use — the first load downloads
the per-language stopword files from the Hugging Face hub — so importing this
module stays cheap for callers that don't use sparse retrieval.
"""

from qhld_engine.domain.ports.sparse_embeddings import SparseEmbedderPort
from qhld_engine.domain.ports.vector_store import SparseVector
from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register


class Bm25SparseEmbedder(SparseEmbedderPort):
    def __init__(self, model: str, language: str):
        self._model_name = model
        self._language = language
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                model_name=self._model_name, language=self._language)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._convert(embedding) for embedding in self.model.embed(texts)]

    def embed_query(self, text: str) -> SparseVector:
        return self._convert(next(iter(self.model.query_embed(text))))

    @staticmethod
    def _convert(embedding) -> SparseVector:
        # fastembed returns numpy arrays; the port carries plain ints/floats.
        return SparseVector(
            indices=[int(index) for index in embedding.indices],
            values=[float(value) for value in embedding.values],
        )


@_register("bm25")
def create(settings: Settings) -> Bm25SparseEmbedder:
    return Bm25SparseEmbedder(settings.sparse_model, settings.sparse_language)
