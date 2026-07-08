from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Extraction
    module_extractor: str = "spain"
    id_legislatura: int = 0
    current_legislature: bool = True
    limit_date_to_sync: str = "2000-01-01"  # str: consumer parses with strptime()
    amendments_feature: bool = False
    # Initiative types whose debates the daily speech extraction sweeps
    # (JSON list, e.g. '["172", "173", "210", "162"]'). Empty means the sweep
    # does nothing: each environment opts in explicitly, since every listed
    # type costs one API probe per reference on every run.
    speech_extraction_types: list[str] = []

    # Alerts
    use_alerts: bool = False

    # Stats legislature window (empty-string sentinel is load-bearing)
    legislature_start_date: str = ""
    legislature_end_date: str = ""

    # Logging (field name matches env LOGLEVEL exactly)
    loglevel: str = "INFO"

    # Redis — declared for .env parity; not consumed by the engine today
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db_check: int = 0
    redis_db_denylist: int = 1

    # LLM — first hexagonal AI-feature scaffolding (parliamentary-speech analysis).
    # Field names mirror vinculante so the infrastructure/llm adapters work unchanged.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    mistral_api_key: str = ""
    ollama_base_url: str = "http://host.docker.internal:11434"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant vector store (semantic speech search). Host defaults to the
    # docker-compose service name; ":memory:" selects qdrant-client's in-process
    # mode (used by the Docker-free tests).
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    # Empty -> the index/search services derive a per-model collection name
    # (speeches__<provider>__<model>__<dim>); set to force a fixed name.
    qdrant_collection: str = ""

    # Speech chunking (passage granularity for embeddings). Char-budgeted rather
    # than token-based so it stays provider/tokenizer-agnostic.
    speech_chunk_chars: int = 1200
    speech_chunk_overlap: int = 150

    # Query understanding: parse a NL query into structured
    # filters + a residual semantic query. Decoupled from the main llm_* so the
    # parser can use a different model than any future answer-synthesis; empty
    # provider/model fall back to llm_provider/llm_model.
    query_parser_provider: str = "llm"
    query_parser_llm_provider: str = ""
    query_parser_llm_model: str = ""

    # Cross-encoder reranker. "noop" leaves bi-encoder order
    # untouched (the clean baseline); any other provider over-fetches
    # reranker_top_n passages and reorders them on (query, passage) relevance.
    reranker_provider: str = "noop"
    reranker_model: str = ""
    reranker_top_n: int = 50

    # Sparse lexical embeddings (hybrid retrieval). "none" keeps pure dense
    # retrieval — existing collections and search behavior untouched. When set
    # (e.g. "bm25"), indexing writes a second, lexical vector per passage and
    # search fuses the dense and lexical rankings, which keeps literal tokens
    # (names, road codes, law numbers) from being blurred by the dense embedding.
    sparse_provider: str = "none"
    sparse_model: str = "Qdrant/bm25"
    # Stemmer/stopword language for BM25 tokenization; it must match between
    # indexing and querying. The corpus is multilingual, but "spanish" covers
    # the dominant blocks and proper names are barely affected by stemming.
    sparse_language: str = "spanish"
    # Hybrid fusion tuning: candidates fetched per branch (dense / lexical)
    # before fusion, and the fusion algorithm ("rrf" or "dbsf").
    hybrid_prefetch_limit: int = 50
    hybrid_fusion: str = "rrf"

    # Mention extraction (index-time NER → resolved deputies on Speech.mentions).
    # PER spans are found by spaCy over the Spanish text block, then fuzzy-matched
    # against the deputies catalog; token_set_ratio scores subset matches high, so
    # a high threshold stays both forgiving (surname-only) and precise.
    ner_provider: str = "spacy"
    ner_model: str = "es_core_news_lg"
    # Seed the NER pipeline with a gazetteer of distinctive deputy surnames so it also
    # tags the uncommon/compound ones (Catalan/Basque names, hyphenated compounds) the
    # base model misses. Off => base model only.
    ner_gazetteer: bool = True
    mention_match_threshold: int = 90


@lru_cache
def get_settings() -> Settings:
    return Settings()
