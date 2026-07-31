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

    # Backend response cache (Redis DB 8). The engine only ever DELETEs from it,
    # to drop the deputies/groups entries it just made stale. Field names and
    # defaults mirror tipi_backend's Settings one-for-one — same env vars, same
    # values — so a single .env keeps both sides pointing at the same keys.
    cache_redis_host: str = "redis"
    cache_redis_port: int = 6379
    cache_redis_password: str = ""
    cache_redis_db: int = 8
    cache_deputies: str = "deputies"
    cache_deputies_compact: str = "deputies-compact"
    cache_groups: str = "parliamentary-groups"

    # All AI/retrieval configuration (LLM, embeddings, Qdrant, query parsing,
    # reranking, sparse/hybrid, NER/mentions) lives in qhld-ai's own Settings
    # (qhld_ai.infrastructure.config.settings); both classes read the same .env.


@lru_cache
def get_settings() -> Settings:
    return Settings()
