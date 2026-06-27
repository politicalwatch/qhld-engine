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


@lru_cache
def get_settings() -> Settings:
    return Settings()
