"""Write access to the backend's response cache (Redis DB 8).

The engine's only business here is *deleting* — dropping the entries whose data it
has just rewritten. It never reads or writes cache values; that stays entirely in
``tipi_backend.api.cache``.

The client is built on first use, so importing this module (or running ``qhld
--help``) never needs Redis to be reachable.
"""

import redis

from qhld_engine.infrastructure.config.settings import get_settings


_client = None


def get_client():
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.Redis(
            host=settings.cache_redis_host,
            port=settings.cache_redis_port,
            password=settings.cache_redis_password or None,
            db=settings.cache_redis_db,
        )
    return _client


def delete(*keys):
    """Delete the named keys. Returns how many existed. Deliberately targeted —
    DB 8 holds other caches, so never flush the database."""
    if not keys:
        return 0
    return get_client().delete(*keys)
