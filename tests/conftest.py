"""Shared, repo-wide test configuration.

The only thing that must be set up for *every* test (unit and integration alike)
is the MongoDB connection env — ``tipi_data`` builds its ``MongoClient`` at import
time from the ``MONGO_*`` vars, so they have to be in place before the package is
imported anywhere (including the model-only unit tests). The values point at a
throwaway test database on a **fixed** host port; the integration tier
(``tests/integration/conftest.py``) starts the matching container.

``MONGO_SKIP_INDEX_INIT`` keeps the import-time index creation from connecting, so
unit tests import ``tipi_data`` offline with no MongoDB.

Port 47019 is used so this suite never fights another qhld repo for the host
port: qhld-data=47017, qhld-tasks=47018, qhld-engine=47019.

Tiers:
- ``tests/unit`` — no infrastructure; runs anywhere (``-m unit``).
- ``tests/integration`` — needs the throwaway Mongo; auto-skips without Docker
  (``-m integration``).
"""

import os

import pytest

os.environ.setdefault("MONGO_SKIP_INDEX_INIT", "1")
os.environ.setdefault("MONGO_HOST", "localhost")
os.environ.setdefault("MONGO_PORT", "47019")
os.environ.setdefault("MONGO_USER", "qhld")
os.environ.setdefault("MONGO_PASSWORD", "qhld")
os.environ.setdefault("MONGO_DB_NAME", "qhlddb_test")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear the ``get_settings`` lru_cache around every test.

    ``Settings`` is read once and memoised, so without this a test that uses
    ``monkeypatch.setenv`` would either see a stale cached value or leak its
    override into later tests. Clearing before and after keeps each test
    isolated from the others (and from whatever ``.env`` happens to contain).
    """
    from qhld_engine.infrastructure.config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
