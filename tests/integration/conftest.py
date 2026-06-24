"""Integration-tier fixtures.

These tests need a reachable MongoDB; a throwaway ``mongo`` container is started
automatically for the session via ``testcontainers`` and torn down at the end — no
manual setup and no risk of touching real data. When Docker is unavailable the
tests **skip** rather than fail, so a plain ``uv run pytest`` stays green anywhere.

The container is published on the **fixed** host port the ``tipi_data`` client was
built against in the root ``conftest.py`` (the client is built at import time, so
the port must be known up front; it is lazy, so the container can start later).
"""

import os

import pytest
from pymongo.errors import PyMongoError

from tipi_data import client, db, ensure_indexes

_HOST_PORT = int(os.environ["MONGO_PORT"])


@pytest.fixture(scope="session")
def _mongo_container():
    """A throwaway MongoDB for the test session, published on the fixed host port.
    Skips all dependent tests when Docker is unavailable. Torn down at session end."""
    from testcontainers.mongodb import MongoDbContainer

    try:
        # Construct inside the guard too: instantiating the container eagerly
        # creates a Docker client (fetching the server API version), which fails
        # at construction time when the daemon is down — not just at start().
        container = (
            MongoDbContainer("mongo:7.0", username="qhld", password="qhld")
            .with_bind_ports(27017, _HOST_PORT)
        )
        container.start()
    except Exception as exc:  # docker missing / daemon down / port in use
        pytest.skip(f"No Docker available for MongoDB integration tests: {exc}")
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def mongo_db(_mongo_container):
    """The test database, reset before and after each test. Drops all collections
    so each test starts clean, then creates the declared indexes."""
    # Guard: the drop loop below is destructive, so refuse to run against anything
    # that is not an explicit test database — never the prod-data ``qhlddb``.
    assert db.name.endswith("_test"), (
        f"refusing to run destructive tests against database {db.name!r}; "
        "the test database name must end in '_test'"
    )
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip("No MongoDB reachable for repository integration tests")

    for name in db.list_collection_names():
        db.drop_collection(name)
    ensure_indexes()
    yield db
    for name in db.list_collection_names():
        db.drop_collection(name)
