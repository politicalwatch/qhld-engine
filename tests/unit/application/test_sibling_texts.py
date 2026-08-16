"""Unit tests for pairing a co-official passage with its Spanish interpretation,
and for the backfill that writes the pairing onto an already-indexed corpus.
"""

import pytest

from qhld_engine.application.speeches.backfill_siblings import BackfillSiblings
from qhld_engine.application.speeches.sibling_texts import (
    SIBLING_KEY, attach_siblings, siblings_for)
from qhld_ai.domain.ports.vector_store import VectorPoint
from qhld_ai.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        **overrides,
    )


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.0, 0.0, 0.0]  # dim 3, only probed for the collection name


def _point(pid, lang, text, vector):
    return VectorPoint(id=pid, vector=vector, payload={"lang": lang, "text": text})


# --- the pairing ------------------------------------------------------------


def test_each_co_official_passage_gets_its_closest_spanish_passages():
    points = [
        _point("eu0", "eu", "telelana", [1.0, 0.0]),
        _point("eu1", "eu", "osasuna", [0.0, 1.0]),
        _point("es0", "es", "teletrabajo", [1.0, 0.1]),
        _point("es1", "es", "salud", [0.1, 1.0]),
    ]

    attach_siblings(points)

    assert points[0].payload[SIBLING_KEY] == ["teletrabajo", "salud"]
    assert points[1].payload[SIBLING_KEY] == ["salud", "teletrabajo"]


def test_spanish_passages_get_no_sibling():
    """They are already in the query's language — there is nothing to compensate."""
    points = [_point("eu0", "eu", "telelana", [1.0, 0.0]),
              _point("es0", "es", "teletrabajo", [1.0, 0.1])]

    attach_siblings(points)

    assert SIBLING_KEY not in points[1].payload


def test_a_monolingual_speech_gets_no_siblings_at_all():
    """No Spanish block, so no candidates — the key must simply be absent rather
    than empty, and search falls back to scoring the passage alone."""
    points = [_point("es0", "es", "hola", [1.0, 0.0]),
              _point("es1", "es", "adiós", [0.0, 1.0])]

    attach_siblings(points)

    assert all(SIBLING_KEY not in point.payload for point in points)


def test_a_co_official_block_without_a_spanish_one_gets_nothing():
    points = [_point("eu0", "eu", "telelana", [1.0, 0.0])]

    attach_siblings(points)

    assert SIBLING_KEY not in points[0].payload


def test_siblings_for_is_parallel_to_its_input():
    items = [({"lang": "ca", "text": "a"}, [1.0, 0.0]),
             ({"lang": "es", "text": "b"}, [1.0, 0.0])]

    assert siblings_for(items) == [["b"], []]


# --- the backfill -----------------------------------------------------------


class _FakeStore:
    def __init__(self, speeches):
        self.speeches = speeches          # {speech_id: [VectorPoint]}
        self.written = {}
        self.distinct_where = None

    def distinct_values(self, name, key, where=None):
        self.distinct_where = where
        return set(self.speeches)

    def points_by(self, name, key, value):
        return self.speeches[value]

    def set_payload(self, name, payloads):
        self.written.update(payloads)


def _backfill(store):
    return BackfillSiblings(
        settings=_settings(), embedder=_FakeEmbedder(), store=store)


def test_backfill_writes_the_pairing_of_every_co_official_passage():
    store = _FakeStore({"A": [
        _point("eu0", "eu", "telelana", [1.0, 0.0]),
        _point("es0", "es", "teletrabajo", [1.0, 0.1]),
    ]})

    written, skipped = _backfill(store).execute()

    assert store.written == {"eu0": {SIBLING_KEY: ["teletrabajo"]}}
    assert (written, skipped) == (1, 0)


def test_backfill_only_looks_at_co_official_speeches():
    """A monolingual Spanish speech has nothing to pair, so it must not even be
    fetched — that is what keeps this cheap on a corpus that is mostly Spanish."""
    store = _FakeStore({"A": [_point("es0", "es", "hola", [1.0, 0.0])]})

    _backfill(store).execute()

    assert store.distinct_where == {"lang": ["ca", "gl", "eu"]}


def test_backfill_skips_a_speech_with_no_spanish_block():
    store = _FakeStore({"A": [_point("eu0", "eu", "telelana", [1.0, 0.0])]})

    written, skipped = _backfill(store).execute()

    assert store.written == {} and (written, skipped) == (0, 1)


def test_backfill_dry_run_writes_nothing_but_reports_the_same_count():
    store = _FakeStore({"A": [
        _point("eu0", "eu", "telelana", [1.0, 0.0]),
        _point("es0", "es", "teletrabajo", [1.0, 0.1]),
    ]})

    written, _ = _backfill(store).execute(dry_run=True)

    assert store.written == {} and written == 1
