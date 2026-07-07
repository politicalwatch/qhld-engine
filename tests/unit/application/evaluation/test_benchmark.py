"""Offline unit tests for the RunBenchmark service (search service stubbed)."""

import json

import pytest

from qhld_engine.application.evaluation.benchmark import RunBenchmark
from qhld_engine.domain.ports.vector_store import SearchHit

pytestmark = pytest.mark.unit


class _FakeService:
    """Returns canned hits; records the filters it was called with."""

    def __init__(self):
        self.calls = []

    def search(self, query, k=10, filters=None):
        self.calls.append(filters)
        return [
            SearchHit(id="1", score=0.9, payload={"references": ["A/1"], "lang": "es"}),
            SearchHit(id="2", score=0.8, payload={"references": ["A/1"], "lang": "es"}),
            SearchHit(id="3", score=0.7, payload={"references": ["B/2"], "lang": "gl"}),
        ]


@pytest.fixture()
def queryset_file(tmp_path):
    entries = [
        {"id": "T1", "dimension": "topical", "query": "q", "filters": {}, "expected_refs": ["B/2"]},
        {"id": "X1", "dimension": "crosslingual", "query": "q", "filters": {}, "lang": "gl",
         "expected_refs": ["A/1"]},
    ]
    path = tmp_path / "queryset.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def test_run_scores_rows_with_ranked_refs(queryset_file, monkeypatch):
    runner = RunBenchmark(queryset_file)
    fake = _FakeService()
    monkeypatch.setattr(runner, "_service", lambda model, reranker: fake)

    rows = runner.run("some-model", reranker="none", k=10)

    topical = rows[0]
    assert topical["rank"] == 3                       # B/2 first appears at position 3
    assert topical["ranked_refs"] == ["A/1", "B/2"]   # passages de-duped to references
    assert topical["score"] == 0.7


def test_crosslingual_entry_runs_filtered_and_unfiltered(queryset_file, monkeypatch):
    runner = RunBenchmark(queryset_file)
    fake = _FakeService()
    monkeypatch.setattr(runner, "_service", lambda model, reranker: fake)

    rows = runner.run("some-model", reranker="none", k=10)

    cross = rows[1]
    assert cross["rank"] == 1                          # A/1 at position 1
    assert "nolang_rank" in cross and "nolang_score" in cross
    # the lang filter is applied on the first pass, dropped on the penalty pass
    assert fake.calls[-2] == {"lang": "gl"}
    assert fake.calls[-1] is None
