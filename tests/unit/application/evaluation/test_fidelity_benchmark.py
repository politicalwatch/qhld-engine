"""Offline unit tests for RunFidelityBenchmark (Qdrant client stubbed).

The searching itself is deliberately not asserted behaviourally: a beam width or
a rescore flag has no observable effect against a stub, exactly as it has none
in-process against the real adapter. What these tests pin is the wiring that a
wrong result would hide — that ground truth is recomputed per cell, that the
parameters a cell asks for are the parameters that get sent, and that a sample
drawn from one end of the id space would be caught.
"""

import json

import pytest

from qhld_engine.application.evaluation.fidelity_benchmark import RunFidelityBenchmark

pytestmark = pytest.mark.unit


class _Point:
    def __init__(self, point_id, vector=None):
        self.id = point_id
        self.vector = vector


class _Response:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    """Returns a canned ranking per query, and records every search it was asked
    for so the tests can read back which parameters travelled."""

    def __init__(self, exact_ids=None, approx_ids=None, total=8):
        self.calls = []
        self.exact_ids = exact_ids or ["a", "b", "c", "d"]
        self.approx_ids = approx_ids or ["a", "b", "z", "d"]
        self.total = total

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        params = kwargs.get("search_params")
        prefetch = kwargs.get("prefetch")
        if prefetch:
            params = prefetch[0].params
        ids = self.exact_ids if getattr(params, "exact", False) else self.approx_ids
        return _Response([_Point(point_id) for point_id in ids])

    def scroll(self, **kwargs):
        return [_Point(index) for index in range(self.total)], None

    def retrieve(self, ids=None, **kwargs):
        return [_Point(i, {"dense": [float(i)]}) for i in ids]

    def count(self, **kwargs):
        self.calls.append(kwargs)
        return type("C", (), {"count": self.total})()

    def get_collection(self, name):
        vectors = {"dense": object()}
        return type("I", (), {
            "config": type("C", (), {
                "params": type("P", (), {"vectors": vectors})()})()})()


@pytest.fixture()
def asset_file(tmp_path):
    asset = {
        "sample": {"seed": 11, "size": 3},
        "fingerprints": {},
        "filters": [
            {"label": "group=GS", "must": [{"key": "group", "value": "GS"}]},
            {"label": "recent", "must": [{"key": "date", "range": {"gte": 20250101}}]},
        ],
        "cells": {"default": [{"hnsw_ef": 1024, "rescore": False}]},
        "reference_collections": {"float32": "other"},
    }
    path = tmp_path / "fidelity_queryset.json"
    path.write_text(json.dumps(asset), encoding="utf-8")
    return str(path)


def _runner(asset_file, client):
    runner = RunFidelityBenchmark(
        settings=type("S", (), {
            "qdrant_host": "x", "qdrant_port": 1, "qdrant_grpc_port": 2,
            "qdrant_prefer_grpc": False, "hybrid_prefetch_limit": 50,
            "hybrid_fusion": "rrf", "embedding_model": "m"})(),
        collection="coll", asset_path=asset_file)
    runner._client = client
    return runner


def test_run_cell_scores_the_arm_against_exact_search(asset_file):
    client = _FakeClient()
    runner = _runner(asset_file, client)
    row = runner.run_cell([[0.1]], {"hnsw_ef": 1024, "rescore": False}, k=4)
    # 3 of the exact top-4 were found; "z" is the intruder.
    assert row["mean"] == 0.75
    assert row["perfect"] == 0


def test_run_cell_asks_for_exact_ground_truth_every_time(asset_file):
    client = _FakeClient()
    runner = _runner(asset_file, client)
    runner.run_cell([[0.1], [0.2]], {"hnsw_ef": 8, "rescore": True}, k=4)
    # Two queries, each needing its own arm and its own ground truth: a cached
    # truth would compare later filters against the wrong reference.
    assert len(client.calls) == 4
    assert sum(1 for call in client.calls if call["search_params"].exact) == 2


def test_the_cell_parameters_are_the_ones_sent(asset_file):
    client = _FakeClient()
    runner = _runner(asset_file, client)
    runner.run_cell([[0.1]], {"hnsw_ef": 77, "rescore": True, "oversampling": 3.0}, k=4)
    sent = client.calls[0]["search_params"]
    assert sent.hnsw_ef == 77
    assert sent.quantization.rescore is True
    assert sent.quantization.oversampling == 3.0


def test_an_unset_rescore_sends_no_quantization_block(asset_file):
    # A collection with no compression has nothing to rescore; sending the block
    # anyway would make the reference arm untrustworthy.
    client = _FakeClient()
    runner = _runner(asset_file, client)
    runner.run_cell([[0.1]], {"hnsw_ef": 1024}, k=4)
    assert client.calls[0]["search_params"].quantization is None


def test_the_hybrid_shape_puts_params_and_filter_on_the_prefetch_branches(asset_file):
    client = _FakeClient()
    runner = _runner(asset_file, client)
    sparse = type("S", (), {"indices": [1], "values": [0.5]})()
    query_filter = runner.filter_cells()[1]["filter"]
    runner.run_cell([[0.1]], {"hnsw_ef": 9, "rescore": False}, k=4,
                    query_filter=query_filter, sparse_vectors=[sparse])
    call = client.calls[0]
    # Under fusion a top-level filter or params is ignored, so both have to
    # travel on the branches — and only the dense one takes params.
    assert call.get("query_filter") is None
    assert call["prefetch"][0].params.hnsw_ef == 9
    assert call["prefetch"][0].filter == query_filter
    assert call["prefetch"][1].params is None
    assert call["prefetch"][1].filter == query_filter


def test_sampling_is_seeded_and_fingerprinted(asset_file):
    client = _FakeClient(total=50)
    runner = _runner(asset_file, client)
    first_vectors, first_hash = runner.sample_corpus_vectors()
    second_vectors, second_hash = runner.sample_corpus_vectors()
    assert first_hash == second_hash
    assert first_vectors == second_vectors
    assert len(first_vectors) == 3


def test_a_moved_corpus_changes_the_fingerprint(asset_file):
    small = _runner(asset_file, _FakeClient(total=50))
    large = _runner(asset_file, _FakeClient(total=60))
    assert small.sample_corpus_vectors()[1] != large.sample_corpus_vectors()[1]


def test_the_sample_is_drawn_from_the_whole_id_space(asset_file):
    # The failure this guards: sampling one scroll page draws only the lowest
    # ids, which in an insertion-ordered corpus is one end of it, not a sample.
    runner = _runner(asset_file, _FakeClient(total=400))
    runner.asset["sample"]["size"] = 60
    vectors, _ = runner.sample_corpus_vectors()
    drawn = [vector[0] for vector in vectors]
    assert max(drawn) > 200


def test_a_sample_larger_than_the_corpus_takes_the_whole_corpus(asset_file):
    runner = _runner(asset_file, _FakeClient(total=2))
    vectors, _ = runner.sample_corpus_vectors()
    assert len(vectors) == 2


def test_filter_cells_start_from_the_unfiltered_baseline(asset_file):
    runner = _runner(asset_file, _FakeClient())
    cells = runner.filter_cells()
    assert cells[0]["label"] == "unfiltered"
    assert cells[0]["filter"] is None
    assert [cell["label"] for cell in cells[1:]] == ["group=GS", "recent"]


def test_a_range_filter_becomes_a_range_condition(asset_file):
    runner = _runner(asset_file, _FakeClient())
    recent = runner.filter_cells()[2]["filter"]
    assert recent.must[0].range.gte == 20250101
    assert recent.must[0].match is None


def test_real_query_texts_read_both_asset_shapes(asset_file):
    # The labelled set is a bare list; the parser set wraps rows under "queries".
    runner = _runner(asset_file, _FakeClient())
    texts = runner.real_query_texts()
    assert len(texts) == 94
    assert all(isinstance(text, str) and text for text in texts)


def test_the_query_cache_is_rejected_when_the_embedding_model_changed(tmp_path, asset_file):
    # Reading one model's vectors against another model's collection measures
    # nothing at all, so a stale cache has to be refused rather than reused.
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"model": "other", "vectors": [[1.0]]}), encoding="utf-8")
    runner = _runner(asset_file, _FakeClient())
    with pytest.raises(Exception):
        runner.real_query_vectors(cache_path=str(cache))


def test_the_query_cache_is_reused_for_the_same_model(tmp_path, asset_file):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"model": "m", "vectors": [[1.0]]}), encoding="utf-8")
    runner = _runner(asset_file, _FakeClient())
    assert runner.real_query_vectors(cache_path=str(cache)) == [[1.0]]
