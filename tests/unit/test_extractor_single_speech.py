"""Unit tests for targeting individual speeches from ``ExtractorTask`` — no DB.

``ExtractorTask.__init__`` imports a country's extractor modules, which these paths
never touch, so the instance is built without it.
"""

import pytest

from tipi_data import DoesNotExist
from tipi_data.models.speech import Speech

from qhld_engine.extractors.extractor import ExtractorTask

pytestmark = pytest.mark.unit


@pytest.fixture
def task(monkeypatch):
    """An ExtractorTask whose speech extraction is recorded instead of run."""
    instance = object.__new__(ExtractorTask)
    calls = []
    monkeypatch.setattr(
        ExtractorTask, "_extract_speeches",
        lambda self, references, only=None: calls.append((references, only)) or 0)
    instance.calls = calls
    return instance


def _stub_lookup(monkeypatch, speeches):
    """Stub the repository the reference resolution reads, keyed by video id."""
    from tipi_data.repositories import speeches as repo

    def _get(video_id):
        if video_id not in speeches:
            raise DoesNotExist(f"Speech with video_id {video_id} does not exist")
        return speeches[video_id]

    monkeypatch.setattr(repo.Speeches, "get_by_video_id", staticmethod(_get))


def test_a_reference_alone_extracts_all_of_it(task):
    task.single_speeches("161/000123")

    assert task.calls == [(["161/000123"], None)]


def test_video_ids_narrow_the_named_reference(task):
    task.single_speeches("161/000123", ["776209", "776210"])

    assert task.calls == [(["161/000123"], {"776209", "776210"})]


def test_a_video_id_finds_its_own_reference(task, monkeypatch):
    _stub_lookup(monkeypatch, {
        "776209": Speech(id="a", video_id="776209", references=["161/000123"]),
    })

    task.single_speeches(None, ["776209"])

    assert task.calls == [(["161/000123"], {"776209"})]


def test_ids_sharing_a_reference_are_extracted_in_one_pass(task, monkeypatch):
    """Segmenting a sitting is the expensive part and it is per reference, so two
    speeches from the same debate must not pay for it twice."""
    _stub_lookup(monkeypatch, {
        "776209": Speech(id="a", video_id="776209", references=["161/000123"]),
        "776210": Speech(id="b", video_id="776210", references=["161/000123"]),
        "776300": Speech(id="c", video_id="776300", references=["161/000999"]),
    })

    task.single_speeches(None, ["776209", "776210", "776300"])

    assert sorted(task.calls, key=lambda c: c[0]) == [
        (["161/000123"], {"776209", "776210"}),
        (["161/000999"], {"776300"}),
    ]


def test_an_accumulated_debate_uses_the_first_of_its_references(task, monkeypatch):
    """13% of speeches carry several references. They are the initiatives of one joint
    debate, whose expediente numbers the Diario prints as consecutive headings, so
    every one of them segments to the same text — measured byte-identical over four
    sittings. Any of them will do; the argument is there to override the choice."""
    _stub_lookup(monkeypatch, {
        "726583": Speech(id="a", video_id="726583",
                         references=["210/000002", "210/000006", "210/000010"]),
    })

    task.single_speeches(None, ["726583"])

    assert task.calls == [(["210/000002"], {"726583"})]


def test_an_unknown_video_id_is_rejected_rather_than_guessed(task, monkeypatch):
    _stub_lookup(monkeypatch, {})

    with pytest.raises(ValueError, match="No stored speech for video id 776209"):
        task.single_speeches(None, ["776209"])

    assert task.calls == []


def test_neither_a_reference_nor_a_video_id_is_an_error(task):
    with pytest.raises(ValueError):
        task.single_speeches()
