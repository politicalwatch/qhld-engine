"""Characterization test for the speech pipeline against REAL captured data.

Drives ``ExtractSpeeches`` end-to-end with the two network boundaries stubbed by
fixtures captured from initiative 172/000001 (see tests/fixtures/speeches/172_000001/
README.md): the intervention API JSON and the raw Diario de Sesiones text.

This locks in the current behavior over real input — a co-official-language
parliamentarian and a role-based government speaker — so any change to segmentation
shows up here. The per-speech lengths are golden values: update them deliberately
when the extraction logic intentionally changes.
"""

import json
import pathlib

import pytest

from qhld_engine.application.speeches import extract_speeches as mod

pytestmark = pytest.mark.unit

FIX = pathlib.Path(__file__).parents[2] / "fixtures" / "speeches" / "172_000001"


@pytest.fixture
def capture():
    return (
        json.loads((FIX / "interventions_page1.json").read_text(encoding="utf-8")),
        (FIX / "session_raw.txt").read_text(encoding="utf-8"),
        (FIX / "session_link.txt").read_text(encoding="utf-8"),
    )


def test_extract_speeches_172_000001(monkeypatch, capture):
    page1, raw_text, expected_link = capture

    class _FakeApi:
        def get_video(self, reference, page):
            assert reference == "172/000001"
            return type("R", (), {"json": staticmethod(lambda: page1)})()

    seen_links = []

    class _FakePDF:
        def __init__(self, link, format_output=True):
            seen_links.append(link)
            assert format_output is False  # speeches need the raw, un-split text

        def retrieve(self):
            return raw_text

    saved = []
    monkeypatch.setattr(mod, "CongressApi", lambda: _FakeApi())
    monkeypatch.setattr(mod, "PDFExtractor", _FakePDF)
    monkeypatch.setattr(mod.Speeches, "save", lambda speech: saved.append(speech))

    mod.ExtractSpeeches().execute(["172/000001"])

    # one session PDF fetched, for the expected link
    assert seen_links == [expected_link]

    # all four interventions persisted, in document order
    assert [s.order for s in saved] == [1, 2, 3, 4]
    by_order = {s.order: s for s in saved}

    # diputado (has a group) vs government member (no group)
    assert by_order[1].speaker == "Rego Candamil, Néstor"
    assert by_order[1].group == "GMx"
    assert by_order[1].role == "Diputado"
    assert by_order[2].speaker == "Saiz Delgado, Elma"
    assert by_order[2].group is None
    assert by_order[2].role == "Ministra de Inclusión, Seguridad Social y Migraciones"

    # languages differ: the diputado speaks Galician, the minister Spanish
    assert by_order[1].speech.startswith("Grazas, señora presidenta")
    assert by_order[2].speech.startswith("Muchas gracias, presidente")

    # boundaries are correct: the minister's reply is NOT swallowed into the
    # diputado's turn (the regression the broadened SPEAKER_PATTERN fixed)
    assert "Muchas gracias, presidente" not in by_order[1].speech

    # golden lengths — update deliberately if segmentation changes
    assert [len(by_order[i].speech) for i in (1, 2, 3, 4)] == [11020, 9596, 7573, 3688]

    # common fields
    assert all(s.reference == "172/000001" for s in saved)
    assert all(s.legislature == "15" for s in saved)
    assert all(s.session_link == expected_link for s in saved)
    assert all(s.id for s in saved)  # deterministic ids generated
