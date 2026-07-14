"""Characterization test for the speech pipeline against REAL captured data.

Drives ``ExtractSpeeches`` end-to-end with the two network boundaries stubbed by
fixtures captured from initiative 172/000001 (see tests/fixtures/speeches/172_000001/
README.md): the intervention API JSON and the raw Diario de Sesiones text.

This locks in the current behavior over real input — a co-official-language
parliamentarian and a role-based government speaker — so any change to segmentation
or the language split shows up here. It drives the *real* py3langid detector (no
stub), validating the Galician/Spanish separation on real data. The per-block
lengths are golden values: update them deliberately when the logic intentionally
changes.
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
    saved_sessions = []
    monkeypatch.setattr(mod, "CongressApi", lambda: _FakeApi())
    monkeypatch.setattr(mod, "PDFExtractor", _FakePDF)
    monkeypatch.setattr(mod.Speeches, "save", lambda speech: saved.append(speech))
    def _no_stored_speech(speech_id):
        raise mod.DoesNotExist(speech_id)

    monkeypatch.setattr(mod.Speeches, "get", staticmethod(_no_stored_speech))
    # published video ids trigger the provisional-twin cleanup; keep it Mongo-free
    monkeypatch.setattr(mod.Speeches, "delete", staticmethod(lambda id: None))
    monkeypatch.setattr(mod.Sessions, "save", lambda s: saved_sessions.append(s))
    # stub mention tagging: this test locks segmentation/language-split, not NER,
    # and must stay Mongo-free (no deputy catalog) and spaCy-free.
    monkeypatch.setattr(mod.Deputies, "get_all", staticmethod(lambda: []))
    monkeypatch.setattr(
        mod, "MentionTagger",
        lambda deputies: type("T", (), {
            "tag": staticmethod(lambda text: []),
            "tag_entities": staticmethod(lambda text: []),
            "tag_interruptions": staticmethod(lambda text, speaker=None: []),
        })())

    mod.ExtractSpeeches().execute(["172/000001"])

    # one session PDF fetched, for the expected link
    assert seen_links == [expected_link]

    # the sitting is upserted once, carrying the full-session video + a roster, and
    # every speech of the sitting links back to it via the session document id
    assert len(saved_sessions) == 1
    session = saved_sessions[0]
    assert session.name == "Pleno"
    assert session.code == "DSCD-15-PL-13"
    assert session.references == ["172/000001"]
    assert session.video_link  # full-session video captured from videos_fase
    assert all(s.session_id == session.id for s in saved)

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

    # The diputado's turn is published bilingual and is split into two blocks: the
    # full Galician original (as delivered) followed by its full Spanish
    # interpretation. The minister speaks only Spanish → a single original block.
    rego = by_order[1].speech
    assert by_order[1].original_language == "gl"
    assert [(b.lang, b.original) for b in rego] == [("gl", True), ("es", False)]
    assert rego[0].text.startswith("Grazas, señora presidenta")
    assert rego[1].text.startswith("Gracias, señora presidenta")
    assert rego[1].text.rstrip().endswith("Muchas gracias.")

    minister = by_order[2].speech
    assert by_order[2].original_language == "es"
    assert [(b.lang, b.original) for b in minister] == [("es", True)]
    assert minister[0].text.startswith("Muchas gracias, presidente")

    # the second diputado turn is also bilingual; the second minister turn Spanish
    assert [(b.lang, b.original) for b in by_order[3].speech] == [("gl", True), ("es", False)]
    assert [(b.lang, b.original) for b in by_order[4].speech] == [("es", True)]

    # the Galician original carries no Spanish bleed: the boundary is clean and the
    # minister's *reply* (a separate intervention) is not swallowed into the turn.
    assert "Muchas gracias" not in rego[0].text
    assert "He escuchado con mucha atención, señor Rego" not in rego[0].text
    assert "He escuchado con mucha atención, señor Rego" not in rego[1].text

    # the Diario's paragraph structure survives as "\n\n" breaks in every block:
    # the salutation is its own paragraph, breaks land only at sentence ends,
    # and a page turn falling mid-sentence is joined, not broken
    assert rego[0].text.startswith("Grazas, señora presidenta.\n\nAntes ca min")
    assert rego[1].text.startswith("Gracias, señora presidenta.\n\nAntes que yo")
    assert "Só hai que ver, só hai que ollar" in rego[0].text  # page turn at Pág. 41
    for speech in saved:
        for block in speech.speech:
            assert "\n\n" in block.text
            assert not any(
                p and p[-1] not in ".!?…»:\"”)" for p in block.text.split("\n\n"))

    # golden per-block lengths and paragraph counts — update deliberately if the
    # logic changes
    block_shapes = {
        i: [(len(b.text), len(b.text.split("\n\n"))) for b in by_order[i].speech]
        for i in (1, 2, 3, 4)}
    assert block_shapes == {
        1: [(10424, 15), (10850, 11)],
        2: [(9608, 13)],
        3: [(3737, 4), (3842, 5)],
        4: [(3694, 7)],
    }

    # common fields
    assert all(s.references == ["172/000001"] for s in saved)
    assert all(s.legislature == "15" for s in saved)
    assert all(s.session_link == expected_link for s in saved)
    # identity = the Congress intervention id captured in the fixture
    assert [s.video_id for s in saved] == ["726566", "726567", "726572", "726573"]
    assert all(s.id for s in saved)  # deterministic ids generated
