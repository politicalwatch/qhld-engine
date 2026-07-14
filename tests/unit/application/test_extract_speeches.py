"""Unit test for the ExtractSpeeches application service — the API, PDF and
persistence dependencies are stubbed, so this exercises the orchestration and the
intervention-to-``Speech`` mapping without HTTP or a database.
"""

import pytest

from tipi_data import DoesNotExist
from tipi_data.utils import generate_id

from qhld_engine.application.speeches import extract_speeches as mod

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _page(video_id="776209"):
    return {
        "intervenciones_encontradas": "1",
        "lista_intervenciones": {
            "k1": {
                "orador": "Perez, Juan (GP Socialista)",
                "cargo_orador": "Diputado",
                "doc": "3",
                "video_intervencion": {
                    "legislatura": 15,
                    "id01": video_id,
                    "enlace_descarga02": "http://v/3.mp4",
                },
                "pdia": "CONG-1#anchor",
                "fecha": 20240115,
                "sesion": {
                    "nombre_sesion": "Pleno",
                    "idsesion": "12",
                    "videos_fase": {"enlace_descarga": "http://v/full.mp4"},
                },
            }
        },
    }


def _stub_environment(monkeypatch, page, saved, saved_sessions, existing=None,
                      tagged_texts=None, counts=None, deleted=None,
                      pdf_fetches=None):
    """Stub the API, PDF, persistence and NER dependencies. ``existing`` maps
    speech id -> stored Speech for the reuse-on-re-extract path; ``tagged_texts``
    collects every text actually sent to the mention tagger; ``counts`` maps
    reference -> stored-speech count for the incremental skip; ``deleted`` and
    ``pdf_fetches`` collect deleted speech ids and fetched session links."""

    class _FakeApi:
        def get_video(self, reference, page_number):
            return _FakeResponse(page)

    class _FakePDF:
        def __init__(self, link, format_output=True):
            if pdf_fetches is not None:
                pdf_fetches.append(link)

        def retrieve(self):
            return "El señor PEREZ: Hola. La señora GARCIA: Adiós."

    def _get(speech_id):
        if existing and speech_id in existing:
            return existing[speech_id]
        raise DoesNotExist(f"Speech {speech_id} does not exist")

    monkeypatch.setattr(mod, "CongressApi", lambda: _FakeApi())
    monkeypatch.setattr(mod, "PDFExtractor", _FakePDF)
    monkeypatch.setattr(mod.Speeches, "save", lambda speech: saved.append(speech))
    monkeypatch.setattr(mod.Speeches, "get", staticmethod(_get))
    monkeypatch.setattr(mod.Speeches, "count_by_reference",
                        staticmethod(lambda ref: (counts or {}).get(ref, 0)))
    _deleted = deleted if deleted is not None else []
    monkeypatch.setattr(mod.Speeches, "delete",
                        staticmethod(lambda id: _deleted.append(id)))
    monkeypatch.setattr(mod.Sessions, "save", lambda s: saved_sessions.append(s))
    # stub mention tagging: no Mongo (deputy catalog) and no spaCy load here.
    monkeypatch.setattr(mod.Deputies, "get_all", staticmethod(lambda: []))
    collector = tagged_texts if tagged_texts is not None else []
    monkeypatch.setattr(
        mod, "MentionTagger",
        lambda deputies: type(
            "T", (), {
                "tag": staticmethod(lambda text: collector.append(text) or []),
                "tag_entities": staticmethod(lambda text: []),
                "tag_interruptions":
                    staticmethod(lambda text, speaker=None: []),
            })())
    # patch the detector so the test never loads py3langid and is deterministic
    monkeypatch.setattr(mod, "detect", lambda text: "es")


def test_execute_segments_and_saves_a_speech(monkeypatch):
    saved = []
    saved_sessions = []
    _stub_environment(monkeypatch, _page(), saved, saved_sessions)

    mod.ExtractSpeeches().execute(["161/000123"])

    # the sitting that hosts the debate is upserted with API metadata + a roster
    assert len(saved_sessions) == 1
    session = saved_sessions[0]
    assert session.session_link == "/public_oficiales/L15/CONG-1"
    assert session.name == "Pleno"
    assert session.code == "CONG-1"  # filename stem of the session link
    assert session.congress_session_id == "12"
    assert session.legislature == "15"
    assert session.date == 20240115
    assert session.video_link == "http://v/full.mp4"
    assert session.references == ["161/000123"]

    assert len(saved) == 1
    speech = saved[0]
    assert speech.references == ["161/000123"]
    # identity = the Congress intervention id, stable across the initiatives
    # of an accumulated debate
    assert speech.id == generate_id("776209")
    assert speech.video_id == "776209"
    # the speech links to its sitting via the session document's id
    assert speech.session_id == session.id
    assert speech.speaker == "Perez, Juan"
    assert speech.speaker_surname == "Perez"
    assert speech.group == "GP Socialista"
    assert speech.role == "Diputado"
    assert speech.order == 3
    assert speech.legislature == "15"
    assert speech.date == 20240115
    assert speech.session_name == "Pleno"
    assert speech.video_link == "http://v/3.mp4"
    assert speech.session_link == "/public_oficiales/L15/CONG-1"  # anchor stripped
    # a monolingual Spanish speech is stored as a single original block
    assert [(b.lang, b.text, b.original) for b in speech.speech] == [
        ("es", "Hola.", True)
    ]
    assert speech.original_language == "es"


def test_accumulated_debate_yields_one_id_across_references(monkeypatch):
    saved = []
    _stub_environment(monkeypatch, _page(), saved, [])

    # the same physical intervention extracted under both initiatives of an
    # accumulated debate
    mod.ExtractSpeeches().execute(["210/000151", "210/000152"])

    assert len(saved) == 2
    assert saved[0].id == saved[1].id  # same doc upserted, roster accumulates
    assert saved[0].references == ["210/000151"]
    assert saved[1].references == ["210/000152"]


def test_missing_video_id_falls_back_to_content_identity(monkeypatch):
    saved = []
    _stub_environment(monkeypatch, _page(video_id=""), saved, [])

    mod.ExtractSpeeches().execute(["161/000123"])

    speech = saved[0]
    assert speech.video_id is None
    assert speech.id == generate_id(
        "/public_oficiales/L15/CONG-1", "Perez, Juan (GP Socialista)", "3", "Hola.")


def test_incremental_skips_reference_whose_speeches_are_all_stored(monkeypatch):
    saved, pdf_fetches = [], []
    _stub_environment(monkeypatch, _page(), saved, [],
                      counts={"161/000123": 1}, pdf_fetches=pdf_fetches)

    mod.ExtractSpeeches().execute_incremental(["161/000123"])

    # 1 intervention in the API, 1 speech stored -> nothing to do; in
    # particular the session PDF is never downloaded.
    assert saved == []
    assert pdf_fetches == []


def test_incremental_extracts_reference_with_missing_speeches(monkeypatch):
    saved = []
    _stub_environment(monkeypatch, _page(), saved, [], counts={"161/000123": 0})

    mod.ExtractSpeeches().execute_incremental(["161/000123"])

    assert len(saved) == 1
    assert saved[0].references == ["161/000123"]


def test_incremental_skips_undebated_reference(monkeypatch):
    saved, pdf_fetches = [], []
    # the interventions API returns nothing for a reference not (yet) debated
    _stub_environment(monkeypatch, {}, saved, [], pdf_fetches=pdf_fetches)

    mod.ExtractSpeeches().execute_incremental(["161/000124"])

    assert saved == []
    assert pdf_fetches == []


def test_missing_video_intervencion_uses_configured_legislature(monkeypatch):
    from types import SimpleNamespace

    saved, saved_sessions, deleted = [], [], []
    page = _page()
    # the video of a just-finished session is not published yet: the API item
    # carries no video_intervencion at all
    del page["lista_intervenciones"]["k1"]["video_intervencion"]
    _stub_environment(monkeypatch, page, saved, saved_sessions, deleted=deleted)
    monkeypatch.setattr(mod, "get_settings",
                        lambda: SimpleNamespace(id_legislatura=15))

    mod.ExtractSpeeches().execute(["161/000123"])

    speech = saved[0]
    assert speech.session_link == "/public_oficiales/L15/CONG-1"
    assert speech.legislature == "15"
    assert speech.video_id is None
    assert speech.video_link is None
    assert speech.id == generate_id(
        "/public_oficiales/L15/CONG-1", "Perez, Juan (GP Socialista)", "3", "Hola.")
    assert saved_sessions[0].legislature == "15"
    assert deleted == []  # nothing provisional to clean up without a video id


def test_video_id_arrival_deletes_provisional_twin(monkeypatch):
    saved, deleted = [], []
    _stub_environment(monkeypatch, _page(), saved, [], deleted=deleted)

    mod.ExtractSpeeches().execute(["161/000123"])

    # once the Congress intervention id exists, the doc that this intervention
    # would have been stored under while the video was unpublished is removed
    assert saved[0].id == generate_id("776209")
    assert deleted == [generate_id(
        "/public_oficiales/L15/CONG-1", "Perez, Juan (GP Socialista)", "3", "Hola.")]


def test_reextraction_with_same_text_reuses_stored_mentions(monkeypatch):
    from tipi_data.models.speech import Mention, NamedEntity, Speech, SpeechText

    stored = Speech(
        id=generate_id("776209"),
        references=["210/000151"],
        speech=[SpeechText(lang="es", text="Hola.", original=True)],
        mentions=[Mention(person_id="garcia-ana", name="Garcia, Ana", count=1)],
        entities=[NamedEntity(key="eurovision", surface_forms=["Eurovisión"], count=1)],
    )
    saved = []
    tagged_texts = []
    _stub_environment(monkeypatch, _page(), saved, [],
                      existing={stored.id: stored}, tagged_texts=tagged_texts)

    mod.ExtractSpeeches().execute(["210/000152"])

    assert tagged_texts == []  # NER skipped: same intervention, unchanged text
    assert saved[0].mentions == stored.mentions
    assert saved[0].entities == stored.entities
