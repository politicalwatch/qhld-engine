"""Unit test for the ExtractSpeeches application service — the API, PDF and
persistence dependencies are stubbed, so this exercises the orchestration and the
intervention-to-``Speech`` mapping without HTTP or a database.
"""

import pytest

from qhld_engine.application.speeches import extract_speeches as mod

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_execute_segments_and_saves_a_speech(monkeypatch):
    page = {
        "intervenciones_encontradas": "1",
        "lista_intervenciones": {
            "k1": {
                "orador": "Perez, Juan (GP Socialista)",
                "cargo_orador": "Diputado",
                "doc": "3",
                "video_intervencion": {
                    "legislatura": 15,
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

    class _FakeApi:
        def get_video(self, reference, page_number):
            return _FakeResponse(page)

    class _FakePDF:
        def __init__(self, link, format_output=True):
            pass

        def retrieve(self):
            return "El señor PEREZ: Hola. La señora GARCIA: Adiós."

    saved = []
    saved_sessions = []
    monkeypatch.setattr(mod, "CongressApi", lambda: _FakeApi())
    monkeypatch.setattr(mod, "PDFExtractor", _FakePDF)
    monkeypatch.setattr(mod.Speeches, "save", lambda speech: saved.append(speech))
    monkeypatch.setattr(mod.Sessions, "save", lambda s: saved_sessions.append(s))
    # stub mention tagging: no Mongo (deputy catalog) and no spaCy load here.
    monkeypatch.setattr(mod.Deputies, "get_all", staticmethod(lambda: []))
    monkeypatch.setattr(
        mod, "MentionTagger",
        lambda deputies: type("T", (), {"tag": staticmethod(lambda text: [])})())
    # patch the detector so the test never loads py3langid and is deterministic
    monkeypatch.setattr(mod, "detect", lambda text: "es")

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
    assert speech.reference == "161/000123"
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
    assert speech.id  # deterministic id was generated
