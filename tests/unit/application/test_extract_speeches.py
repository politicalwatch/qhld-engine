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
                      pdf_fetches=None, probe=None, probed=None):
    """Stub the API, PDF, persistence, NER and clip-probe dependencies. ``existing``
    maps speech id -> stored Speech for the reuse-on-re-extract path; ``tagged_texts``
    collects every text actually sent to the mention tagger; ``counts`` maps
    reference -> stored-speech count for the incremental skip; ``deleted``,
    ``pdf_fetches`` and ``probed`` collect deleted speech ids, fetched session links
    and probed video links; ``probe`` replaces the duration probe itself."""

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
    # stub the clip probe: it reads a header over the network, and these are offline.
    _probed = probed if probed is not None else []

    def _probe(link):
        _probed.append(link)
        return probe(link) if probe else 61.5

    monkeypatch.setattr(mod, "probe_duration", _probe)
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


# --- how long the intervention's clip runs ---------------------------------

def test_the_clip_length_is_stored(monkeypatch):
    saved, probed = [], []
    _stub_environment(monkeypatch, _page(), saved, [], probed=probed)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert saved[0].duration == 61.5
    assert probed == ["http://v/3.mp4"]


def test_an_unreachable_video_does_not_fail_the_extraction(monkeypatch):
    # The text extracted perfectly well; losing the speech because the Congress CDN
    # was slow would trade a whole intervention for one metadata field.
    def _unavailable(link):
        raise mod.DurationUnavailable(f"could not open {link}")

    saved = []
    _stub_environment(monkeypatch, _page(), saved, [], probe=_unavailable)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert len(saved) == 1
    assert saved[0].duration is None
    assert saved[0].speech  # the text is there regardless


def test_a_speech_with_no_video_is_not_probed(monkeypatch):
    page = _page()
    page["lista_intervenciones"]["k1"]["video_intervencion"] = {"legislatura": 15}
    saved, probed = [], []
    _stub_environment(monkeypatch, page, saved, [], probed=probed)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert probed == []
    assert saved[0].duration is None


def test_reextraction_reuses_a_stored_duration_instead_of_reprobing(monkeypatch):
    # Re-extracting the corpus must not re-read 4,000 headers off the CDN.
    from tipi_data.models.speech import Speech

    stored = Speech(id=generate_id("776209"), video_link="http://v/3.mp4",
                    duration=352.0)
    saved, probed = [], []
    _stub_environment(monkeypatch, _page(), saved, [],
                      existing={stored.id: stored}, probed=probed)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert probed == []
    assert saved[0].duration == 352.0


def test_a_new_video_link_is_reprobed(monkeypatch):
    # The provisional copy of an intervention carries no video; when one is
    # published the stored length (if any) describes a different clip.
    from tipi_data.models.speech import Speech

    stored = Speech(id=generate_id("776209"), video_link="http://v/old.mp4",
                    duration=999.0)
    saved, probed = [], []
    _stub_environment(monkeypatch, _page(), saved, [],
                      existing={stored.id: stored}, probed=probed)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert probed == ["http://v/3.mp4"]
    assert saved[0].duration == 61.5


# --- saving only some of a reference's speeches -----------------------------

def _multi_page(*interventions):
    """A page of several interventions, given as ``(orador, video_id)`` in the order
    they took the floor."""
    return {
        "intervenciones_encontradas": str(len(interventions)),
        "lista_intervenciones": {
            f"k{i}": {
                "orador": orador,
                "cargo_orador": "Diputado",
                "doc": str(i + 1),
                "video_intervencion": {
                    "legislatura": 15,
                    "id01": video_id,
                    "enlace_descarga02": f"http://v/{video_id}.mp4",
                },
                "pdia": "CONG-1#anchor",
                "fecha": 20240115,
                "sesion": {
                    "nombre_sesion": "Pleno",
                    "idsesion": "12",
                    "videos_fase": {"enlace_descarga": "http://v/full.mp4"},
                },
            }
            for i, (orador, video_id) in enumerate(interventions)
        },
    }


def _diario(monkeypatch, text):
    """Replace the stubbed Diario with a specific sitting's text."""
    class _PDF:
        def __init__(self, link, format_output=True):
            pass

        def retrieve(self):
            return text

    monkeypatch.setattr(mod, "PDFExtractor", _PDF)


def test_only_the_targeted_speech_is_saved(monkeypatch):
    page = _multi_page(("Perez, Juan (GP Socialista)", "776209"),
                       ("Garcia, Ana (GP Popular)", "776210"))
    saved = []
    _stub_environment(monkeypatch, page, saved, [])
    _diario(monkeypatch, "El señor PEREZ: Hola. La señora GARCIA: Adiós.")

    count = mod.ExtractSpeeches().execute(["161/000123"], only={"776210"})

    assert count == 1
    assert [s.video_id for s in saved] == ["776210"]
    assert [(b.lang, b.text) for b in saved[0].speech] == [("es", "Adiós.")]


def test_a_skipped_neighbour_still_places_the_speech_that_follows_it(monkeypatch):
    # The segmenter's cursor only advances inside next_speech, so a skipped
    # intervention must still be segmented. Were the loop narrowed instead of the
    # save, a speaker who takes the floor twice would be found at their FIRST
    # heading and stored under the second one's id.
    page = _multi_page(("Perez, Juan (GP Socialista)", "776209"),
                       ("Garcia, Ana (GP Popular)", "776210"),
                       ("Perez, Juan (GP Socialista)", "776211"))
    saved = []
    _stub_environment(monkeypatch, page, saved, [])
    _diario(monkeypatch,
            "El señor PEREZ: Primera. La señora GARCIA: Media. "
            "El señor PEREZ: Segunda.")

    mod.ExtractSpeeches().execute(["161/000123"], only={"776211"})

    assert [s.video_id for s in saved] == ["776211"]
    assert [(b.lang, b.text) for b in saved[0].speech] == [("es", "Segunda.")]


def test_a_skipped_speech_is_neither_deleted_nor_probed(monkeypatch):
    # What targeting buys is blast radius: no write and no CDN read for a
    # speech nobody asked for.
    page = _multi_page(("Perez, Juan (GP Socialista)", "776209"),
                       ("Garcia, Ana (GP Popular)", "776210"))
    saved, deleted, probed = [], [], []
    _stub_environment(monkeypatch, page, saved, [], deleted=deleted, probed=probed)
    _diario(monkeypatch, "El señor PEREZ: Hola. La señora GARCIA: Adiós.")

    mod.ExtractSpeeches().execute(["161/000123"], only={"776210"})

    assert probed == ["http://v/776210.mp4"]
    # one provisional-twin cleanup, the target's — the skipped speech is not touched
    assert len(deleted) == 1


def test_a_speech_with_no_video_id_cannot_be_targeted(monkeypatch):
    # Until the sitting's video is published the intervention has no id to key on.
    page = _multi_page(("Perez, Juan (GP Socialista)", "776209"))
    page["lista_intervenciones"]["k0"]["video_intervencion"] = {"legislatura": 15}
    saved = []
    _stub_environment(monkeypatch, page, saved, [])
    _diario(monkeypatch, "El señor PEREZ: Hola.")

    assert mod.ExtractSpeeches().execute(["161/000123"], only={"776209"}) == 0
    assert saved == []


def test_no_targeting_saves_every_speech_of_the_reference(monkeypatch):
    page = _multi_page(("Perez, Juan (GP Socialista)", "776209"),
                       ("Garcia, Ana (GP Popular)", "776210"))
    saved = []
    _stub_environment(monkeypatch, page, saved, [])
    _diario(monkeypatch, "El señor PEREZ: Hola. La señora GARCIA: Adiós.")

    assert mod.ExtractSpeeches().execute(["161/000123"]) == 2
    assert [s.video_id for s in saved] == ["776209", "776210"]


# --- how the shape was decided ---------------------------------------------

def _blocks(*texts):
    from tipi_data.models.speech import SpeechText
    return [SpeechText(lang="ca", text=t, original=i == 0)
            for i, t in enumerate(texts)]


def _split(undecided):
    from qhld_engine.domain.speeches.language_split import Block, Split
    return Split("ca", [Block("ca", "whatever", True)], undecided)


def test_a_speech_the_evidence_places_records_no_verdict():
    assert mod.ExtractSpeeches._verdict(_split(False), _blocks("text"), None) is None


def test_a_speech_nothing_places_is_recorded_as_undecided():
    verdict = mod.ExtractSpeeches._verdict(_split(True), _blocks("text"), None)

    assert verdict.method == "undecided"
    assert verdict.fingerprint


def test_an_acoustic_verdict_survives_a_re_extraction_of_the_same_text():
    # Establishing it cost a video download; nothing in extraction can reproduce it.
    from tipi_data.models.speech import Speech, SplitVerdict
    blocks = _blocks("una", "dues")
    settled = mod.ExtractSpeeches._verdict(_split(True), blocks, None)
    stored = Speech(id="s1", split_verdict=SplitVerdict(
        method="acoustic", fingerprint=settled.fingerprint))

    verdict = mod.ExtractSpeeches._verdict(_split(True), blocks, stored)

    assert verdict.method == "acoustic"


def test_an_acoustic_verdict_is_discarded_when_the_text_has_changed():
    # It describes a speech that no longer exists in that form, so trusting it would
    # carry a judgement about one text over onto another.
    from tipi_data.models.speech import Speech, SplitVerdict
    stored = Speech(id="s1", split_verdict=SplitVerdict(
        method="acoustic", fingerprint="of some older text"))

    verdict = mod.ExtractSpeeches._verdict(_split(True), _blocks("una", "dues"), stored)

    assert verdict.method == "undecided"


# --- one person, one stored speaker string ---------------------------------

def _orador_page(orador, video_id="776209"):
    page = _page(video_id=video_id)
    page["lista_intervenciones"]["k1"]["orador"] = orador
    return page


def _curate(monkeypatch, records):
    monkeypatch.setattr(mod, "load_deputy_profiles", lambda: records)


def _collect_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        mod, "log",
        type("L", (), {"warning": staticmethod(lambda msg: warnings.append(msg)),
                       "info": staticmethod(lambda msg: None)})())
    return warnings


def test_a_curated_variant_is_stored_under_the_canonical_name(monkeypatch):
    # The source credits this deputy two ways; only one spelling matches the catalog.
    # Both must land on the catalog one, or her speeches split across two filter values.
    saved = []
    _stub_environment(monkeypatch, _orador_page("Ogou Corbi, Viviane (GSUMAR)"), saved, [])
    _curate(monkeypatch, [{"deputy_id": "ogou-i-corbi-viviane",
                           "name": "Ogou i Corbi, Viviane",
                           "speaker_variants": ["Ogou Corbi, Viviane"]}])

    mod.ExtractSpeeches().execute(["161/000123"])

    assert saved[0].speaker == "Ogou i Corbi, Viviane"
    # the surname follows the canonical spelling, not the one the source printed
    assert saved[0].speaker_surname == "Ogou i Corbi"


def test_an_uncurated_speaker_is_stored_exactly_as_the_source_spelled_it(monkeypatch):
    saved = []
    _stub_environment(monkeypatch, _page(), saved, [])
    _curate(monkeypatch, [])

    mod.ExtractSpeeches().execute(["161/000123"])

    assert saved[0].speaker == "Perez, Juan"
    assert saved[0].speaker_surname == "Perez"


def test_renaming_a_speaker_does_not_move_the_speech_id(monkeypatch):
    # The content id keys on the RAW orador, so curating a variant must not re-file
    # every one of that person's existing speeches under a new document.
    saved = []
    _stub_environment(monkeypatch, _orador_page("Ogou Corbi, Viviane (GSUMAR)",
                                                video_id=None), saved, [])
    _curate(monkeypatch, [{"deputy_id": "ogou-i-corbi-viviane",
                           "name": "Ogou i Corbi, Viviane",
                           "speaker_variants": ["Ogou Corbi, Viviane"]}])

    mod.ExtractSpeeches().execute(["161/000123"])

    assert saved[0].speaker == "Ogou i Corbi, Viviane"
    assert saved[0].id == generate_id(
        "/public_oficiales/L15/CONG-1", "Ogou Corbi, Viviane (GSUMAR)", "3",
        "||".join(block.text for block in saved[0].speech))


def test_an_uncurated_second_spelling_is_warned_about(monkeypatch):
    saved = []
    _stub_environment(monkeypatch, _orador_page("Ogou Corbi, Viviane (GSUMAR)"),
                      saved, [])
    _curate(monkeypatch, [])
    monkeypatch.setattr(mod.Deputies, "get_all", staticmethod(
        lambda: [type("D", (), {"name": "Ogou i Corbi, Viviane"})()]))
    warnings = _collect_warnings(monkeypatch)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert any("second spelling of 'Ogou i Corbi, Viviane'" in w for w in warnings)


def test_a_genuine_non_deputy_speaker_is_not_warned_about(monkeypatch):
    # Ministers and witnesses are legitimately absent from the deputy catalog, so
    # absence alone must not warn — only a name that NESTS with a catalog one.
    saved = []
    _stub_environment(monkeypatch, _orador_page("Cuerpo Caballero, Carlos"), saved, [])
    _curate(monkeypatch, [])
    monkeypatch.setattr(mod.Deputies, "get_all", staticmethod(
        lambda: [type("D", (), {"name": "Sierra Caballero, Francisco"})()]))
    warnings = _collect_warnings(monkeypatch)

    mod.ExtractSpeeches().execute(["161/000123"])

    assert not any("second spelling" in w for w in warnings)


# --- discovery by date ------------------------------------------------------
#
# The daily sweep enumerates interventions by date instead of walking every
# initiative reference. Extraction is deliberately untouched: the rows are grouped
# back by reference and handed to the same code path, so these tests are about
# grouping, filtering and the completeness comparison, not about segmentation.


def _dated_row(video_id, reference, doc="1"):
    """One row as the interventions search returns it for a date range.

    The reference arrives with a trailing sequence the stored reference does not
    carry, which is why ``_reference_of`` drops it."""
    row = _page(video_id)["lista_intervenciones"]["k1"]
    row = dict(row)
    row["doc"] = doc
    row["iniciativa"] = {
        "enlace_expediente": {"id_iniciativa": f"{reference}/0000"}
    }
    return row


def _date_page(rows):
    return {
        "intervenciones_encontradas": str(len(rows)),
        "lista_intervenciones": {f"k{n}": r for n, r in enumerate(rows)},
    }


def ExtractSpeeches_reference_of(row):
    return mod.ExtractSpeeches._reference_of(row)


def _stub_by_date(monkeypatch, page, saved, saved_sessions, types,
                  counts=None, pdf_fetches=None):
    from types import SimpleNamespace

    _stub_environment(monkeypatch, page, saved, saved_sessions,
                      counts=counts, pdf_fetches=pdf_fetches)

    class _FakeApi:
        """Serves both queries, because discovery and extraction use different ones.

        ``doc`` is deliberately different between them: the date listing numbers rows
        across the whole range, the per-reference query numbers them within the
        reference. That is what the source does, and the stored order must come from
        the second."""

        def get_interventions_by_date(self, since, until, page_number):
            return _FakeResponse(page)

        def get_video(self, reference, page_number):
            rows = [
                dict(r, doc=str(n))
                for n, r in enumerate(
                    (page.get("lista_intervenciones") or {}).values(), start=1)
                if ExtractSpeeches_reference_of(r) == reference
            ]
            return _FakeResponse(_date_page(rows))

    monkeypatch.setattr(mod, "CongressApi", lambda: _FakeApi())
    monkeypatch.setattr(
        mod, "get_settings",
        lambda: SimpleNamespace(id_legislatura=15, speech_extraction_types=types))


def test_by_date_groups_rows_under_the_reference_they_carry(monkeypatch):
    saved, sessions = [], []
    page = _date_page([_dated_row("776209", "161/000123")])
    _stub_by_date(monkeypatch, page, saved, sessions, ["161"])

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    assert len(saved) == 1
    # the trailing /0000 of the listed expediente is not part of the stored reference
    assert saved[0].references == ["161/000123"]


def test_by_date_ignores_types_that_are_not_configured(monkeypatch):
    saved, sessions, pdfs = [], [], []
    page = _date_page([_dated_row("776209", "184/000001")])
    _stub_by_date(monkeypatch, page, saved, sessions, ["161"], pdf_fetches=pdfs)

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    # one date query serves every type, so the scope is applied to the rows; a
    # reference outside it must not even reach the PDF
    assert saved == []
    assert pdfs == []


def test_by_date_skips_a_reference_whose_speeches_are_all_stored(monkeypatch):
    saved, sessions, pdfs = [], [], []
    page = _date_page([_dated_row("776209", "161/000123")])
    _stub_by_date(monkeypatch, page, saved, sessions, ["161"],
                  counts={"161/000123": 1}, pdf_fetches=pdfs)

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    assert saved == []
    assert pdfs == []


def test_by_date_excludes_vote_entries(monkeypatch):
    saved, sessions = [], []
    vote = _dated_row("776300", "161/000123", doc="2")
    vote["tipo_intervencion"] = "Votación"
    page = _date_page([_dated_row("776209", "161/000123"), vote])
    _stub_by_date(monkeypatch, page, saved, sessions, ["161"],
                  counts={"161/000123": 1})

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    # 2 rows listed, 1 of them a vote: the one stored speech means complete
    assert saved == []


def test_by_date_does_not_recount_an_intervention_listed_twice(monkeypatch):
    """The source lists some interventions more than once under the same reference.

    Measured over the legislature: 5 references of 6052, the worst reporting 25 rows
    for 13 interventions. Counted naively the stored count can never catch up, so the
    reference would be re-extracted on every run for ever."""
    saved, sessions, pdfs = [], [], []
    page = _date_page([
        _dated_row("776209", "161/000123", doc="1"),
        _dated_row("776209", "161/000123", doc="1"),
    ])
    _stub_by_date(monkeypatch, page, saved, sessions, ["161"],
                  counts={"161/000123": 1}, pdf_fetches=pdfs)

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    assert saved == []
    assert pdfs == []


def test_by_date_does_nothing_when_no_types_are_configured(monkeypatch):
    saved, sessions, pdfs = [], [], []
    page = _date_page([_dated_row("776209", "161/000123")])
    _stub_by_date(monkeypatch, page, saved, sessions, [], pdf_fetches=pdfs)

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    assert saved == []
    assert pdfs == []


def test_by_date_stores_the_order_of_the_reference_not_of_the_date_listing(monkeypatch):
    """``doc`` is a position in the result set, not a property of the intervention.

    The same speech is doc 1 of its reference and doc 134 of its sitting's date. It
    is what becomes the stored order, so extraction has to read it from the
    reference's own query or the corpus ends up with orders that depend on which
    path happened to extract the speech."""
    saved, sessions = [], []
    row = _dated_row("776209", "161/000123", doc="134")
    _stub_by_date(monkeypatch, _date_page([row]), saved, sessions, ["161"])

    mod.ExtractSpeeches().execute_by_date("01/01/2024", "31/01/2024")

    assert len(saved) == 1
    assert saved[0].order == 1
