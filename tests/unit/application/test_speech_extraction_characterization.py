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
import re

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
    # Stub the clip probe: it reads a header off the Congress CDN, and this test is
    # offline. The lengths come from the capture's own `inicio01`/`fin01`, so each
    # intervention gets the clip it really had (750s, 559s, 303s, 208s) — the split
    # weighs text against clip length, and a made-up duration would quietly send every
    # speech down the "nothing places this" path and prove nothing.
    starts_ends = {
        (video["id01"]): video
        for entry in page1["lista_intervenciones"].values()
        for video in [entry.get("video_intervencion") or {}]
        if video.get("id01")
    }

    def _seconds(clock):
        hours, minutes, seconds = (int(part) for part in clock.split(":"))
        return hours * 3600 + minutes * 60 + seconds

    def _probe(link):
        probed.append(link)
        for video_id, video in starts_ends.items():
            if video_id in link:
                return float(_seconds(video["fin01"]) - _seconds(video["inicio01"]))
        raise AssertionError(f"no capture for {link}")

    probed = []
    monkeypatch.setattr(mod, "probe_duration", _probe)

    # Stub the paragraph similarity: it is the third network boundary, and the one this
    # test would otherwise reach every run. The stub has to be PLAUSIBLE, not merely
    # present — a similarity that returned zero would make the classifier refuse every
    # speech and the test would go green by exercising the fallback, which is precisely
    # how the stubbed clip duration once hid the whole feature. Galician and Spanish
    # share a great deal of vocabulary, so "several long words in common" tracks a real
    # rendering closely enough on this fixture, while staying offline and deterministic.
    def _similar(source, candidate):
        words = lambda t: {w for w in re.findall(r"[\wáéíóúñüàèòç]{6,}", t.lower())}
        shared = words(source) & words(candidate)
        return 0.95 if len(shared) >= 4 else 0.05

    monkeypatch.setattr(mod, "create_paragraph_similarity", lambda: _similar)

    # Record what the splitter was handed, so the blocks can be checked against the
    # document they came from rather than only against each other.
    split_inputs = []
    _split = mod.split_languages

    def _spy_split(text, *args, **kwargs):
        split_inputs.append(text)
        return _split(text, *args, **kwargs)

    monkeypatch.setattr(mod, "split_languages", _spy_split)
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

    # every intervention carries the length of its own clip, probed once each
    assert [s.duration for s in saved] == [750.0, 559.0, 303.0, 208.0]
    assert probed == [s.video_link for s in saved]

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
    assert rego[1].text.rstrip().endswith("Muchas gracias.")
    # Not "Muchas gracias.", and the reason is in the source rather than the split: a
    # page turn has merged the tail of the Galician and the tail of its interpretation
    # into ONE paragraph ("… como a aplicación destes coeficientes redutores. Moito
    # obrigado. … como la aplicación de estos coeficientes reductores."). A paragraph
    # that is genuinely two languages has to be assigned to one of them — Galician here,
    # since most of its characters are — and the closing line after it follows. 138
    # characters land on the wrong side; the alternative readings misplace as much.
    assert "Moito obrigado. … como la aplicación" in rego[0].text
    # Both salutations read as SPANISH to the detector — "señora presidenta" outweighs
    # the one Galician word in "Grazas, señora presidenta." However short a paragraph
    # gets no run of its own, and that same weak reading only picks which neighbour it
    # joins: the Galician one has no Spanish before it to join and stays with the
    # Galician, while "Gracias, señora presidenta." opens the interpretation and goes
    # with it.
    assert "Grazas, señora presidenta" not in rego[1].text
    assert "Gracias, señora presidenta.\n\nAntes que yo" in rego[1].text
    # It no longer *opens* the Spanish block, because the block now opens with a Galician
    # paragraph the interpretation has no counterpart for — see the shared-paragraph
    # golden below.
    assert rego[1].text.startswith("No Real decreto 1299/2006")

    minister = by_order[2].speech
    assert by_order[2].original_language == "es"
    assert [(b.lang, b.original) for b in minister] == [("es", True)]
    assert minister[0].text.startswith("Muchas gracias, presidente")

    # the second diputado turn is also bilingual; the second minister turn Spanish
    assert [(b.lang, b.original) for b in by_order[3].speech] == [("gl", True), ("es", False)]
    assert [(b.lang, b.original) for b in by_order[4].speech] == [("es", True)]

    # The minister's *reply* — a separate intervention — is not swallowed into the turn.
    # (The Galician block does end on Spanish, but only through the merged page-turn
    # paragraph described above, not through the boundary drifting.)
    assert "He escuchado con mucha atención, señor Rego" not in rego[0].text
    assert "He escuchado con mucha atención, señor Rego" not in rego[1].text

    # the Diario's paragraph structure survives as "\n\n" breaks in every block:
    # the salutation is its own paragraph, breaks land only at sentence ends,
    # and a page turn falling mid-sentence is joined, not broken
    assert rego[0].text.startswith("Grazas, señora presidenta.\n\nAntes ca min")
    assert "Gracias, señora presidenta." not in rego[0].text
    assert "Só hai que ver, só hai que ollar" in rego[0].text  # page turn at Pág. 41
    for speech in saved:
        for block in speech.speech:
            assert "\n\n" in block.text
            assert not any(
                p and p[-1] not in ".!?…»:\"”)" for p in block.text.split("\n\n"))

    # Every block holds the whole speech: between them the blocks account for every
    # paragraph of the document, because each is the whole text minus what the other one
    # covers. Nothing else in the suite checks this, and it is the property the reader and
    # the subtitle tracks both depend on.
    #
    # Paragraph for paragraph, and so only for a speech the evidence placed. The
    # provisional reading a refused speech falls back to cuts on a SENTENCE boundary, so it
    # can leave half a paragraph on each side — 729026 is the case, and it loses no words,
    # only the paragraph's integrity. All four here are decided (asserted below).
    assert len(split_inputs) == len(saved)
    for source, speech in zip(split_inputs, saved):
        paragraphs = {p.strip() for p in source.strip().split("\n\n") if p.strip()}
        for block in speech.speech:
            paragraphs -= {p.strip() for p in block.text.split("\n\n")}
        assert not paragraphs, (speech.video_id, [p[:70] for p in paragraphs])

    # Which paragraphs sit in BOTH blocks, and why. A paragraph appears twice exactly when
    # nothing on the other side renders it, so this golden is a direct read-out of the
    # pairing: 726572's interpretation is complete and shares nothing, while 726566 shares
    # three — the merged page-turn paragraph described above, the closing line that follows
    # it, and the decree citation, which the SIMILARITY STUB fails to pair. The real
    # embedder does pair that one (measured), so this third entry is the stub's coarseness
    # showing through, and it is worth leaving visible: under the mirror a pairing failure
    # duplicates a paragraph instead of silently dropping it out of the Spanish block.
    shared = {
        speech.video_id: len(set(speech.speech[0].text.split("\n\n"))
                             & set(speech.speech[1].text.split("\n\n")))
        for speech in saved if len(speech.speech) == 2}
    assert shared == {"726566": 3, "726572": 0}

    # golden per-block lengths and paragraph counts — update deliberately if the
    # logic changes
    block_shapes = {
        i: [(len(b.text), len(b.text.split("\n\n"))) for b in by_order[i].speech]
        for i in (1, 2, 3, 4)}
    assert block_shapes == {
        1: [(10566, 17), (11282, 12)],
        2: [(9608, 13)],
        3: [(3737, 4), (3842, 5)],  # unchanged from the single-boundary split
        4: [(3694, 7)],
    }

    # every one of them decided from the text and the clip; none needed the audio, and
    # none was left provisional
    assert [s.split_verdict for s in saved] == [None, None, None, None]
    assert all(not b.partial for s in saved for b in s.speech)

    # common fields
    assert all(s.references == ["172/000001"] for s in saved)
    assert all(s.legislature == "15" for s in saved)
    assert all(s.session_link == expected_link for s in saved)
    # identity = the Congress intervention id captured in the fixture
    assert [s.video_id for s in saved] == ["726566", "726567", "726572", "726573"]
    assert all(s.id for s in saved)  # deterministic ids generated
