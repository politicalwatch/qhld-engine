"""Unit tests for MentionTagger wiring — stubbed NER, no spaCy, no Mongo.

Verifies the tagger runs NER over the Spanish block only and resolves the spans
against its (once-built) deputy index.
"""

from types import SimpleNamespace

import pytest

from tipi_data.models.speech import SpeechText

from qhld_engine.application.speeches.mention_tagging import MentionTagger, es_text

pytestmark = pytest.mark.unit


class FakeDeputy:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


class RecordingNer:
    """Stub NerPort: records the text it was asked to parse, returns fixed spans."""

    def __init__(self, spans):
        self.spans = spans
        self.seen = None

    def person_spans(self, text):
        self.seen = text
        return self.spans


SETTINGS = SimpleNamespace(mention_match_threshold=90)
DEPUTIES = [FakeDeputy("d1", "Rufián Romero, Gabriel")]


def test_es_text_joins_only_spanish_blocks():
    blocks = [
        SpeechText(lang="ca", text="text en català", original=True),
        SpeechText(lang="es", text="texto en castellano", original=False),
    ]
    assert es_text(blocks) == "texto en castellano"


def test_es_text_empty_when_no_spanish_block():
    blocks = [SpeechText(lang="eu", text="euskaraz", original=True)]
    assert es_text(blocks) == ""


def test_tag_runs_ner_and_resolves():
    ner = RecordingNer(["el señor Rufián"])
    tagger = MentionTagger(DEPUTIES, ner=ner, settings=SETTINGS,
                           curated=[], nondeputy_speakers=[])
    mentions = tagger.tag("El señor Rufián intervino.")
    assert ner.seen == "El señor Rufián intervino."
    assert [m.name for m in mentions] == ["Rufián Romero, Gabriel"]


def test_tag_speech_uses_spanish_block():
    ner = RecordingNer(["Rufián"])
    tagger = MentionTagger(DEPUTIES, ner=ner, settings=SETTINGS,
                           curated=[], nondeputy_speakers=[])
    speech = SimpleNamespace(speech=[
        SpeechText(lang="ca", text="original en català", original=True),
        SpeechText(lang="es", text="traducción con Rufián", original=False),
    ])
    mentions = tagger.tag_speech(speech)
    assert ner.seen == "traducción con Rufián"
    assert [m.name for m in mentions] == ["Rufián Romero, Gabriel"]


def test_tag_strips_stenographer_annotations_before_ner():
    ner = RecordingNer([])
    tagger = MentionTagger(DEPUTIES, ner=ner, settings=SETTINGS,
                           curated=[], nondeputy_speakers=[])
    tagger.tag("Eso lo saben. (El señor Rufián: ¡No es verdad!) Y lo repito.")
    assert ner.seen == "Eso lo saben. Y lo repito."


def test_tag_interruptions_resolves_interrupter_and_their_quote_mentions():
    deputies = DEPUTIES + [FakeDeputy("d2", "Tellado Filgueira, Miguel")]
    ner = RecordingNer(["Rufián"])  # spans found inside the interruption quote
    tagger = MentionTagger(deputies, ner=ner, settings=SETTINGS,
                           curated=[], nondeputy_speakers=[])
    interruptions = tagger.tag_interruptions(
        "Señorías… (Rumores.―El señor Tellado Filgueira: Rufián lo dijo)")
    assert ner.seen == "Rufián lo dijo"  # NER ran over the quote, not the speech
    assert [i.name for i in interruptions] == ["Tellado Filgueira, Miguel"]
    assert interruptions[0].quotes == ["Rufián lo dijo"]
    assert [m.name for m in interruptions[0].mentions] == ["Rufián Romero, Gabriel"]


def test_tag_interruptions_skips_the_speaker_themself():
    ner = RecordingNer([])
    tagger = MentionTagger(DEPUTIES, ner=ner, settings=SETTINGS,
                           curated=[], nondeputy_speakers=[])
    interruptions = tagger.tag_interruptions(
        "Gracias. (Aplausos.―Risas del señor Rufián Romero)",
        speaker="Rufián Romero, Gabriel")
    assert interruptions == []


def test_tag_resolves_non_deputy_from_curated_and_bootstrap():
    # A deputy, a curated non-deputy (Ayuso) and a bootstrapped minister all resolve.
    ner = RecordingNer(["Rufián", "Ayuso", "Aagesen"])
    tagger = MentionTagger(
        DEPUTIES, ner=ner, settings=SETTINGS,
        curated=[{"person_id": "isabel-diaz-ayuso", "person_type": "regional_president",
                  "name": "Díaz Ayuso, Isabel", "aliases": ["Ayuso"]}],
        nondeputy_speakers=[{"speaker": "Aagesen Muñoz, Sara", "role": "Ministra"}])
    by_type = {m.person_type: m.name for m in tagger.tag("...")}
    assert by_type["deputy"] == "Rufián Romero, Gabriel"
    assert by_type["regional_president"] == "Díaz Ayuso, Isabel"
    assert by_type["minister"] == "Aagesen Muñoz, Sara"
