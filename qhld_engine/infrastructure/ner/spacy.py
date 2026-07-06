"""spaCy NER adapter for mention extraction.

Loads ``es_core_news_lg`` (the same model the rule-based query parser uses) and
returns every PER span. spaCy is lazy-imported and the model lazy-loaded on first
use, so importing this module (and factory self-registration) stays cheap; the
model is loaded once per adapter instance and reused across a whole extract/
backfill run (``MentionTagger`` holds a single instance).

We run NER only over the Spanish text block upstream, so a single Spanish model
covers monolingual and co-official speeches alike.
"""

from qhld_engine.domain.ports.ner import NerPort

from .factory import _register


class SpacyNer(NerPort):
    def __init__(self, settings):
        self.settings = settings
        self._nlp = None

    def _model(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.settings.ner_model)
        return self._nlp

    def person_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._model()(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PER"]


@_register("spacy")
def create(settings) -> SpacyNer:
    return SpacyNer(settings)
