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
    def __init__(self, settings, gazetteer=None):
        self.settings = settings
        self._gazetteer = tuple(gazetteer or ())
        self._nlp = None

    def _model(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.settings.ner_model)
            if self._gazetteer:
                # Only gazetteer surnames the model has NO representation for (out of
                # vocabulary). In-vocabulary surfaces — common words the model won't tag
                # as a person ("Madrid", "Torres") and common surnames it already knows
                # — are left to its context-sensitive judgement; overriding them with a
                # blunt rule tags every occurrence and wrecks precision.
                terms = [t for t in self._gazetteer if self._nlp.vocab[t.lower()].is_oov]
                if terms:
                    # An entity ruler before the statistical NER; case-sensitive, so it
                    # matches the Title-case name and supplements (not overrides) the model.
                    ruler = self._nlp.add_pipe(
                        "entity_ruler", before="ner", config={"overwrite_ents": False})
                    ruler.add_patterns(
                        [{"label": "PER", "pattern": term} for term in terms])
        return self._nlp

    def person_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._model()(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PER"]


@_register("spacy")
def create(settings, gazetteer=None) -> SpacyNer:
    return SpacyNer(settings, gazetteer=gazetteer)
