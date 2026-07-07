"""Tag a speech with the people it mentions (index-time NER → resolved names).

Composition seam between the NER adapter (``NerPort``) and the pure resolver
(``domain.speeches.mentions``). The person index — deputies plus non-deputies
(curated figures + speakers bootstrapped from the corpus) — is built once at
construction and reused for every speech, so a whole extract/backfill run does one
catalog load and loads the spaCy model once.

NER runs only over the Spanish text block: co-official speeches always carry a
Spanish translation alongside the original, so one Spanish model covers the whole
corpus and we never NER Basque/Galician/Catalan (where the model is weak).
"""

from qhld_engine.application.speeches.persons_catalog import load_person_index
from qhld_engine.domain.speeches.mentions import (
    COMMON_WORD_SURNAMES,
    build_surname_gazetteer,
    context_excluded_surnames,
    resolve_mentions,
)
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_engine.infrastructure.ner.factory import create_ner_from_env


def es_text(blocks) -> str:
    """Join the Spanish (``lang == 'es'``) blocks of a speech. Monolingual Spanish
    speeches have one such block; co-official speeches have the translation."""
    return " ".join(b.text for b in (blocks or []) if b.lang == "es" and b.text)


class MentionTagger:
    def __init__(self, deputies, ner=None, settings=None,
                 curated=None, nondeputy_speakers=None):
        self.settings = settings or get_settings()
        self._threshold = self.settings.mention_match_threshold
        self._index = load_person_index(
            deputies, self._threshold,
            curated=curated, nondeputy_speakers=nondeputy_speakers)
        if ner is not None:
            self._ner = ner
        else:
            gazetteer = (build_surname_gazetteer(deputies)
                         if getattr(self.settings, "ner_gazetteer", False) else None)
            self._ner = create_ner_from_env(self.settings, gazetteer=gazetteer)

    def tag(self, text: str):
        """Return the ``Mention``s named in ``text`` (already the Spanish block).

        A span resolves to a deputy or a non-deputy in the person catalog. The
        exclusion set only guards DEPUTY resolutions — common-word false friends
        ("Bueno") and surnames the speech's own wording marks as a non-deputy office
        holder (magistrate/judge/prosecutor/Franco-the-dictator); a resolved
        non-deputy is never dropped."""
        spans = self._ner.person_spans(text)
        excluded = COMMON_WORD_SURNAMES | context_excluded_surnames(text)
        return resolve_mentions(spans, self._index, self._threshold, excluded)

    def tag_speech(self, speech):
        """Convenience: tag a ``Speech`` from its stored Spanish block(s)."""
        return self.tag(es_text(speech.speech))
