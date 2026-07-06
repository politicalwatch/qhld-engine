"""Tag a speech with the deputies it mentions (index-time NER → resolved names).

Composition seam between the NER adapter (``NerPort``) and the pure resolver
(``domain.speeches.mentions``). The deputy index is built once from the catalog
at construction and reused for every speech, so a whole extract/backfill run does
one ``Deputies.get_all()`` and loads the spaCy model once.

NER runs only over the Spanish text block: co-official speeches always carry a
Spanish translation alongside the original, so one Spanish model covers the whole
corpus and we never NER Basque/Galician/Catalan (where the model is weak).
"""

from qhld_engine.domain.speeches.mentions import (
    COMMON_WORD_SURNAMES,
    NON_DEPUTY_SURNAMES,
    build_deputy_index,
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
    def __init__(self, deputies, ner=None, settings=None):
        self.settings = settings or get_settings()
        self._ner = ner or create_ner_from_env(self.settings)
        self._index = build_deputy_index(deputies)
        self._threshold = self.settings.mention_match_threshold

    def tag(self, text: str):
        """Return the ``Mention``s named in ``text`` (already the Spanish block).

        Spans that name a flagged non-deputy are dropped: a fixed denylist of famous
        non-deputies (ex-PMs, autonomous-community presidents) plus common-word false
        friends, and any surname the speech's own wording marks as a non-deputy office
        holder (magistrate/judge/prosecutor/ex-president/Franco-the-dictator)."""
        spans = self._ner.person_spans(text)
        excluded = (
            NON_DEPUTY_SURNAMES | COMMON_WORD_SURNAMES
            | context_excluded_surnames(text))
        return resolve_mentions(spans, self._index, self._threshold, excluded)

    def tag_speech(self, speech):
        """Convenience: tag a ``Speech`` from its stored Spanish block(s)."""
        return self.tag(es_text(speech.speech))
