"""Tag a speech with the people it mentions (index-time NER → resolved names)
and the non-person entities it references (``tag_entities``).

Composition seam between the NER adapter (``NerPort``) and the pure resolver
(``domain.speeches.mentions``). The person index — deputies plus non-deputies
(curated figures + speakers bootstrapped from the corpus) — is built once at
construction and reused for every speech, so a whole extract/backfill run does one
catalog load and loads the spaCy model once.

NER runs only over the Spanish text block: co-official speeches always carry a
Spanish translation alongside the original, so one Spanish model covers the whole
corpus and we never NER Basque/Galician/Catalan (where the model is weak).

Stenographer annotations — the parenthesized stage directions of the Diario de
Sesiones — are stripped before mention NER (they are the transcript's voice, not
the speaker's: an interjection like "(El señor Tellado Filgueira: Ábalos…)" must
not credit the speaker with mentioning either name). The same annotations feed
``tag_interruptions``, which records who interjected instead.
"""

from qhld_ai.application.persons_catalog import (
    gazetteer_surfaces,
    load_deputy_profiles,
    load_person_index,
)
from qhld_ai.domain.annotations import (
    extract_annotations,
    parse_utterances,
    resolve_interruptions,
    strip_annotations,
)
from qhld_ai.domain.entities import aggregate_entities
from qhld_ai.domain.mentions import (
    COMMON_WORD_SURNAMES,
    build_office_surfaces,
    build_surname_gazetteer,
    context_excluded_surnames,
    resolve_mentions,
)
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.ner.factory import create_ner_from_env


def taggable_text(blocks) -> str:
    """The text of a speech to run NER over — one block of it, never a concatenation.

    Spanish is preferred — the models and the persons catalog are Spanish, and a
    co-official speech usually carries the Diario's Spanish interpretation. But not
    every speech has a Spanish block: one delivered wholly in a co-official language
    with no interpretation published has only its original. Falling back to that
    original tags it imperfectly rather than not at all, which is the difference
    between a speech that is merely harder to find and one that is invisible to every
    person and entity filter.

    A speech whose co-official passage the Diario also printed in Spanish has two
    blocks that are both ``es``, and each holds the whole speech. Reading both would
    put the same words through NER twice, counting every mention and interruption in
    them double. The published rendering is the one taken, being the block that is
    Spanish the whole way through."""
    spanish = [b for b in (blocks or []) if b.lang == "es" and b.text]
    if spanish:
        return next((b.text for b in spanish if not b.original), spanish[0].text)
    return next((b.text for b in (blocks or []) if b.original and b.text), "")


class MentionTagger:
    def __init__(self, deputies, ner=None, settings=None,
                 curated=None, nondeputy_speakers=None, deputy_profiles=None,
                 speaker_offices=None):
        self.settings = settings or get_settings()
        self._threshold = self.settings.mention_match_threshold
        if deputy_profiles is None:
            deputy_profiles = load_deputy_profiles()
        self._index = load_person_index(
            deputies, self._threshold,
            curated=curated, nondeputy_speakers=nondeputy_speakers,
            deputy_profiles=deputy_profiles, speaker_offices=speaker_offices)
        if ner is not None:
            self._ner = ner
        else:
            # Curated public names go in alongside the surnames: without a pattern the
            # model never spans them ("Tesh" is out of its vocabulary), so the alias
            # keys in the index would have nothing to resolve.
            gazetteer = (build_surname_gazetteer(
                deputies, extra=gazetteer_surfaces(deputy_profiles))
                if getattr(self.settings, "ner_gazetteer", False) else None)
            # Which surnames a role apposition may claim ("el ministro Cuerpo"). Built
            # from the assembled index, so it covers every tier that holds an office, and
            # switched on here rather than in the adapter: the adapter takes data, the
            # application decides policy, exactly as with the gazetteer above.
            offices = (build_office_surfaces(self._index)
                       if getattr(self.settings, "ner_role_apposition", True) else None)
            self._ner = create_ner_from_env(
                self.settings, gazetteer=gazetteer, office_surfaces=offices)

    def tag(self, text: str):
        """Return the ``Mention``s named in ``text`` (already the Spanish block),
        with stenographer annotations stripped first — only the speaker's own
        words are tagged.

        A span resolves to a deputy or a non-deputy in the person catalog. The
        exclusion set only guards DEPUTY resolutions — common-word false friends
        ("Bueno") and surnames the speech's own wording marks as a non-deputy office
        holder (magistrate/judge/prosecutor/Franco-the-dictator); a resolved
        non-deputy is never dropped.

        Three signals read the whole speech, which is why the spans are resolved as a batch
        here rather than one at a time: gendered courtesy forms are pooled, so one "la
        señora Muñoz" settles every bare "Muñoz"; a role apposition names an office, so one
        "el presidente Sánchez" settles every bare "Sánchez"; and a surname still tied
        afterwards is attached to the one tied person the speech names elsewhere in full.
        The role apposition is why the spoken text goes in alongside the spans — the role
        word sits outside the span the NER returns."""
        spoken = strip_annotations(text)
        spans = self._ner.person_spans(spoken)
        excluded = COMMON_WORD_SURNAMES | context_excluded_surnames(spoken)
        return resolve_mentions(
            spans, self._index, self._threshold, excluded,
            gender_gate=getattr(self.settings, "mention_gender_gate", True),
            gender_veto=getattr(self.settings, "mention_gender_veto", True),
            coreference=getattr(self.settings, "mention_speech_coreference", True),
            text=spoken,
            role_apposition=getattr(self.settings, "mention_role_apposition", True))

    def tag_entities(self, text: str):
        """Return the ``NamedEntity``s referenced in ``text`` (already the Spanish
        block): every non-person NER span, aggregated by canonical key. Same
        annotation-stripping as ``tag`` — the stenographer's stage directions are
        not the speaker's references.

        "Non-person" includes what the adapter's post-passes claim: a surname the model
        mislabelled but the gazetteer or a role apposition recognised is a person, so it no
        longer reaches the entity index ("El ministro Albares" came back as MISC).

        Call right after ``tag`` (before ``tag_interruptions``): both run over the
        identical stripped string, so the NER adapter's doc memo makes the pair
        cost a single spaCy parse."""
        spoken = strip_annotations(text)
        return aggregate_entities(self._ner.entity_spans(spoken))

    def tag_interruptions(self, text: str, speaker: str | None = None):
        """Return the ``Interruption``s recorded in ``text``'s stenographer
        annotations: who interjected while the speech was delivered, their quotes
        and reactions, and the people named inside the quotes (resolved with the
        same NER + catalog as mentions). ``speaker`` (the speech's own orator, as
        stored in ``Speech.speaker``) filters out annotations about the speaker
        themself, e.g. their speech-closing applause."""
        utterances = [
            utterance
            for annotation in extract_annotations(text)
            for utterance in parse_utterances(annotation)]
        if not utterances:
            return []
        excluded = (COMMON_WORD_SURNAMES
                    | context_excluded_surnames(strip_annotations(text)))
        return resolve_interruptions(
            utterances, self._index, self._threshold, self._ner.person_spans,
            excluded, speaker_name=speaker)

    def tag_speech(self, speech):
        """Convenience: tag a ``Speech`` from its stored text blocks."""
        return self.tag(taggable_text(speech.speech))
