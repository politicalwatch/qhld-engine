"""Application service: extract the *text* of parliamentary speeches for a set of
initiative references and persist them as ``Speech`` documents.

This is the orchestration + I/O layer. It drives the pure segmentation logic in
``domain.speeches.segmentation`` with data fetched from the Congress "intervenciones"
API (``CongressApi``) and the Diario de Sesiones PDF (``PDFExtractor``), and persists
through the qhld-data ``Speeches`` repository (the persistence port).

It is the speeches sibling of ``VideoExtractor``: both read the same per-initiative
intervention API, but this also downloads the session PDF and segments the speech
text per speaker. Only Spain currently has speeches, so the Congress infra is used
directly; a source port would only be introduced if a second country needed one.
"""

import json
import math
import os
from collections import OrderedDict

from tqdm import tqdm

from qhld_engine.logger import get_logger
from qhld_engine.application.speeches.mention_tagging import MentionTagger, es_text
from qhld_engine.domain.speeches import segmentation
from qhld_engine.domain.speeches.language_split import split_languages
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.language import detect
from qhld_engine.extractors.spain.congress_api import CongressApi
from qhld_engine.extractors.spain.initiative_extractors.utils.pdf_parsers import (
    PDFExtractor,
)

from tipi_data import DoesNotExist
from tipi_data.utils import generate_id
from tipi_data.models.session import Session
from tipi_data.models.speech import Speech, SpeechText
from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.sessions import Sessions
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)

INTERVENTIONS_PER_PAGE = 25
SESSION_PATH = "/public_oficiales/"


class ExtractSpeeches:

    def __init__(self):
        self.api = CongressApi()
        self._tagger = None

    @property
    def tagger(self):
        """Mention tagger, built once (loads the deputy catalog + spaCy model on
        first use). Lazy so importing/constructing the service stays Mongo-free."""
        if self._tagger is None:
            self._tagger = MentionTagger(Deputies.get_all())
        return self._tagger

    def execute(self, references):
        for reference in references:
            self._extract_reference(reference)

    def execute_incremental(self, references):
        """Extract only the references whose stored speeches are incomplete.

        Each reference costs one cheap interventions-API probe; the expensive
        PDF work only happens when the API lists more interventions than we
        have stored. A reference not yet debated (no interventions) or whose
        Diario PDF is not yet published (probe succeeds, extraction saves
        nothing) is simply retried on the next run — no state is kept, so any
        gap heals itself once the source publishes."""
        for reference in tqdm(references, desc="Checking speeches", unit="ref"):
            interventions = self._retrieve_all_interventions(reference)
            if not interventions:
                continue
            stored = Speeches.count_by_reference(reference)
            if stored >= len(interventions):
                continue
            log.info(
                f"{reference}: {stored}/{len(interventions)} speeches stored, extracting")
            self._process_interventions(reference, interventions)

    def _extract_reference(self, reference):
        log.info(f"Getting speeches from {reference}")
        interventions = self._retrieve_all_interventions(reference)
        if not interventions:
            return
        self._process_interventions(reference, interventions)

    def _process_interventions(self, reference, interventions):
        surnames = [
            segmentation.speaker_surname_upper(i["orador"]) for i in interventions
        ]

        for session_link, items in self._group_by_session(interventions).items():
            raw = PDFExtractor(session_link, format_output=False).retrieve()
            if not raw:
                log.warning(f"No session text for {reference} at {session_link}")
                continue
            session_id = generate_id(session_link)
            self._save_session(items[0], session_link, session_id, reference)
            speaker_regexes = [
                segmentation.build_speaker_regex(i["orador"]) for i in items
            ]
            text = segmentation.normalize_session_text(
                raw, reference, speaker_regexes)
            text = segmentation.fix_speaker_typos(text, surnames)
            segmenter = segmentation.SpeechSegmenter(text)
            for intervention, regex, upcoming in zip(
                    items, speaker_regexes, speaker_regexes[1:] + [None]):
                self._extract_one(
                    intervention, session_link, session_id, segmenter,
                    reference, regex, upcoming)

    def _save_session(self, intervention, session_link, session_id, reference):
        """Upsert the sitting that hosts this debate. Metadata is taken from any of
        the sitting's interventions (identical across them); ``references`` carries
        only this run's reference and is accumulated by the repository."""
        sesion = intervention.get("sesion", {})
        videos_fase = sesion.get("videos_fase", {})
        session = Session(
            id=session_id,
            legislature=self._legislature(intervention),
            session_link=session_link,
            name=sesion.get("nombre_sesion"),
            code=self._session_code(session_link),
            congress_session_id=sesion.get("idsesion"),
            date=intervention.get("fecha"),
            video_link=videos_fase.get("enlace_descarga"),
            references=[reference],
        )
        Sessions.save(session)

    def _session_code(self, session_link):
        """The canonical Diario document code = the PDF filename stem, e.g.
        ``/public_oficiales/L15/CONG/DS/PL/DSCD-15-PL-13.PDF`` -> ``DSCD-15-PL-13``."""
        return os.path.splitext(os.path.basename(session_link))[0]

    def _legislature(self, intervention):
        """Legislature of the intervention. Right after a session ends the API
        lists its interventions without ``video_intervencion`` (the video is
        published later), so fall back to the configured current legislature."""
        video = intervention.get("video_intervencion") or {}
        legislature = video.get("legislatura")
        if legislature:
            return str(legislature)
        return str(get_settings().id_legislatura)

    def _extract_one(self, intervention, session_link, session_id, segmenter,
                     reference, speaker_regex, upcoming_regex):
        speaker, group, surname = segmentation.parse_speaker(intervention["orador"])
        if speaker is None:
            log.warning(
                f"Unparseable speaker {intervention.get('orador')!r} for {reference}")
            return

        text = segmenter.next_speech(speaker_regex, upcoming_regex)
        if text is None:
            # The API sometimes credits the intervention to someone who never
            # took the floor (a Government reply is listed under the office
            # holder, not the minister who answered); the office heading is
            # then the only one printed. A failed search leaves the cursor
            # unmoved, so the retry is safe.
            role_regex = segmentation.build_role_regex(
                intervention.get("cargo_orador"))
            text = segmenter.next_speech(role_regex, upcoming_regex)
        if text is None:
            log.warning(f"Speaker heading not found for {reference}")
            original_language, blocks = None, []
        else:
            original_language, parts = split_languages(text, detect)
            blocks = [SpeechText(lang=lang, text=t, original=orig)
                      for lang, t, orig in parts]

        video = intervention.get("video_intervencion") or {}
        order = int(intervention["doc"])
        fallback_id = self._content_id(
            session_link, intervention["orador"], order, blocks)
        video_id = video.get("id01")
        speech_id = generate_id(video_id) if video_id else fallback_id
        if speech_id != fallback_id:
            # A run that happened before the video was published stored this
            # same intervention under its content identity; drop that copy now
            # that the canonical id is known.
            Speeches.delete(fallback_id)
        speech = Speech(
            id=speech_id,
            references=[reference],
            video_id=video.get("id01") or None,
            session_id=session_id,
            speaker=speaker,
            speaker_surname=surname,
            group=group,
            role=intervention.get("cargo_orador"),
            order=order,
            legislature=self._legislature(intervention),
            date=intervention.get("fecha"),
            session_name=intervention.get("sesion", {}).get("nombre_sesion"),
            video_link=video.get("enlace_descarga02"),
            session_link=session_link,
            speech=blocks,
            original_language=original_language,
            mentions=self._mentions(speech_id, blocks),
        )
        Speeches.save(speech)

    @staticmethod
    def _content_id(session_link, orador, order, blocks):
        """Fallback identity of the *physical* intervention, used while the
        Congress intervention id (``video_intervencion.id01``) does not exist
        yet — it stays empty until the sitting's video is published.

        The id keys on the intervention's observable coordinates plus its
        text: identical-text copies from the same speaker collapse (so an
        accumulated debate — several initiatives debated jointly — yields one
        document whose ``references`` roster accumulates), while a speaker's
        distinct speeches under the same document number (numbering restarts
        per initiative within a sitting) stay apart."""
        text = "||".join(block.text for block in blocks)
        return generate_id(session_link, orador, str(order), text)

    def _mentions(self, speech_id, blocks):
        """Run NER over the Spanish text — unless this intervention was already
        extracted with the same text (the earlier initiative of an accumulated
        debate), in which case its stored mentions are reused."""
        text = es_text(blocks)
        try:
            existing = Speeches.get(speech_id)
        except DoesNotExist:
            existing = None
        if existing is not None and es_text(existing.speech) == text:
            return existing.mentions
        return self.tagger.tag(text)

    # -- API retrieval ---------------------------------------------------------

    def _retrieve_all_interventions(self, reference):
        """All interventions for the initiative across every page, vote entries
        excluded, sorted by document order."""
        first = self._retrieve_json(reference, 1)
        if not first or "error" in first or "intervenciones_encontradas" not in first:
            return []

        total = int(first["intervenciones_encontradas"])
        pages = math.ceil(total / INTERVENTIONS_PER_PAGE)

        raw = dict(first.get("lista_intervenciones", {}))
        for page in range(2, pages + 1):
            page_json = self._retrieve_json(reference, page)
            if page_json:
                raw.update(page_json.get("lista_intervenciones", {}))

        interventions = [
            value for value in raw.values() if "tipo_intervencion" not in value
        ]
        return sorted(interventions, key=lambda v: int(v["doc"]))

    def _retrieve_json(self, reference, page):
        try:
            response = self.api.get_video(reference, page)
            return response.json()
        except json.JSONDecodeError:
            log.error(f"Error decoding interventions for {reference}")
            return None

    # -- grouping --------------------------------------------------------------

    def _group_by_session(self, interventions):
        grouped = OrderedDict()
        for intervention in interventions:
            grouped.setdefault(self._session_link(intervention), []).append(
                intervention)
        return grouped

    def _session_link(self, intervention):
        pdia = intervention["pdia"].split("#")[0]
        return f"{SESSION_PATH}L{self._legislature(intervention)}/{pdia}"
