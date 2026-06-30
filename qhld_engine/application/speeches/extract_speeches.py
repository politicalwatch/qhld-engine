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
from collections import OrderedDict

from qhld_engine.logger import get_logger
from qhld_engine.domain.speeches import segmentation
from qhld_engine.extractors.spain.congress_api import CongressApi
from qhld_engine.extractors.spain.initiative_extractors.utils.pdf_parsers import (
    PDFExtractor,
)

from tipi_data.utils import generate_id
from tipi_data.models.speech import Speech
from tipi_data.repositories.speeches import Speeches


log = get_logger(__name__)

INTERVENTIONS_PER_PAGE = 25
SESSION_PATH = "/public_oficiales/"


class ExtractSpeeches:

    def __init__(self):
        self.api = CongressApi()

    def execute(self, references):
        for reference in references:
            self._extract_reference(reference)

    def _extract_reference(self, reference):
        log.info(f"Getting speeches from {reference}")
        interventions = self._retrieve_all_interventions(reference)
        if not interventions:
            return

        surnames = [
            segmentation.speaker_surname_upper(i["orador"]) for i in interventions
        ]

        for session_link, items in self._group_by_session(interventions).items():
            raw = PDFExtractor(session_link, format_output=False).retrieve()
            if not raw:
                log.warning(f"No session text for {reference} at {session_link}")
                continue
            text = segmentation.normalize_session_text(raw, reference)
            text = segmentation.fix_speaker_typos(text, surnames)
            segmenter = segmentation.SpeechSegmenter(text)
            for intervention in items:
                self._extract_one(intervention, session_link, segmenter, reference)

    def _extract_one(self, intervention, session_link, segmenter, reference):
        speaker, group, surname = segmentation.parse_speaker(intervention["orador"])
        if speaker is None:
            log.warning(
                f"Unparseable speaker {intervention.get('orador')!r} for {reference}")
            return

        speaker_regex = segmentation.build_speaker_regex(intervention["orador"])
        text = segmenter.next_speech(speaker_regex)
        if text is None:
            log.warning(f"Speaker heading not found for {reference}")

        video = intervention["video_intervencion"]
        order = int(intervention["doc"])
        speech = Speech(
            id=generate_id(reference, session_link, str(order)),
            reference=reference,
            speaker=speaker,
            speaker_surname=surname,
            group=group,
            role=intervention.get("cargo_orador"),
            order=order,
            legislature=str(video["legislatura"]),
            date=intervention.get("fecha"),
            session_name=intervention.get("sesion", {}).get("nombre_sesion"),
            video_link=video.get("enlace_descarga02"),
            session_link=session_link,
            speech=text,
        )
        Speeches.save(speech)

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
        legislature = intervention["video_intervencion"]["legislatura"]
        pdia = intervention["pdia"].split("#")[0]
        return f"{SESSION_PATH}L{legislature}/{pdia}"
