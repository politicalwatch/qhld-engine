from importlib import import_module as im

from qhld_engine.application.freshness import (
    DEPUTIES,
    INITIATIVES,
    PARLIAMENTARY_GROUPS,
    mark_refreshed,
)
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_engine.logger import get_logger

log = get_logger(__name__)


class ExtractorTask():
    task_namespace = 'extractors'

    def __init__(self):
        module = get_settings().module_extractor
        self.members_extractor = im('qhld_engine.extractors.{}.members'.format(module)).MembersExtractor()
        self.groups_extractor = im('qhld_engine.extractors.{}.groups'.format(module)).GroupsExtractor()
        self.initiatives_extractor = im('qhld_engine.extractors.{}.initiatives'.format(module)).InitiativesExtractor()
        super().__init__()

    def run(self):
        print("{task}(says: ready to extract data!".format(task=self.__class__.__name__))
        self.members()
        self.calculate_composition_groups()
        self.initiatives()

    def members(self):
        self.members_extractor.extract()
        # Groups keep a stored 'composition' derived from the deputies, but only
        # calculate_composition_groups() recomputes it — so nothing about the
        # groups changed here, and their cached response is still valid.
        mark_refreshed(DEPUTIES)

    def load_groups(self, groups_file):
        self.groups_extractor.load(groups_file)
        mark_refreshed(PARLIAMENTARY_GROUPS)

    def calculate_composition_groups(self):
        self.groups_extractor.calculate_composition()
        mark_refreshed(PARLIAMENTARY_GROUPS)

    def totals(self):
        self.initiatives_extractor.extract_references()
        print(self.initiatives_extractor.all_references)

    def initiatives(self):
        self.initiatives_extractor.extract()
        mark_refreshed(INITIATIVES)

    def votes(self):
        self.initiatives_extractor.extract_references()
        self.initiatives_extractor.extract_votes()

    def interventions(self):
        self.initiatives_extractor.extract_references()
        self.initiatives_extractor.extract_videos()

    def speeches(self):
        """Daily speech extraction: sweep the full reference range of every
        configured debate type and extract only what is missing. Speeches are
        deliberately not driven by initiative freshness — the Diario PDF that
        carries the text is published after the initiative reaches its final
        status, so an initiative-based increment would never revisit them."""
        types = get_settings().speech_extraction_types
        if not types:
            log.warning(
                "No speech extraction types configured "
                "(SPEECH_EXTRACTION_TYPES); nothing to do")
            return
        for type_code in types:
            self.initiatives_extractor.extract_all_references_from_type(type_code)
        self._extract_speeches_incremental(self.initiatives_extractor.all_references)

    def references(self):
        self.initiatives_extractor.extract_references()
        print(self.initiatives_extractor.all_references)

    def all_initiatives(self):
        self.initiatives_extractor.extract_all_references()
        self.initiatives_extractor.extract_initiatives()
        mark_refreshed(INITIATIVES)

    def all_votes(self):
        self.initiatives_extractor.extract_all_references()
        self.initiatives_extractor.extract_votes()

    def all_interventions(self):
        self.initiatives_extractor.extract_all_references()
        self.initiatives_extractor.extract_videos()

    def all_speeches(self):
        self.initiatives_extractor.extract_all_references()
        self._extract_speeches(self.initiatives_extractor.all_references)

    def all_references(self):
        self.initiatives_extractor.extract_all_references()
        print(self.initiatives_extractor.all_references)

    def single_initiatives(self, reference):
        self.initiatives_extractor.all_references = [reference]
        self.initiatives_extractor.extract_initiatives()

    def single_interventions(self, reference):
        self.initiatives_extractor.all_references = [reference]
        self.initiatives_extractor.extract_videos()

    def single_speeches(self, reference=None, video_ids=None):
        """Extract one initiative's speeches, optionally saving only some of them.

        ``video_ids`` are Congress intervention ids (``video_intervencion.id01``). With
        a reference, that reference is processed and only those speeches are saved; with
        no reference, each id's own reference is looked up and the ids are grouped so a
        sitting is downloaded and segmented once however many speeches are targeted.

        Which reference is used does not change the text: a speech carrying several is
        an accumulated debate, and the Diario prints their expediente numbers as
        consecutive headings, so the windows differ but converge on the same first
        speaker line — measured byte-identical over four sittings. Hence the first of
        the roster, with the reference argument there to override it.

        Deliberately a debugging and gold-set tool. Saving only the targets leaves their
        neighbours holding blocks produced by older code, so the corpus quietly drifts
        out of step with the classifier — anything corpus-wide stays per-reference."""
        if not video_ids:
            if reference is None:
                raise ValueError("single_speeches needs a reference or a video id")
            return self._extract_speeches([reference])
        if reference is not None:
            return self._extract_speeches([reference], only=set(video_ids))
        saved = 0
        for resolved, targets in self._references_of(video_ids).items():
            print(f"{', '.join(sorted(targets))} -> {resolved}")
            saved += self._extract_speeches([resolved], only=targets)
        return saved

    @staticmethod
    def _references_of(video_ids):
        """Group the given intervention ids by the reference each one is extracted
        under. A speech that has never been extracted has no reference to find, so it
        has to be named explicitly."""
        from tipi_data import DoesNotExist
        from tipi_data.repositories.speeches import Speeches

        grouped = {}
        for video_id in video_ids:
            try:
                speech = Speeches.get_by_video_id(video_id)
            except DoesNotExist:
                raise ValueError(
                    f"No stored speech for video id {video_id}; pass the reference "
                    f"explicitly to extract it for the first time") from None
            if not speech.references:
                raise ValueError(
                    f"Speech {video_id} carries no reference; pass one explicitly")
            grouped.setdefault(speech.references[0], set()).add(video_id)
        return grouped

    def single_votes(self, reference):
        self.initiatives_extractor.all_references = [reference]
        self.initiatives_extractor.extract_votes()

    def type_initiatives(self, type_code):
        self.initiatives_extractor.extract_references_from_type(type_code)
        self.initiatives_extractor.extract_initiatives()

    def type_references(self, type_code):
        self.initiatives_extractor.extract_references_from_type(type_code)
        print(self.initiatives_extractor.all_references)

    def type_interventions(self, type_code):
        self.initiatives_extractor.extract_references_from_type(type_code)
        self.initiatives_extractor.extract_videos()

    def type_speeches(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self._extract_speeches_incremental(self.initiatives_extractor.all_references)

    def type_votes(self, type_code):
        self.initiatives_extractor.extract_references_from_type(type_code)
        self.initiatives_extractor.extract_votes()

    def type_all_initiatives(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self.initiatives_extractor.extract_initiatives()

    def type_all_references(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        print(self.initiatives_extractor.all_references)

    def type_all_interventions(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self.initiatives_extractor.extract_videos()

    def type_all_speeches(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self._extract_speeches(self.initiatives_extractor.all_references)

    def _extract_speeches(self, references, only=None):
        from qhld_engine.application.speeches.extract_speeches import ExtractSpeeches
        return ExtractSpeeches().execute(references, only)

    def _extract_speeches_incremental(self, references):
        from qhld_engine.application.speeches.extract_speeches import ExtractSpeeches
        ExtractSpeeches().execute_incremental(references)

    def type_all_votes(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self.initiatives_extractor.extract_votes()
