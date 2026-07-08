from importlib import import_module as im

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

    def load_groups(self, groups_file):
        self.groups_extractor.load(groups_file)

    def calculate_composition_groups(self):
        self.groups_extractor.calculate_composition()

    def totals(self):
        self.initiatives_extractor.extract_references()
        print(self.initiatives_extractor.all_references)

    def initiatives(self):
        self.initiatives_extractor.extract()

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

    def single_speeches(self, reference):
        self._extract_speeches([reference])

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

    def _extract_speeches(self, references):
        from qhld_engine.application.speeches.extract_speeches import ExtractSpeeches
        ExtractSpeeches().execute(references)

    def _extract_speeches_incremental(self, references):
        from qhld_engine.application.speeches.extract_speeches import ExtractSpeeches
        ExtractSpeeches().execute_incremental(references)

    def type_all_votes(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self.initiatives_extractor.extract_votes()
