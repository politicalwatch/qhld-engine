from importlib import import_module as im

from extractors.config import MODULE_EXTRACTOR


class ExtractorTask():
    task_namespace = 'extractors'

    def __init__(self):
        self.members_extractor = im('extractors.{}.members'.format(MODULE_EXTRACTOR)).MembersExtractor()
        self.groups_extractor = im('extractors.{}.groups'.format(MODULE_EXTRACTOR)).GroupsExtractor()
        self.initiatives_extractor = im('extractors.{}.initiatives'.format(MODULE_EXTRACTOR)).InitiativesExtractor()
        super().__init__()

    def run(self):
        print("{task}(says: ready to extract data!".format(task=self.__class__.__name__))
        self.members()
        self.groups()
        self.initiatives()

    def members(self):
        self.members_extractor.extract()

    def groups(self):
        self.groups_extractor.extract()

    def totals(self):
        self.initiatives_extractor.sync_totals()
        print(self.initiatives_extractor.totals_by_type)

    def initiatives(self):
        self.initiatives_extractor.extract()

    def votes(self):
        self.initiatives_extractor.extract_references()
        self.initiatives_extractor.extract_votes()

    def interventions(self):
        self.initiatives_extractor.extract_references()
        self.initiatives_extractor.extract_videos()

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

    def all_references(self):
        self.initiatives_extractor.extract_all_references()
        print(self.initiatives_extractor.all_references)

    def single_initiatives(self, reference):
        self.initiatives_extractor.all_references = [reference]
        self.initiatives_extractor.extract_initiatives()

    def single_interventions(self, reference):
        self.initiatives_extractor.all_references = [reference]
        self.initiatives_extractor.extract_videos()

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

    def type_all_votes(self, type_code):
        self.initiatives_extractor.extract_all_references_from_type(type_code)
        self.initiatives_extractor.extract_votes()
