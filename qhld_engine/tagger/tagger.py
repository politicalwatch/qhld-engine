from qhld_engine.utils import FILES
from qhld_engine.extractors.extractor import ExtractorTask
from .tag_initiatives import TagInitiatives


class TaggerTask():
    task_namespace = 'tagger'

    def requires(self):
        return ExtractorTask()

    def run(self):
        print("{task} says: ready to tag initiatives!".format(task=self.__class__.__name__))
        TagInitiatives().run()
