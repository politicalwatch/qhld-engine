from qhld_engine.alerts.alerts import GenerateAlertsTask
from qhld_engine.utils import clean_files, FILES
from .process_stats import GenerateStats


class GenerateStatsTask(:
    task_namespace = 'stats'

    def requires(self):
        return GenerateAlertsTask()


    def run(self):
        print("{task} says: ready to generate stats!".format(task=self.__class__.__name__))
        GenerateStats().generate()
        clean_files()
