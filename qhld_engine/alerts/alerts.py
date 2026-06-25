from .send_alerts import SendAlerts
from .settings import USE_ALERTS
from qhld_engine.tagger.tagger import TaggerTask
from qhld_engine.utils import FILES


class GenerateAlertsTask():
    task_namespace = 'alerts'

    def requires(self):
        return TaggerTask()

    def run(self):
        print("{task} says: ready to generate alerts!".format(task=self.__class__.__name__))
        if USE_ALERTS:
            SendAlerts()
