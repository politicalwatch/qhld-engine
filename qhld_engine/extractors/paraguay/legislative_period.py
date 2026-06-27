import requests

from qhld_engine.infrastructure.config.settings import get_settings
from .api import ENDPOINT


class LegislativePeriod:
    def __init__(self):
        self.__years = ""
        id_legislatura = get_settings().id_legislatura
        response = requests.get(ENDPOINT.format(method='periodo'))
        if response.ok:
            period = [x for x in response.json() if x['idPeriodoLegislativo'] == id_legislatura]
            if len(period):
                self.__years = period[0]['periodoLegislativo']

    def get(self):
        return self.__years

