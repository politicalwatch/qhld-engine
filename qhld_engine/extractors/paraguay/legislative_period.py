import requests

from qhld_engine.extractors.config import ID_LEGISLATURA
from .api import ENDPOINT


class LegislativePeriod:
    def __init__(self):
        self.__years = ""
        response = requests.get(ENDPOINT.format(method='periodo'))
        if response.ok:
            period = [x for x in response.json() if x['idPeriodoLegislativo'] == ID_LEGISLATURA]
            if len(period):
                self.__years = period[0]['periodoLegislativo']

    def get(self):
        return self.__years

