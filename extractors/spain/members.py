from logger import get_logger

from concurrent.futures import as_completed

from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

from .congress_api import CongressApi, CongressError, CongressForbiddenError
from .deputy_extractors.deputy_extractor import DeputyExtractor


log = get_logger(__name__)


class MembersExtractor:
    def __init__(self):
        self.ITEMS_PER_PAGE = 20
        self.BASE_URL = 'https://www.congreso.es/busqueda-de-diputados'
        self.total = 0
        self.references = []
        self.parliamentarygroups = ParliamentaryGroups.get_all()
        self.api = CongressApi()

    def extract(self):
        try:
            response = self.api.get_deputies()
        except CongressForbiddenError:
            log.error('Error 403 extracting the deputies.')
            return
        except CongressError:
            log.error('Unknown error extracting the deputies.')
            return

        json_data = response.json()

        items = json_data.get('data')
        for deputy in items:
            self.references.append(deputy['codParlamentario'])

        self.extract_deputies()

    def extract_deputies(self):
        future_requests = []

        for reference in self.references:
            future_requests.append(self.api.get_deputy(reference))

        for future in as_completed(future_requests):
            response = future.result()
            if response.ok:
                DeputyExtractor(response, self.parliamentarygroups).extract()
