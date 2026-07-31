from qhld_engine.logger import get_logger

from concurrent.futures import as_completed

from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

from qhld_engine.extractors.errors import ExtractionError

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
            raise ExtractionError('The deputies list answered 403')
        except CongressError:
            log.error('Unknown error extracting the deputies.')
            raise ExtractionError('The deputies list could not be read')

        json_data = response.json()

        items = json_data.get('data')
        for deputy in items:
            self.references.append(deputy['codParlamentario'])

        self.extract_deputies()

    def extract_deputies(self):
        future_requests = []

        for reference in self.references:
            future_requests.append(self.api.get_deputy(reference))

        extracted = 0
        for future in as_completed(future_requests):
            response = future.result()
            if response.ok:
                DeputyExtractor(response, self.parliamentarygroups).extract()
                extracted += 1
            else:
                log.error(f'Error {response.status_code} extracting a deputy on {response.url}')

        # A deputy failing on its own is tolerable; the whole roster failing is not,
        # and it looks identical to a clean run unless we say so.
        if self.references and not extracted:
            log.error(f'None of the {len(self.references)} deputies could be extracted.')
            raise ExtractionError('No deputy could be extracted')
