import time
from logger import get_logger

import requests
from bs4 import BeautifulSoup
from requests_futures.sessions import FuturesSession
from concurrent.futures import as_completed

from tipi_data.models.deputy import Deputy
from tipi_data.models.parliamentarygroup import ParliamentaryGroup
from tipi_data.models.place import Place
from tipi_data.repositories.initiatives import Initiatives

from extractors.config import ID_LEGISLATURA
from .initiative_types import INITIATIVE_TYPES
from .initiative_extractor_factory import InitiativeExtractorFactory
from .initiative_extractors.initiative_status import NOT_FINAL_STATUS
from .initiative_extractors.video_extractor import VideoExtractor
from .initiative_extractors.vote_extractor import VoteExtractor
from .utils import int_to_roman


log = get_logger(__name__)


class InitiativesExtractor:

    def __init__(self):
        self.LEGISLATURE = ID_LEGISLATURA
        self.INITIATIVES_PER_PAGE = 25
        self.BASE_URL = 'https://www.congreso.es/web/guest/indice-de-iniciativas'
        self.SAFETY_EXTRACTION_GAP = 3
        self.totals_by_type = dict()
        self.all_references = list()
        self.deputies = Deputy.objects()
        self.parliamentarygroups = ParliamentaryGroup.objects()
        self.places = Place.objects()

    def get_types(self):
        return INITIATIVE_TYPES

    def sync_totals(self):
        query_params = {
                'p_p_id': 'iniciativas',
                'p_p_lifecycle': 2,
                'p_p_state': 'normal',
                'p_p_mode': 'view',
                'p_p_resource_id': 'cambiarLegislaturaIndice',
                '_iniciativas_legislatura': self.LEGISLATURE
                }
        response = requests.get(
                self.BASE_URL,
                params=query_params
                )
        soup = BeautifulSoup(response.json()['content'], 'lxml')
        for element in soup.select('.listado_1 li'):
            initiative_type = element.select_one('a').getText().strip()
            if initiative_type[-1] == '.':
                initiative_type = initiative_type[:-1]
            count = int(element.select_one('span').getText().strip('()'))
            self.totals_by_type[initiative_type] = count

    def extract(self):
        start_time = time.perf_counter()
        self.extract_references()

        log.info(f"Getting {len(self.all_references)} initiatives references")
        log.debug("--- %s seconds getting references---" % (time.perf_counter() - start_time))
        log.info("Processing initiatives...")

        start_time = time.perf_counter()
        self.extract_initiatives()
        log.debug("--- %s seconds getting initiatives ---" % (time.perf_counter() - start_time))

    def extract_all_references(self):
        self.sync_totals()
        for initiative_type in self.get_types():
            code = initiative_type['code']
            title = initiative_type['type']

            db_last_reference = 0
            db_total = 0
            origin_total = self.totals_by_type[title] if title in self.totals_by_type else 0

            new_items = origin_total - db_total
            if not new_items:
                continue

            for extra in range(1, new_items + self.SAFETY_EXTRACTION_GAP + 1):
                self.all_references.append(self.format_reference(db_last_reference + extra, code))

    def extract_all_references_from_type(self, type_code):
        self.sync_totals()
        for initiative_type in self.get_types():
            code = initiative_type['code']
            if type_code != code:
                continue
            title = initiative_type['type']

            db_last_reference = 0
            db_total = 0
            origin_total = self.totals_by_type[title] if title in self.totals_by_type else 0

            new_items = origin_total - db_total
            if not new_items:
                continue

            for extra in range(1, new_items + self.SAFETY_EXTRACTION_GAP + 1):
                self.all_references.append(self.format_reference(db_last_reference + extra, code))

    def extract_references_from_type(self, type_code):
        self.sync_totals()
        initiatives = Initiatives.get_all().filter(
                initiative_type=type_code).order_by('reference').only('reference', 'status')

        last_references = {}
        totals = {}
        previous_ref = 1

        for initiative in initiatives:
            if 'reference' not in initiative:
                continue
            items = initiative['reference'].split('/')
            initiative_type = items[0]
            reference = items[1]

            if initiative_type not in totals:
                totals[initiative_type] = 0

            last_references[initiative_type] = reference
            int_reference = int(reference)
            missing_references = self.calculate_references_between(previous_ref, int_reference, initiative_type)
            self.all_references += missing_references

            if initiative['status'] in NOT_FINAL_STATUS:
                self.all_references.append(initiative['reference'])

            totals[initiative_type] += 1 + len(missing_references)
            previous_ref = int_reference + 1

        for initiative_type in self.get_types():
            code = initiative_type['code']
            if type_code != code:
                continue
            title = initiative_type['type']

            db_last_reference = int(last_references[code]) if code in last_references else 0
            origin_total = self.totals_by_type[title] if title in self.totals_by_type else 0
            db_total = totals[code] if code in totals else 0

            new_items = origin_total - db_total
            if not new_items:
                continue

            for extra in range(1, new_items + self.SAFETY_EXTRACTION_GAP + 1):
                self.all_references.append(self.format_reference(db_last_reference + extra, code))

    def extract_references(self):
        self.sync_totals()
        initiatives = Initiatives.get_all().order_by('reference').only('reference', 'status')

        last_references = {}
        totals = {}
        previous_ref = 1

        for initiative in initiatives:
            if 'reference' not in initiative:
                continue
            items = initiative['reference'].split('/')
            initiative_type = items[0]
            reference = items[1]

            if initiative_type not in totals:
                totals[initiative_type] = 0

            last_references[initiative_type] = reference
            int_reference = int(reference)
            missing_references = self.calculate_references_between(previous_ref, int_reference, initiative_type)
            self.all_references += missing_references

            if initiative['status'] in NOT_FINAL_STATUS:
                self.all_references.append(initiative['reference'])

            totals[initiative_type] += 1 + len(missing_references)
            previous_ref = int_reference + 1

        for initiative_type in self.get_types():
            code = initiative_type['code']
            title = initiative_type['type']

            db_last_reference = int(last_references[code]) if code in last_references else 0
            origin_total = self.totals_by_type[title] if title in self.totals_by_type else 0
            db_total = totals[code] if code in totals else 0

            new_items = origin_total - db_total
            if not new_items:
                continue

            for extra in range(1, new_items + self.SAFETY_EXTRACTION_GAP + 1):
                self.all_references.append(self.format_reference(db_last_reference + extra, code))

    def calculate_references_between(self, previous_ref, new_ref, initiative_type):
        missing_references = []
        while previous_ref < new_ref:
            missing_reference = self.format_reference(previous_ref, initiative_type)
            previous_ref += 1
            missing_references.append(missing_reference)

        return missing_references

    def extract_initiatives(self):
        def callback(response):
            InitiativeExtractorFactory.create(
                response,
                self.deputies,
                self.parliamentarygroups,
                self.places
            ).extract()

        self.process_initiatives(callback)

    def process_initiatives(self, callback):
        session = FuturesSession()
        futures_requests = list()
        for reference in self.all_references:
            query_params = {
                    'p_p_id': 'iniciativas',
                    'p_p_lifecycle': 0,
                    'p_p_state': 'normal',
                    'p_p_mode': 'view',
                    '_iniciativas_mode': 'mostrarDetalle',
                    '_iniciativas_legislatura': int_to_roman(ID_LEGISLATURA),
                    '_iniciativas_id': reference
                    }
            futures_requests.append(session.get(
                self.BASE_URL,
                params=query_params))
        for future in as_completed(futures_requests):
            response = future.result()
            if response.ok:
                callback(response)

    def extract_videos(self):
        for reference in self.all_references:
            extractor = VideoExtractor(reference)
            extractor.extract()

    def extract_votes(self):
        def callback(response):
            extractor = InitiativeExtractorFactory.create(response, [], [], [])
            votes_extractor = VoteExtractor(extractor.node_tree, extractor.get_reference())
            votes_extractor.extract()

        self.process_initiatives(callback)

    def format_reference(self, ref, initiative_type):
        reference = str(ref)
        missing_zeros = 6 - len(reference)
        return initiative_type + '/' + ('0' * missing_zeros) + reference
