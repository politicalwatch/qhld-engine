import requests

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from .deputy_extractors.deputy_extractor import DeputyExtractor

from logger import get_logger


log = get_logger(__name__)


class MembersExtractor:
    def __init__(self):
        self.ITEMS_PER_PAGE = 20
        self.BASE_URL = 'https://www.congreso.es/web/guest/busqueda-de-diputados'
        self.total = 0
        self.LEGISLATURE = 14
        self.references = []

    def extract(self):
        query_params = {
            'p_p_id': 'diputadomodule',
            'p_p_lifecycle': 2,
            'p_p_state': 'normal',
            'p_p_mode': 'view',
            'p_p_resource_id': 'searchDiputados',
            'p_p_cacheability': 'cacheLevelPage'
        }
        form_data = {
            '_diputadomodule_nombreCircunscripcion': '',
            '_diputadomodule_filtroProvincias': '[]',
            '_diputadomodule_formacion': 'all',
            '_diputadomodule_apellidos': '',
            '_diputadomodule_nombre': '',
            '_diputadomodule_tipo': '2',
            '_diputadomodule_grupo': 'all',
            '_diputadomodule_genero': '0',
            '_diputadomodule_idLegislatura': self.LEGISLATURE
        }

        response = requests.post(
            self.BASE_URL,
            params=query_params,
            data=form_data
        )
        items = response.json().get('data')
        for deputy in items:
            self.references.append(deputy['codParlamentario'])

        self.extract_deputies()

    def extract_deputies(self):
        future_requests = []
        for reference in self.references:
            query_params = {
                'p_p_id': 'diputadomodule',
                'p_p_lifecycle': 0,
                'p_p_state': 'normal',
                'p_p_mode': 'view',
                '_diputadomodule_mostrarFicha': 'true',
                'codParlamentario': reference,
                'idLegislatura': 'XIV',
                'mostrarAgenda': 'false'
            }
            session = FuturesSession()
            # This header value prevents parsed dates on raw HTML content
            session.headers.update({"Accept-Language": 'es-ES'})
            future_requests.append(session.get(
                self.BASE_URL,
                params=query_params
            ))
        for future in as_completed(future_requests):
            response = future.result()
            if response.ok:
                DeputyExtractor(response).extract()
