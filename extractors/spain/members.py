import os
import json
import requests
from requests.structures import CaseInsensitiveDict
from logger import get_logger

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from .deputy_extractors.deputy_extractor import DeputyExtractor


log = get_logger(__name__)


class MembersExtractor:
    def __init__(self):
        self.ITEMS_PER_PAGE = 20
        self.BASE_URL = 'https://www.congreso.es/busqueda-de-diputados'
        self.total = 0
        self.LEGISLATURE = 14
        self.references = []
        self.parliamentarygroups = self.__load_groups()

    def __load_groups(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dirname}/groups.json', 'r') as f:
            return json.loads(f.read())


    def extract(self):
        headers = CaseInsensitiveDict()
        headers["User-Agent"] = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0"
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        headers["Accept-Language"] = "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3"
        headers["Accept-Encoding"] = "gzip, deflate, br"
        headers["DNT"] = "1"
        headers["Connection"] = "keep-alive"
        headers["Upgrade-Insecure-Requests"] = "1"
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "none"
        headers["Sec-Fetch-User"] = "?1"

        req = requests.Request('get', self.BASE_URL, headers=headers)
        req = req.prepare()

        session = requests.Session()
        response = session.send(req)

        if not response.ok:
            log.error(f"Error {response.status_code} when requesting the members list on URL {response.url}.")
            return

        cookies = response.cookies

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
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
            data=form_data,
            headers=headers,
            cookies=cookies
        )

        if not response.ok:
            log.error(f"Error {response.status_code} when requesting the members list on URL {response.url}.")
            return

        json_data = response.json()

        items = json_data.get('data')
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
                DeputyExtractor(response, self.parliamentarygroups).extract()
