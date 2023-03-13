import requests
from requests.structures import CaseInsensitiveDict
from requests_futures.sessions import FuturesSession

from extractors.config import ID_LEGISLATURA, CURRENT_LEGISLATURE
from .utils import int_to_roman


class CongressHeadersBuilder:

    def __init__(self):
        self.headers = CaseInsensitiveDict()

    def set_defaults(self):
        self.set_user_agent()
        self.set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
        self.set("Accept-Language", "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3")
        self.set("Accept-Encoding", "gzip, deflate, br")
        self.set("DNT", "1")
        self.set("Connection", "keep-alive")
        self.set("Host", "www.congreso.es")

    def set_user_agent(self, user_agent=None):
        if not user_agent:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:4.8) Goanna/20220409 PaleMoon/29.4.6"
        self.set("User-Agent", user_agent)

    def for_api(self):
        self.set_defaults()
        self.set("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        self.set("X-Requested-With", "XMLHttpRequest")
        self.set("Origin", "https://www.congreso.es")
        self.set("Sec-Fetch-Dest", "empty")
        self.set("Sec-Fetch-Mode", "cors")
        self.set("Sec-Fetch-Site", "same-origin")
        self.set("TE", "trailers")
        return self.headers

    def for_web(self):
        self.set_user_agent()
        self.set('Cookie', 'GUEST_LANGUAGE_ID=es_ES')
        self.set('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8')
        self.set("Host", "www.congreso.es")
        self.set('Accept-Language', 'en-us')
        self.set('Accept-Encoding', 'br, gzip, deflate')
        self.set("Connection", "keep-alive")
        return self.headers

    def for_pdf(self):
        self.set_user_agent()
        self.set('Cookie', 'GUEST_LANGUAGE_ID=es_ES')
        self.set('Accept', 'application/pdf;q=0.9,*/*;q=0.8')
        self.set("Host", "www.congreso.es")
        self.set('Accept-Language', 'en-us')
        self.set('Accept-Encoding', 'br, gzip, deflate')
        self.set("Connection", "keep-alive")
        return self.headers

    def set(self, header, value):
        self.headers[header] = value
        return self


class CongressUrlBuilder:
    def __init__(self):
        self.url = "https://www.congreso.es/es/"

    def for_deputies(self):
        return f'{self.url}busqueda-de-diputados?p_p_id=diputadomodule&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=searchDiputados&p_p_cacheability=cacheLevelPage'

    def for_deputy(self, code):
        return f'{self.url}busqueda-de-diputados?p_p_id=diputadomodule&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_diputadomodule_mostrarFicha=true&codParlamentario={code}&idLegislatura={int_to_roman(ID_LEGISLATURA)}&mostrarAgenda=false'

    def for_cookies(self):
        return f'{self.url}/busqueda-de-diputados'

    def for_initiative_totals(self):
        return f'{self.url}indice-de-iniciativas?p_p_id=iniciativas&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=cambiarLegislaturaIndice&p_p_cacheability=cacheLevelPage'

    def for_initiative(self, reference):
        return f"{self.url}busqueda-de-iniciativas?p_p_id=iniciativas&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_iniciativas_mode=mostrarDetalle&_iniciativas_legislatura={int_to_roman(ID_LEGISLATURA)}&_iniciativas_id={reference.replace('/', '%2F')}"

    def for_video(self, reference):
        return f'https://www.congreso.es/web/guest/busqueda-de-intervenciones?p_p_id=intervenciones&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=filtrarListado&p_p_cacheability=cacheLevelPage&_intervenciones_mode=view&_intervenciones_legislatura={int_to_roman(ID_LEGISLATURA)}&_intervenciones_id_iniciativa={reference}'

    def for_url(self, link):
        if link.startswith('http'):
            return link
        return f"{self.url}{link}"


class CongressForbiddenError(Exception):
    pass


class CongressError(Exception):
    pass


class CongressApi(object):
    _instance = None
    cookies = None
    session = None

    def __init__(self):
        self.url_builder = CongressUrlBuilder()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CongressApi, cls).__new__(cls)
        return cls._instance

    def get_session(self):
        if not self.session:
            self.session = FuturesSession()
        return self.session

    def get_cookies(self):
        if not self.cookies:
            self.request_cookies()
        return self.cookies

    def request_cookies(self):
        headers = CongressHeadersBuilder().for_web()
        url = self.url_builder.for_cookies()
        response = requests.get(url, headers=headers)
        self.cookies = response.cookies

    def get_deputies(self):
        url = self.url_builder.for_deputies()
        headers = CongressHeadersBuilder().for_api()
        deputies_type = '2' if CURRENT_LEGISLATURE else '1'
        data = f"_diputadomodule_idLegislatura={ID_LEGISLATURA}&_diputadomodule_genero=0&_diputadomodule_grupo=all&_diputadomodule_tipo={deputies_type}&_diputadomodule_nombre=&_diputadomodule_apellidos=&_diputadomodule_formacion=all&_diputadomodule_filtroProvincias=%5B%5D&_diputadomodule_nombreCircunscripcion="
        return self.post(url, headers, data)

    def get_deputy(self, code):
        url = self.url_builder.for_deputy(code)
        headers = CongressHeadersBuilder().for_web()
        return self.async_get(url, headers)

    def get_initiative_totals(self):
        url = self.url_builder.for_initiative_totals()
        headers = CongressHeadersBuilder().for_api()
        data = f"_iniciativas_legislatura={ID_LEGISLATURA}+"
        return self.post(url, headers, data)

    def get_initiative(self, reference):
        url = self.url_builder.for_initiative(reference)
        headers = CongressHeadersBuilder().for_web()
        return self.async_get(url, headers)

    def get_url(self, url):
        url = self.url_builder.for_url(url)
        headers = CongressHeadersBuilder().for_web()
        return self.get(url, headers)

    def get_amendment(self, link):
        url = self.url_builder.for_url(link)
        headers = CongressHeadersBuilder().for_web()
        return self.get(url, headers)

    def get_vote(self, url):
        headers = CongressHeadersBuilder().for_web()
        return self.get(url, headers)

    def get_video(self, reference, page):
        url = self.url_builder.for_video(reference)
        headers = CongressHeadersBuilder().for_api()
        data = {
            '_intervenciones_paginaActual': page
        }
        return self.post(url, headers, data)

    def get_pdf(self, url):
        headers = CongressHeadersBuilder().for_pdf()
        return self.get(url, headers)

    def get(self, url, headers):
        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            raise CongressForbiddenError
        if not response.ok:
            raise CongressError
        return response

    def async_get(self, url, headers):
        session = self.get_session()
        cookies = self.get_cookies()
        return session.get(url, headers=headers, cookies=cookies)

    def post(self, url, headers, data):
        cookies = self.get_cookies()
        response = requests.post(url, headers=headers, data=data, cookies=cookies)
        if response.status_code == 403:
            raise CongressForbiddenError
        if not response.ok:
            raise CongressError
        return response

    @staticmethod
    def test():
        print('Getting cookies')
        api = CongressApi()
        api.get_cookies()

        print('Getting deputies')
        api.get_deputies()

        print('Getting single deputies')
        api.get_deputy('267')
        api.get_deputy('237')

        print('Getting initiative totals')
        api.get_initiative_totals()

        print('Getting initiatives')
        api.get_initiative('181/001810')
        api.get_initiative('161/001542')
        api.get_initiative('184/057540')


#CongressApi.test()
