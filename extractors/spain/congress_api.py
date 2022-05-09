import requests
from requests.structures import CaseInsensitiveDict
from requests_futures.sessions import FuturesSession


class CongressHeadersBuilder:

    def __init__(self):
        self.headers = CaseInsensitiveDict()
        self.set_defaults()

    def set_defaults(self):
        self.set_user_agent()
        self.set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
        self.set("Accept-Language", "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3")
        self.set("Accept-Encoding", "gzip, deflate, br")
        self.set("DNT", "1")
        self.set("Connection", "keep-alive")

    def set_user_agent(self, user_agent=None):
        if not user_agent:
            user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0"
        self.set("User-Agent", user_agent)

    def for_web(self):
        self.set("Upgrade-Insecure-Requests", "1")
        self.set("Sec-Fetch-Dest", "document")
        self.set("Sec-Fetch-Mode", "navigate")
        self.set("Sec-Fetch-Site", "none")
        self.set("Sec-Fetch-User", "?1")
        return self.headers

    def for_api(self):
        self.set("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        self.set("X-Requested-With", "XMLHttpRequest")
        self.set("Origin", "https://www.congreso.es")
        self.set("Sec-Fetch-Dest", "empty")
        self.set("Sec-Fetch-Mode", "cors")
        self.set("Sec-Fetch-Site", "same-origin")
        self.set("TE", "trailers")
        return self.headers

    def set(self, header, value):
        self.headers[header] = value
        return self

class CongressUrlBuilder:
    def __init__(self):
        self.url = "https://www.congreso.es/"

    def for_deputies(self):
        return self.url + 'busqueda-de-diputados?p_p_id=diputadomodule&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=searchDiputados&p_p_cacheability=cacheLevelPage'

    def for_deputy(self, code):
        return self.url + f'busqueda-de-diputados?p_p_id=diputadomodule&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_diputadomodule_mostrarFicha=true&codParlamentario={code}&idLegislatura=XIV&mostrarAgenda=false'

    def for_cookies(self):
        return self.url

    def for_initiative_totals(self):
        return self.url + "indice-de-iniciativas?p_p_id=iniciativas&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=cambiarLegislaturaIndice&p_p_cacheability=cacheLevelPage"

    def for_initiative(self, reference):
        return self.url + "busqueda-de-iniciativas?p_p_id=iniciativas&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_iniciativas_mode=mostrarDetalle&_iniciativas_legislatura=XIV&_iniciativas_id=" + reference.replace('/', '%2F')

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
        response = self.get(url, headers)
        self.cookies = response.cookies

    def get_deputies(self):
        url = self.url_builder.for_deputies()
        headers = CongressHeadersBuilder().for_api()
        data = "_diputadomodule_idLegislatura=14&_diputadomodule_genero=0&_diputadomodule_grupo=all&_diputadomodule_tipo=0&_diputadomodule_nombre=&_diputadomodule_apellidos=&_diputadomodule_formacion=all&_diputadomodule_filtroProvincias=%5B%5D&_diputadomodule_nombreCircunscripcion="
        return self.post(url, headers, data)

    def get_deputy(self, code):
        url = self.url_builder.for_deputy(code)
        headers = CongressHeadersBuilder().for_web()
        return self.async_get(url, headers)

    def get_initiative_totals(self):
        url = self.url_builder.for_initiative_totals()
        headers = CongressHeadersBuilder().for_api()
        data = "_iniciativas_legislatura=14+"
        return self.post(url, headers, data)

    def get_initiative(self, reference):
        url = self.url_builder.for_initiative(reference)
        headers = CongressHeadersBuilder().for_web()
        return self.async_get(url, headers)

    def get(self, url, headers):
        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            raise CongressForbiddenError
        if not response.ok:
            raise CongressError
        return response

    def async_get(self, url, headers):
        session = self.get_session()
        return session.get(url, headers=headers)

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
        api = CongressApi()
        api.get_deputies()

        print('Getting single deputies')
        api = CongressApi()
        api.get_deputy('267')
        api.get_deputy('237')

        print('Getting initiative totals')
        api = CongressApi()
        api.get_initiative_totals()

        print('Getting initiatives')
        api = CongressApi()
        api.get_initiative('181/001810')
        api.get_initiative('161/001542')
        api.get_initiative('184/057540')

# CongressApi.test()
