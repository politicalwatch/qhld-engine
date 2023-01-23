import math
import json
from logger import get_logger

from tipi_data.utils import generate_id
from tipi_data.models.video import Video

from extractors.config import ID_LEGISLATURA
from extractors.spain.utils import int_to_roman
from ..congress_api import CongressApi


log = get_logger(__name__)


class VideoExtractor():

    def __init__(self, reference):
        self.reference = reference
        self.api = CongressApi()

    def extract(self):
        log.info(f"Getting videos from {self.reference}")
        json = self.retrieve_json(1)
        if 'error' in json or 'intervenciones_encontradas' not in json:
            return

        total = int(json['intervenciones_encontradas'])
        pages = math.ceil(total / 25)

        self.extract_interventions(json['lista_intervenciones'])

        if pages > 1:
            for x in range(2, pages):
                json = self.retrieve_json(x)
                self.extract_interventions(json['lista_intervenciones'])

    def extract_interventions(self, interventions):
        for intervention_key in interventions:
            json = interventions[intervention_key]

            video = Video()
            video['link'] = json['video_intervencion']['enlace_descarga02']
            video['id'] = self.generate_id(video['link'])
            video['reference'] = self.reference
            video['date'] = json['fecha']
            video['session_name'] = json['sesion']['nombre_sesion']

            if 'tipo_intervencion' in json:
                video['type'] = json['tipo_intervencion']
            if 'orador' in json:
                video['speaker'] = json['orador']

            video.save()

    def retrieve_json(self, page):
        try:
            response = self.api.get_video(self.reference, page)
            return response.json()
        except json.decoder.JSONDecodeError:
            log.error(f"Error getting video for {self.reference}")

    def generate_id(self, link):
        return generate_id(link)
