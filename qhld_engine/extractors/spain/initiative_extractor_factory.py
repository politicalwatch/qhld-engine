from .initiative_extractors.initiative_extractor import InitiativeExtractor
from .initiative_types import INITIATIVE_TYPES
from urllib.parse import urlparse, parse_qs


class InitiativeExtractorFactory:
    @staticmethod
    def get_type(response):
        url = urlparse(response.request.path_url)
        query = parse_qs(url.query)
        return query.get('_iniciativas_id')[0].split('/')[0]

    @staticmethod
    def create(response, deputies, parliamentarygroups, grouped_deputies, places):
        extractor = InitiativeExtractor

        initiative_code = InitiativeExtractorFactory.get_type(response)

        for initiative_type in INITIATIVE_TYPES:
            if initiative_code == initiative_type.get('code') and "class" in initiative_type:
                extractor = initiative_type.get("class")

        return extractor(response, deputies, parliamentarygroups, grouped_deputies, places)

