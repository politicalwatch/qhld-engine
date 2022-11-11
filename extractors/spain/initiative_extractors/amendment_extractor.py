import requests
import re

from lxml.html import document_fromstring

from tipi_data.repositories.amendments import Amendments
from .amendments.partial_amendments import PartialAmendments
from .amendments.senate_amendments import SenateAmendments
from .amendments.totallity_amendments import TotallityAmendments

# amendment_types = [SenateAmendments]
amendment_types = [PartialAmendments, SenateAmendments, TotallityAmendments]

class AmendmentExtractor:
    TAG_RE = re.compile(r'<[^>]+>')  # TODO Move to utils
    BASE_URL = 'https://www.congreso.es'

    @staticmethod
    def can_have_amendments(initiative_type):
        types = ['120', '121', '122', '123', '124', '125', '127', '162', '162', '171', '173']

        return initiative_type in types

    def __init__(self, initiative, soup, node_tree):
        self.soup = soup
        self.initiative = initiative
        self.node_tree = node_tree

    def has_amendments(self):
        elements = self.find_amendment_elements()

        self.bulletins = {}
        for element in elements:
            bulletin_name = element[0].text_content()
            bulletin_link = element[1][0].attrib['href']

            self.bulletins[bulletin_name] = bulletin_link

        return bool(self.bulletins)

    def extract_amendments(self, bulletin_name):
        bulletin_content = self.extract_bulletin(bulletin_name)

        for type in amendment_types:
            if type.should_extract(bulletin_name):
                extractor = type(self.initiative['reference'], bulletin_content, bulletin_name)
                extractor.extract()

    def extract_bulletin(self, bulletin_name):
        bulletin_link = self.bulletins[bulletin_name]
        content = list()
        try:
            bulletin_tree = document_fromstring(requests.get(
                f"{self.BASE_URL}{bulletin_link}"
                ).text)
            content += self.retrieve_bulletin_content(bulletin_tree)

            more_links = bulletin_tree.xpath("//a[contains(text(), 'parte ')]")
            for link in more_links:
                page_url = link.get('href')
                page_bulletin_tree = document_fromstring(requests.get(
                    f"{page_url}"
                    ).text)
                new_content = self.retrieve_bulletin_content(page_bulletin_tree)
                content += new_content

            return '\n'.join(content)
        except IndexError:
            return list()
        except Exception as e:
            return list()

    def retrieve_bulletin_content(self, tree):
        content = []
        bulletin_content = tree.cssselect('.textoIntegro')
        if bulletin_content:
            content += [line for line in list(map(
                lambda x: self.TAG_RE.sub('', x).strip(),
                bulletin_content[0].itertext()
                )) if line != '']
        return content

    def find_amendment_elements(self):
        xpath = "//ul[@class='boletines']/li/div/text()[contains(.,'Enmiendas')]/../.."
        return self.node_tree.xpath(xpath)

    def extract(self):
        for bulletin_name in self.bulletins:
            amendments = Amendments.by_reference_and_bulletin(self.initiative['reference'], bulletin_name)
            if amendments:
                continue

            self.extract_amendments(bulletin_name)
