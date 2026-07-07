import re
import json
from qhld_engine.logger import get_logger

from lxml.etree import tostring
from html import unescape

from tipi_data.repositories.initiatives import Initiatives
from tipi_data.repositories.votings import Votings

from ..congress_api import CongressApi


log = get_logger(__name__)


def extract_json_link(html):
    regex = re.compile(r'<a[\sa-zA-Z\"\.=0-9_\/:]+\>JSON\<\/a\>')
    matches = regex.findall(html)
    if not matches:
        return None
    tag = matches[0]
    start = tag.find('href="') + 6
    link = tag[start:]
    end = link.find('"')
    return link[:end]


class VoteExtractor():
    JSON_XPATH = "//div[@class='votaciones']/div[1]/a[contains(text(), 'JSON')]"
    VOTE_TYPES = [
            'Toma en consideración',
            'Debate de totalidad',
            'Votación de conjunto'
            ]

    def __init__(self, tree, reference):
        self.tree = tree
        self.api = CongressApi()
        self.reference = reference
        initiative = Initiatives.get_first_by_reference(reference)
        self.title = initiative['title'] if initiative else ''

    def extract(self):
        votes_html = self.get_votes_html()
        has_a_type_vote = False
        for item in votes_html:
            for type in self.VOTE_TYPES:
                if type in item:
                    has_a_type_vote = True
                    self.__extract_item(item, type)
            if not has_a_type_vote:
                if self.title in item:
                    self.__extract_item(item, 'Principal')

    def __extract_item(self, item, label):
        link = extract_json_link(item)
        if link is None:
            log.warning(f"No JSON link found for {self.reference}")
            return
        log.info(f"Extracting '{label}' votes for {self.reference}")
        self.extract_votes(link)

    def get_votes_html(self):
        elements = self.tree.cssselect('.votaciones')
        if len(elements) == 0:
            return []
        element = elements[0]
        html_string = tostring(element).decode('utf-8').replace('&#13;', '')
        html = html_string[24:].strip()
        return self.split_html(html)

    def split_html(self, html):
        items = html.split('JSON</a>')
        cleaned = []
        for item in items:
            cleaned.append(unescape(item) + 'JSON</a>')

        return cleaned[:len(cleaned) - 1]

    def extract_votes(self, url):
        try:
            response = self.api.get_vote(url)
            data = response.json()
            self.save_votes(data)
        except json.decoder.JSONDecodeError:
            log.error(f"Error getting votes from {url}")

    def save_votes(self, data):
        Votings.save(self.reference, data)
