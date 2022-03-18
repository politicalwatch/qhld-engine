import re
import time
from datetime import datetime
from lxml.html import document_fromstring
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from mongoengine.errors import DoesNotExist

from tipi_data.models.initiative import Initiative
from tipi_data.models.parliamentarygroup import ParliamentaryGroup
from tipi_data.models.alert import create_alert
from tipi_data.utils import generate_id

from logger import get_logger
from .initiative_status import get_status, is_final_status
from .vote_extractor import VoteExtractor
from .video_extractor import VideoExtractor


log = get_logger(__name__)


TYPES_WITH_AUTHORS_ON_CONTENT = [
        'Proposición no de Ley ante el Pleno',
        'Proposición no de Ley en Comisión',
        'Proposición de ley de Grupos Parlamentarios del Congreso',
        'Proposición de ley de Diputados',
        ]


class InitiativeExtractor:

    def __init__(self, response, deputies, parliamentarygroups, grouped_deputies, places):
        self.attempts = 1
        self.MAX_ATTEMPTS = 3
        self.response = response
        self.url = response.url
        self.BASE_URL = 'https://www.congreso.es'
        self.date_regex = r'[0-9]{2}/[0-9]{2}/[0-9]{4}'
        self.deputies = deputies
        self.parliamentarygroups = parliamentarygroups
        self.places = places
        self.grouped_deputies = grouped_deputies
        self.parliamentarygroup_sufix = r' en el Congreso'
        self.__prepare(response)

    def __prepare(self, response):
        self.response = response
        self.url = response.url
        self.node_tree = document_fromstring(self.response.text)
        self.soup = BeautifulSoup(self.response.text, 'lxml')
        try:
            self.initiative = Initiative.objects.get(
                    reference=self.get_reference(),
                    initiative_type_alt__ne='Respuesta'
                    )
        except DoesNotExist:
            self.initiative = Initiative()
        except Exception as e:
            log.error('An error occurred trying to get the initiative')
            log.error(str(e))
            return

    def __try_again(self, url):
        if self.attempts >= self.MAX_ATTEMPTS:
            return
        self.attempts += 1
        response = requests.get(url)
        self.__prepare(response)
        self.extract()

    def extract(self):
        try:
            check = self.soup.select_one('.entradilla-iniciativa')
            if check is None:
                log.error(f"Error 404 getting {self.url} initiative. RETRYINGS...")
                self.__try_again(self.url)
                return
            self.extract_commons()
            previous_content = self.initiative['content'] if self.has_content() else list()
            if self.should_extract_content():
                self.extract_content()
                if self.should_extract_deputies_from_content():
                    self.populate_deputies_and_groups_from_content()
            self.initiative['id'] = self.generate_id(self.initiative)
            self.initiative['oldid'] = self.generate_oldid(self.initiative)
            if previous_content != self.initiative['content']:
                self.untag()
            else:
                if is_final_status(self.initiative['status']) and self.has_knowledge_bases():
                    create_alert(self.initiative)
            self.initiative.save()
            log.info(f"Iniciativa {self.initiative['reference']} procesada")
        except Exception as e:
            log.error(e)
            log.error(f"Error processing initiative {self.url}")

    def extract_content(self):
        self.initiative['content'] = []

    def extract_votes(self):
        votes_extractor = VoteExtractor(self.node_tree, self.initiative['reference'])
        votes_extractor.extract()

    def should_extract_content(self):
        return not self.has_content()

    def should_extract_deputies_from_content(self):
        # TODO Sacar esto a ../initiative_types.py
        if self.initiative['initiative_type_alt'] in TYPES_WITH_AUTHORS_ON_CONTENT:
            return True
        return False

    def extract_videos(self):
        extractor = VideoExtractor(self.initiative['reference'])
        extractor.extract()

    def extract_commons(self):
        try:
            # TODO get source initiative
            full_title = self.soup.select_one('.entradilla-iniciativa').text
            reference = self.get_reference()
            position = (len(reference) - reference.find('(', -15) + 2) * -1
            title = full_title[:position]

            self.initiative['title'] = title
            self.initiative['reference'] = reference
            self.initiative['initiative_type'] = self.initiative['reference'].split('/')[0]
            self.initiative['initiative_type_alt'] = self.soup.select('.titular-seccion')[1].text[:-1]
            self.initiative['place'] = self.get_place()
            if not self.has_authors():
                self.populate_authors()
            try:
                self.initiative['created'] = self.__parse_date(re.search(
                    self.date_regex,
                    self.soup.select_one('.f-present').text.split(',')[0].strip()).group())
            except Exception as e:
                log.error(f"Error extracting the creation date at {self.url}")
                log.error(str(e))
            self.initiative['updated'] = self.get_last_date()
            self.initiative['history'] = self.get_history()
            self.initiative['status'] = self.get_status()
            self.initiative['url'] = self.url
            if not self.has('tagged'):
                self.untag()
        except AttributeError as e:
            log.error(f"Error processing some attributes for initiative {self.url}")
            log.error(str(e))
        except Exception as e:
            log.error(str(e))

    def populate_deputies_and_groups_from_content(self):
        found_deputies = list()
        found_groups = list()

        def normalize_name(name):
            return ' '.join(name.split(',')[::-1]).strip()

        regex_line_with_deputies = r"Palacio del Congreso de los Diputados.*, Diputad[oa](s)?"
        try:
            groups = self.initiative['author_parliamentarygroups']
            deputies_to_search = self.grouped_deputies.get_deputies(groups)
        except Exception:
            deputies_to_search = self.grouped_deputies.get_deputies()
        content = ' '.join(self.initiative['content'])
        result = re.search(regex_line_with_deputies, content)
        if result is None:
            return
        for deputy in deputies_to_search:
            if re.search(normalize_name(deputy['name']), result.group()):
                found_deputies.append(deputy['name'])
                found_groups.append(
                        ParliamentaryGroup.objects(
                            shortname=deputy['parliamentarygroup']
                            ).first()['name'])

        self.initiative['author_deputies'] = found_deputies
        if not self.initiative['author_parliamentarygroups']:
            self.initiative['author_parliamentarygroups'] = list(set(found_groups))

    def generate_id(self, initiative):
        return initiative['reference'].replace('/', '-')

    def generate_oldid(self, initiative):
        return generate_id(initiative['reference'])

    def get_reference(self):
        url = urlparse(self.url)
        query = parse_qs(url[4])
        return query['_iniciativas_id'][0].replace('%2F', '/')

    def get_last_date(self):
        try:
            all_dates = re.findall(self.date_regex, self.soup.select_one('#portlet_iniciativas').text.strip())
            all_dates.sort(key=lambda d: time.mktime(time.strptime(d, "%d/%m/%Y")), reverse=True)
            return self.__parse_date([
                d
                for d in all_dates
                if time.mktime(time.strptime(d, "%d/%m/%Y")) < time.time()
                ][0])
        except Exception:
            return None

    def populate_authors(self):
        try:
            self.initiative['author_deputies'] = []
            self.initiative['author_parliamentarygroups'] = []
            self.initiative['author_others'] = []
            xpath = "//section[@id='portlet_iniciativas']//div[@class=' portlet-content-container']//h3[contains(text(),'Autor')]/following-sibling::ul[1]/li"
            authors_list = self.node_tree.xpath(xpath)

            for item in authors_list:
                a_tags = item.cssselect('a')
                if len(a_tags) == 0:
                    self.initiative['author_others'].append(item.text_content())
                else:
                    regex_short_parliamentarygroup = r' \(.+\)'
                    regex_more_deputies = r' y [0-9]+ Diputados'
                    _short_parliamentarygroup = re.search(regex_short_parliamentarygroup, item.text_content())
                    has_short_parliamentarygroup = _short_parliamentarygroup \
                        and self.__is_short_parliamentarygroup(_short_parliamentarygroup.group().strip()[1:-1])
                    if has_short_parliamentarygroup:
                        deputy_name = re.sub(regex_short_parliamentarygroup, '', item.text_content())
                        if self.__is_deputy(deputy_name):
                            self.initiative['author_deputies'].append(deputy_name)
                            parliamentarygroup_name = self.__get_parliamentarygroup_name(
                                    _short_parliamentarygroup.group()[2:][:-1])
                            if parliamentarygroup_name:
                                self.initiative['author_parliamentarygroups'].append(parliamentarygroup_name)
                    else:
                        if not re.search(regex_more_deputies, item.text_content()):
                            if self.__is_deputy(item.text_content()):
                                self.initiative['author_deputies'].append(item.text_content())
                                parliamentarygroup_name = self.__get_parliamentarygroup_name(
                                        self.__get_parliamentarygroup_from_deputy(item.text_content()))
                                if parliamentarygroup_name:
                                    self.initiative['author_parliamentarygroups'].append(parliamentarygroup_name)
                            else:
                                parliamentarygroup_name = item.text_content() \
                                    if self.parliamentarygroup_sufix not in item.text_content() \
                                    else re.sub(self.parliamentarygroup_sufix, '', item.text_content())
                                self.initiative['author_parliamentarygroups'].append(parliamentarygroup_name)
            self.initiative['author_parliamentarygroups'] = list(set(self.initiative['author_parliamentarygroups']))
        except Exception:
            log.error(f"Error extracting parliamentary groups at {self.url}")
            self.initiative['author_parliamentarygroups'] = list()

    def get_place(self):
        try:
            place_wrapper = self.soup.select_one('.comisionesCompetentes')
            if place_wrapper:
                return place_wrapper.text.strip()
            else:
                history = self.get_history()
                for place in self.places:
                    for history_item in history:
                        if place['name'] in history_item:
                            return place['name']
        except Exception as e:
            log.error("Error getting the initiative place")
            log.error(str(e))
            return ''

    def get_history(self):
        history = list()
        try:
            history_wrapper = self.soup.select_one('.iniciativaTramitacion')
            if history_wrapper:
                TAG_RE = re.compile(r'<[^>]+>')
                history = list(map(
                    lambda x: TAG_RE.sub('', x).strip(),
                    str(history_wrapper).split('<br/>')
                    ))
            final_status_wrapper = self.soup.select_one('.resultadoTramitacion')
            if final_status_wrapper:
                history.append(final_status_wrapper.text)
        except Exception as e:
            log.error("Error getting the initiative history")
            log.error(str(e))
            pass
        return history

    def get_status(self):
        return get_status(self.initiative['history'], self.initiative['initiative_type'])

    def __is_deputy(self, name):
        for deputy in self.deputies:
            if deputy.name == name:
                return True
        return False

    def __get_parliamentarygroup_from_deputy(self, name):
        for deputy in self.deputies:
            if deputy.name == name:
                return deputy['parliamentarygroup']
        return None

    def __is_parliamentarygroup(self, name):
        for parliamentarygroup in self.parliamentarygroups:
            if parliamentarygroup.name == name:
                return True
        return False

    def __is_short_parliamentarygroup(self, shortname):
        for parliamentarygroup in self.parliamentarygroups:
            if parliamentarygroup.shortname == shortname:
                return True
        return False

    def __get_parliamentarygroup_name(self, shortname):
        for parliamentarygroup in self.parliamentarygroups:
            if parliamentarygroup.shortname == shortname:
                return parliamentarygroup.name
        return None

    def __parse_date(self, str_date):
        split_date = str_date.split('/')
        if len(split_date) != 3:
            return None
        return datetime(int(split_date[2]), int(split_date[1]), int(split_date[0]))

    def has(self, field):
        try:
            if field not in self.initiative:
                return False
            return True
        except TypeError:
            return False

    def has_deputies(self):
        return self.has('author_deputies') and self.initiative['author_deputies'] != []

    def has_parliamentarygroups(self):
        return self.has('author_parliamentarygroups') and self.initiative['author_parliamentarygroups'] != []

    def has_others(self):
        return self.has('author_others') and self.initiative['author_others'] != []

    def has_authors(self):
        return self.has_deputies() or self.has_parliamentarygroups() or self.has_others()

    def has_content(self):
        return self.has('content') and len(self.initiative['content']) > 0

    def has_knowledge_bases(self):
        return self.has('tagged') and len(self.initiative['tagged']) > 0

    def untag(self):
        self.initiative['tagged'] = None
