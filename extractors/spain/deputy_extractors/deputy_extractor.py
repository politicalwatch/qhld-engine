import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from lxml.html import document_fromstring

from tipi_data.models.deputy import Deputy

from tipi_data.utils import generate_slug

class DeputyExtractor():
    BASE_URL = 'https://www.congreso.es'

    def __init__(self, response):
        self.response = response
        self.node_tree = document_fromstring(response.text)
        self.deputy = Deputy()

    def extract(self):
        self.deputy['name'] = self.get_text_by_css('.nombre-dip')
        self.deputy['parliamentarygroup'] = self.get_abbr_group()
        self.deputy['image'] = self.BASE_URL + self.get_src_by_css('.img-dip img')
        self.deputy['public_position'] = self.get_public_positions()
        self.deputy['party_logo'] = self.get_src_by_css('.logo-partido img')
        self.deputy['party_name'] = self.get_text_by_css('.siglas-partido')
        self.deputy['url'] = self.response.url
        self.deputy['gender'], self.deputy['constituency'] = self.get_gender_and_constituency_from(
                self.get_text_by_css('.cargo-dip')
                )
        self.deputy['id'] = self.generate_id()

        self.extract_social_media()
        self.extract_extras()
        self.extract_dates()
        self.extract_from_text()
        self.extract_mail()
        self.deputy.save()

    def get_src_by_css(self, selector):
        item = self.get_by_css(selector)
        if len(item) == 0:
            return ''

        return self.clean_str(item[0].get('src'))

    def get_text_by_css(self, selector):
        item = self.get_by_css(selector)
        if len(item) == 0:
            return ''
        return self.clean_str(item[0].text)


    def get_by_css(self, selector):
        return self.node_tree.cssselect(selector)

    def get_by_xpath(self, xpath):
        return self.node_tree.xpath(xpath)

    def get_abbr_group(self):
        abbr_group_regex = '\(([^)]+)'
        group = self.get_text_by_css('.grupo-dip a')
        if not group:
            return ''
        return re.search(abbr_group_regex, group).group(1).strip()

    def extract_mail(self):
        mail = self.get_text_by_css('.email-dip a')
        if mail != '':
            self.deputy['email'] = mail

    def clean_str(self, string):
        return re.sub('\s+', ' ', string).strip()

    def parse_date(self, string):
        try:
            return datetime.strptime(re.sub(r'CE(S)?T ', '', string.strip()), "%c")
        except Exception as e:
            return None

    def get_public_positions(self):
        positions = []
        for position in self.get_by_css('.cargos:not(.ult-init) li'):
            positions.append(self.clean_str(position.text_content()))
        return positions

    def extract_dates(self):
        date_elements = self.get_by_css('.f-alta')
        end_date = self.clean_str(date_elements[1].text_content()).replace("Causó baja el ", "")[:28]

        self.deputy['start_date'] = self.parse_date(self.clean_str(date_elements[0].text_content()).replace("Condición plena: ", "")[:28])
        if end_date != '':
            self.deputy['end_date'] = self.parse_date(end_date)
        self.deputy['active'] = end_date == ''

    def extract_social_media(self):
        social_links = self.get_by_css('.rrss-dip a')
        for link in social_links:
            img_src = link.getchildren()[0].get('src')
            if 'twitter' in img_src:
                self.deputy['twitter'] = self.get_link_url(link)
            if 'facebook' in img_src:
                self.deputy['facebook'] = self.get_link_url(link)
            if 'web' in img_src:
                self.deputy['web'] = self.get_link_url(link)

    def get_link_url(self, link):
        url = link.get('href')
        if url.find('http') != 0:
            url = 'http://' + url
        return url

    def get_gender_and_constituency_from(self, string):
        array_string = string.split()
        gender = 'Mujer' if array_string[0] == 'Diputada' else 'Hombre'
        for _ in range(2):
            array_string.pop(0)
        constituency = " ".join(array_string)
        return gender, constituency

    def extract_extras(self):
        self.deputy['extra'] = {}
        links = self.get_by_css('.declaraciones-dip a')
        if links:
            self.deputy['extra']['declarations'] = {}
        for link in links:
            self.deputy['extra']['declarations'][self.clean_str(link.text)] = self.BASE_URL + link.get('href')

    def extract_from_text(self):
        birthday_paragraph = self.clean_str(self.get_by_xpath("//h3[normalize-space(text()) = 'Ficha personal']/following-sibling::p[1]")[0].text)
        birthday = birthday_paragraph.replace("Nacido el ", "").replace("Nacida el ", "")[:29]
        if birthday != '':
            self.deputy['birthdate'] = self.parse_date(birthday)

        legislatures_paragraph = self.clean_str(self.get_by_xpath("//h3[normalize-space(text()) = 'Ficha personal']/following-sibling::p[2]")[0].text)
        self.deputy['legislatures'] = legislatures_paragraph.replace("Diputada", "").replace("Diputado", "").replace(" de la ", "").replace(" Legislaturas", "").replace("y ", "").split(", ")

        bio = self.clean_str(self.get_by_xpath("//h3[normalize-space(text()) = 'Ficha personal']/parent::div")[0].text_content())
        bio = bio.replace("Ficha personal", "").replace(birthday_paragraph, "").replace(legislatures_paragraph, "")
        pos = bio.find(' Condición plena')
        self.deputy['bio'] = self.clean_str(bio[:pos]).split('. ')

    def generate_id(self):
        return generate_slug(self.deputy['name'])
