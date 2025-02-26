import re
import html

from lxml.html import document_fromstring, tostring

from .initiative_extractor import InitiativeExtractor

from logger import get_logger


log = get_logger(__name__)


class BulletinsExtractor(InitiativeExtractor):
    LETTER = 'Z'
    TAG_RE = re.compile(r'<[^>]+>')  # TODO Move to utils

    def extract_content(self):
        self.initiative['content'] = self.retrieve_bulletin()

    def should_extract_content(self):
        urls = self.find_urls()
        has_content = self.has_content()
        has_extra = self.has('extra')
        has_imported_bulletins = has_extra and 'imported_bulletins' in self.initiative['extra']
        has_different_bulletins = has_imported_bulletins and self.initiative['extra']['imported_bulletins'] != len(urls)

        return not has_content or has_different_bulletins

    def retrieve_bulletin(self):
        content = list()
        urls = self.find_urls()
        self.initiative['extra']['imported_bulletins'] = len(urls)
        try:
            for url in urls:
                bulletin_tree = document_fromstring(self.api.get_url(
                    f"{self.BASE_URL}{url}"
                    ).text)
                content += self.retrieve_bulletin_content(bulletin_tree)

                more_links = bulletin_tree.xpath("//a[contains(text(), 'parte ')]")
                for link in more_links:
                    page_url = link.get('href')
                    page_bulletin_tree = document_fromstring(self.api.get_url(
                        f"{page_url}"
                        ).text)
                    new_content = self.retrieve_bulletin_content(page_bulletin_tree)
                    content += new_content

            return content
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

    def find_urls(self):
        xpath = self.get_xpath()
        return self.node_tree.xpath(xpath)

    def get_xpath(self):
        return f"//ul[@class='boletines']/li/div[contains(text(),'{self.LETTER}-')]/following-sibling::div/a[1]/@href"


class ABulletinsExtractor(BulletinsExtractor):
    LETTER = 'A'


class BBulletinsExtractor(BulletinsExtractor):
    LETTER = 'B'


class CBulletinsExtractor(BulletinsExtractor):
    LETTER = 'C'


class DBulletinsExtractor(BulletinsExtractor):
    LETTER = 'D'


class EBulletinsExtractor(BulletinsExtractor):
    LETTER = 'E'


class FirstBulletinExtractor(BulletinsExtractor):
    LETTER = 'Z'

    def get_xpath(self):
        return f"(//ul[@class='boletines']/li/div[contains(text(),'{self.LETTER}-')])[1]/following-sibling::div/a[1]/@href"

    def should_extract_content(self):
        return not self.has_content()


class FirstABulletinExtractor(FirstBulletinExtractor):
    LETTER = 'A'


class FirstBBulletinExtractor(FirstBulletinExtractor):
    LETTER = 'B'


class FirstCBulletinExtractor(FirstBulletinExtractor):
    LETTER = 'C'


class FirstDBulletinExtractor(FirstBulletinExtractor):
    LETTER = 'D'


class FirstEBulletinExtractor(FirstBulletinExtractor):
    LETTER = 'E'


class NonExclusiveBulletinExtractor(InitiativeExtractor):
    BASE_URL = 'https://www.congreso.es'
    PAGE_FIND_REGEX = r'Pág.:\s([0-9]+)'
    HTML_STRIP_REGEX = r'<[^>]+>'
    INITIATIVE_REFERENCE_REGEX = r'[0-9]{3}\/[0-9]{6}'

    def extract_content(self):
        self.initiative['content'] = self.extract_bulletin_content()

    def extract_bulletin_metadata(self):
        text = self.node_tree.xpath("//ul[@class='boletines']/li[1]/div[1]")[0].text_content()
        link = self.node_tree.xpath("//ul[@class='boletines']/li[1]/div[2]/a[1]/@href")[0]

        self.page = re.search(self.PAGE_FIND_REGEX, text).group(1)
        self.link = self.BASE_URL + link

    def extract_bulletin_content(self):
        try:
            self.extract_bulletin_metadata()
        except Exception:
            # No Bulletin yet
            return []

        tree = document_fromstring(self.api.get_url(self.link).text)
        cleanup_content = ''

        try:
            element = tree.xpath("//div[contains(@class, 'textoIntegro')]")[0]
            cleanup_content += self.cleanup_content(element)

            more_links = tree.xpath("//a[contains(text(), 'parte ')]")
            for link in more_links:
                page_url = link.get('href')
                page_bulletin_tree = document_fromstring(self.api.get_url(
                    f"{page_url}"
                    ).text)
                element = page_bulletin_tree.xpath("//div[contains(@class, 'textoIntegro')]")[0]
                cleanup_content += self.cleanup_content(element)
        except Exception:
            # Bulletin not properly formatted
            return []

        output = self.extract_initiative_from_bulletin(cleanup_content)

        # Initiative is published with the Senate.
        if output == [' (CD) ']:
            output = self.extract_senate_initiative(cleanup_content)

        return output

    def extract_initiative_from_bulletin(self, full_content):
        clean_content = self.clean_str_to_substr(full_content, 'Página ' + self.page)
        clean_content = self.clean_str_to_substr(clean_content, self.initiative['reference'])

        try:
            end_pos = re.search(self.INITIATIVE_REFERENCE_REGEX, clean_content).start()
        except Exception:
            # Last initiative in the Bulletin.
            return clean_content.split("\n")

        content = clean_content[:end_pos]
        return content.split("\n")

    def extract_senate_initiative(self, full_content):
        clean_content = self.clean_str_to_substr(full_content, 'Página ' + self.page)
        clean_content = self.clean_str_to_substr(clean_content, self.initiative['reference'])

        start_pos = re.search(self.INITIATIVE_REFERENCE_REGEX, clean_content).start()

        clean_content = clean_content[start_pos + 15:]
        try:
            end_pos = re.search(self.INITIATIVE_REFERENCE_REGEX, clean_content).start()
        except Exception:
            # Last initiative in the Bulletin.
            return clean_content.split("\n")

        content = clean_content[:end_pos]
        return content.split("\n")

    def cleanup_content(self, element):
        full_content = str(tostring(element))
        full_content = full_content.replace('<br><br><br><br>', "\n").replace('<br><br><br>', "\n").replace('<br>', " ")
        full_content = full_content.replace('    ', " ").replace('   ', " ").replace('  ', " ")
        full_content = re.sub(self.HTML_STRIP_REGEX, '', full_content)
        full_content = html.unescape(full_content)

        return full_content

    def clean_str_to_substr(self, haystack, needle):
        pos = haystack.find(needle) + len(needle)
        return haystack[pos:]

class BulletinAndSenateExtractor(NonExclusiveBulletinExtractor):
    SENATE_INITIATIVE_RE = r'[0-9]{3}\/[0-9]{6} \(S\)'

    def extract_initiative_from_bulletin(self, full_content):
        clean_content = self.clean_str_to_substr(full_content, 'Página ' + self.page)
        clean_content = self.clean_str_to_substr(clean_content, self.initiative['reference'])

        pos = re.search(self.SENATE_INITIATIVE_RE, clean_content).end()
        clean_content = clean_content[pos:]

        try:
            end_pos = re.search(self.INITIATIVE_REFERENCE_REGEX, clean_content).start()
        except Exception:
            # Last initiative in the Bulletin.
            return clean_content.split("\n")

        content = clean_content[:end_pos]
        return content.split("\n")
