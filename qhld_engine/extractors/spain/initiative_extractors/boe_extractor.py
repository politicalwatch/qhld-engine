from .initiative_extractor import InitiativeExtractor
from .utils.pdf_parsers import PDFExtractor

class BoeExtractor(InitiativeExtractor):
    XPATH = "//ul[@class='boes']/li/div[2]/a/@href"

    def extract_content(self):
        self.initiative['content'] = self.retrieve_boe()

    def retrieve_boe(self):
        content = []

        for url in self.find_url():
            extractor = PDFExtractor(url)
            content += extractor.retrieve()

        return content


    def find_url(self):
        return self.node_tree.xpath(self.XPATH)

class FirstBoeExtractor(BoeExtractor):
    XPATH = "//ul[@class='boes']/li[1]/div[2]/a/@href"
