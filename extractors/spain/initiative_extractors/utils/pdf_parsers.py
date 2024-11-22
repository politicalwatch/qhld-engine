import tempfile
from pdf2image import convert_from_path
from pytesseract import image_to_string
from pdfminer.high_level import extract_text

from ...congress_api import CongressApi


class PDFParser:
    def __init__(self, file):
        self.file = file

    def extract(self):
        try:
            content = extract_text(self.file.name)
            content = content.strip()
            content = content.replace("\f", " ").replace("\t", "").split("\n")
            return content
        except Exception:
            return []


class PDFImageParser:
    def __init__(self, file):
        self.file = file

    def extract(self):
        images = convert_from_path(self.file.name)
        texts = []

        for i in range(len(images)):
            text = image_to_string(images[i], lang="spa")
            texts.append(text)

        return texts


class PDFExtractor:
    BASE_URL = "https://www.congreso.es"

    def __init__(self, url, is_img=False):
        self.url = self.BASE_URL + url
        self.is_img = is_img
        self.api = CongressApi()

    def retrieve(self):
        response = self.api.get_pdf(self.url)
        content = []
        if not response.ok:
            return content
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as file:
                file.write(bytes(response.content))
                content = self.extract(file, self.is_img)
                file.close()
        except KeyError as e:
            print(e)
            pass
        return content

    def extract(self, pdf, is_img=False):
        content = []
        if is_img:
            parser = PDFImageParser(pdf)
        else:
            parser = PDFParser(pdf)

        try:
            content = parser.extract()
            if content == [] and not is_img:
                parser = PDFImageParser(pdf)
                content = parser.extract()

        except Exception as e:
            print(e)
            pass
        return content
