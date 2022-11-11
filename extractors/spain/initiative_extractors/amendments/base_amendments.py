import re

from tipi_data.models.amendment import Amendment
from tipi_data.repositories.deputies import Deputies

class BaseAmendments:

    @staticmethod
    def should_extract(name):
        return '' in name
    
    def __init__(self, reference, content, name):
        self.reference = reference
        self.content = content
        self.name = name
        self.all_deputies = None

    def extract(self):
        amendments_texts = self.cleanup()

        for amendment_text in amendments_texts:
            self.create_amendment(amendment_text)

    def cleanup(self):
        delimiter = r"ENMIENDA NÚM"
        match = re.search(delimiter, self.content)
        if not match:
            # No amendments.
            return []
        content = self.content[match.start():]
        content = content.split("ENMIENDA NÚM.")
        content = [element for element in content if element != ""]
        return content

    def create_amendment(self, text):
        text_list = text.split('\n')
        amendment = Amendment(bulletin_name=self.name, reference=self.reference)
        amendment.mark_as_congress()

        self.process_text(amendment, text_list)

    def process_authorship(self, amendment, item):
        if item.startswith('A la Mesa de la Comisión de Presupuestos'):
            return True

        if item[0] == '(' or item[0:5] == 'Grupo':
            parliamentary_group = item.replace('(', '').replace(')', '')
            if parliamentary_group == 'Grupo Parlamentario Confederal de Unidas Podemos-En Comú Podem-Galicia en':
                parliamentary_group += ' Común'
            if parliamentary_group == 'Grupo Parlamentario Socialista Grupo Parlamentario Confederal de Unidas':
                amendment.add_group('Grupo Parlamentario Socialista')
                amendment.add_group('Grupo Parlamentario Confederal de Unidas Podemos-En Comú Podem-Galicia en Común')
                return True
            amendment.add_group(parliamentary_group)
            return True

        deputies = self.get_all_deputies()
        if item[-1] == ')':
            if item.startswith('Grupo'):
                amendment.add_group(item)
                return True

            group_regex = re.compile(r".*\((?P<group>.*)\)")
            result = group_regex.search(item)
            parts = result.groupdict()
            amendment.add_group(parts['group'])
            for deputy in deputies:
                if deputy.get_fullname() in item:
                    amendment.add_author(deputy.name)
            return True

        for deputy in deputies:
            if deputy.get_fullname() in item:
                amendment.add_author(deputy.name)
        return False

    def get_all_deputies(self):
        if not self.all_deputies:
            self.all_deputies = Deputies.get_all()
        return self.all_deputies

    def process_text(self, amendment, text_list):
        pass

    def should_skip(self, item):
        skipped = ['', '[**********página con cuadro**********]', 'Común', 'Podemos-En Comú Podem-Galicia en Común', '[...]\'', 'FIRMANTE:']
        return item.startswith('Página') or item in skipped
