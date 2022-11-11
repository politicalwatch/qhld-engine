from .base_amendments import BaseAmendments

class TotallityAmendments(BaseAmendments):

    @staticmethod
    def should_extract(name):
        return 'Enmiendas a la totalidad' in name
    
    def process_text(self, amendment, text_list):
        has_authors = False
        intro_skipped = False

        author_list = []
        parliamentary_group = ''
        content = []

        for index, item in enumerate(text_list):
            if self.should_skip(item):
                continue

            if index == 0:
                amendment.set_id(item)
                continue

            if item.isnumeric():
                continue

            if not has_authors:
                has_authors = self.process_authorship(amendment, item)
                continue

            if not intro_skipped:
                if item.startswith('JUSTIFICACIÓN'):
                    intro_skipped = True
                continue

            content.append(item)

        amendment.set_justification(content)
        amendment.add_type('Enmienda a la totalidad de devolución')
        amendment.save()
