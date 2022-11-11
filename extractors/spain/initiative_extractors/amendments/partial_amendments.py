from .base_amendments import BaseAmendments

class PartialAmendments(BaseAmendments):

    @staticmethod
    def should_extract(name):
        return 'Enmiendas e índice de enmiendas al articulado' in name
    
    def process_text(self, amendment, text_list):
        has_authors = False
        has_propossed_change = False
        has_type = False

        applies_to = ''
        author_list = []
        justification = []
        propossed_change = []

        # TODO fix this.
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

            if not has_type:
                if not item.startswith('De '):
                    applies_to += ' ' + item
                    continue

                if applies_to:
                    amendment.add_applies_to(applies_to.strip())

                amendment.add_type(item.replace('.', ''))
                has_type = True
                continue

            if not has_propossed_change:
                if item.startswith('JUSTIFICACIÓN') or item.startswith('MOTIVACIÓN'):
                    has_propossed_change = True
                    continue

                propossed_change.append(item)
                continue

            justification.append(item)

        amendment.set_justification(justification)
        amendment.set_propossed_change(propossed_change)
        amendment.save()
