import re

from .base_amendments import BaseAmendments

class SenateAmendments(BaseAmendments):

    @staticmethod
    def should_extract(name):
        return 'Enmiendas (Senado)' in name
    
    def process_text(self, amendment, text_list):
        has_authors = False
        has_propossed_change = False
        has_type = False
        intro_skipped = False

        applies_to = ''
        author_list = []
        justification = []
        propossed_change = []

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

            if item.startswith('RETIRADA'):
                amendment.add_type('RETIRADA')
                break

            if not intro_skipped:
                if item.startswith('ENMIENDA'):
                    intro_skipped = True
                continue

            if not has_type:
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
        amendment.mark_as_senate()
        amendment.save()

    def process_authorship(self, amendment, item):
        if item == 'ENMIENDA':
            return True

        group_regex = re.compile(r"\((?P<group>[a-zA-Z]*)\)")
        result = group_regex.search(item)
        if result is None:
            return False

        parts = result.groupdict()
        group = parts['group']
        amendment.add_group(group)
        return True
