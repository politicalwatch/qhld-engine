import json
from logger import get_logger

from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
from tipi_data.models.parliamentarygroup import ParliamentaryGroup


log = get_logger(__name__)


class GroupsExtractor:

    def load(self, groups_file):
        try:
            with open(f'{groups_file}', 'r') as f:
                self.__save(json.loads(f.read()))
                self.calculate_composition()
        except FileNotFoundError:
            log.error('Cannot import parliamentary groups due file not found')
        except Exception as e:
            log.error(f'Cannot import parliamentary groups due "{e}"')
        pass

    def __save(self, groups):
        for g in groups:
            try:
                group = ParliamentaryGroup()
                group['id'] = g['_id']
                group['name'] = g['name']
                group['shortname'] = g['shortname']
                group['composition'] = ParliamentaryGroups.get_composition(g['shortname'])
                group['color'] = g['color']
                parties = []
                for party in g['parties']:
                    parties.append(party)
                group['parties'] = parties
                group.save()
                log.info(f"{g['name']} loaded!")
            except Exception as e:
                log.error(f'Cannot create parliamentary group {g["_id"]} "{e}"')

    def calculate_composition(self):
        for group in ParliamentaryGroups.get_all():
            try:
                group['composition'] = ParliamentaryGroups.get_composition(group['shortname'])
                group.save()
            except Exception as e:
                log.error(f'Cannot calculate composition for parliamentary group {group["_id"]} "{e}"')
