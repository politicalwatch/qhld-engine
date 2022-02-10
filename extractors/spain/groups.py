import os
import json
from logger import get_logger

from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
from tipi_data.models.parliamentarygroup import ParliamentaryGroup


log = get_logger(__name__)


class GroupsExtractor:

    def __init__(self):
        try:
            dirname = os.path.dirname(os.path.realpath(__file__))
            with open(f'{dirname}/groups.json', 'r') as f:
                self.groups = json.loads(f.read())
        except FileNotFoundError:
            log.error('Cannot import parliamentary groups due file not found')
        except Exception as e:
            log.error(f'Cannot import parliamentary groups due "{e}"')

    def extract(self):
        for g in self.groups:
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
            except Exception as e:
                log.error(f'Cannot create parliamentary group {g["_id"]} "{e}"')
