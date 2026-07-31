import json
from qhld_engine.extractors.errors import ExtractionError
from qhld_engine.logger import get_logger

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
            raise ExtractionError(f'No parliamentary groups file at {groups_file}')
        except ExtractionError:
            raise
        except Exception as e:
            log.error(f'Cannot import parliamentary groups due "{e}"')
            raise ExtractionError('The parliamentary groups file could not be imported')

    def __save(self, groups):
        for g in groups:
            try:
                group = ParliamentaryGroup(id=g['_id'])
                group['name'] = g['name']
                group['shortname'] = g['shortname']
                group['composition'] = ParliamentaryGroups.get_composition(g['shortname'])
                group['color'] = g['color']
                parties = []
                for party in g['parties']:
                    parties.append(party)
                group['parties'] = parties
                ParliamentaryGroups.save(group)
                log.info(f"{g['name']} loaded!")
            except Exception as e:
                log.error(f'Cannot create parliamentary group {g["_id"]} "{e}"')

    def calculate_composition(self):
        groups = ParliamentaryGroups.get_all()
        calculated = 0
        for group in groups:
            try:
                group['composition'] = ParliamentaryGroups.get_composition(group['shortname'])
                ParliamentaryGroups.save(group)
                calculated += 1
            except Exception as e:
                log.error(f'Cannot calculate composition for parliamentary group {group["_id"]} "{e}"')

        # One group failing is tolerable; none of them succeeding means the step did
        # nothing, which must not be reported as a completed run.
        if groups and not calculated:
            log.error(f'Could not calculate the composition of any of the {len(groups)} groups.')
            raise ExtractionError('No parliamentary group composition could be calculated')
