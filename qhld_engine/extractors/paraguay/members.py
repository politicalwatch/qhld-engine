import re

import requests
from tipi_data import DoesNotExist
from tipi_data.models.deputy import Deputy
from tipi_data.models.parliamentarygroup import ParliamentaryGroup
from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
from tipi_data.utils import generate_id

from qhld_engine.logger import get_logger
from .api import ENDPOINT
from .legislative_period import LegislativePeriod


log = get_logger(__name__)


class MembersExtractor:
    def __init__(self):
        self.__legislative_period = LegislativePeriod().get()

    def __is_valid_user_part(self, user):
        USER_REGEX = re.compile(r"^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*\Z")
        return USER_REGEX.match(user) is not None

    def __is_valid_domain_part(self, domain):
        DOMAIN_REGEX = re.compile(r"((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+)(?:[A-Z0-9-]{2,63}(?<!-))\Z")
        return DOMAIN_REGEX.match(domain) is not None

    def __validate_email(self, email):
        if email == "":
            return None
        user, domain = email.rsplit('@', 1)
        if self.__is_valid_user_part(user) and self.__is_valid_domain_part(domain):
            return email
        return None


    def __create_or_update(self, remote_member):
        member = Deputy(
                id=str(remote_member['idParlamentario']),
                name="{} {}".format(
                    remote_member['nombres'].strip().title(),
                    remote_member['apellidos'].strip().title()),
                parliamentarygroup=remote_member['partidoPolitico'],
                image=remote_member['fotoURL'],
                email=self.__validate_email(remote_member['emailParlamentario']),
                web=None,
                twitter=None,
                start_date=None,
                end_date=None,
                url=remote_member['appURL'],
                active=False
                )
        Deputies.save(member)
        log.info("Parlamentario {} procesado".format(str(remote_member['idParlamentario'])))

    def __refresh_members(self):
        response = requests.get(ENDPOINT.format(method='parlamentario'))
        if response.ok:
            for member in response.json():
                self.__create_or_update(member)

    def __refresh_parliamentarygroups(self):
        groups = Deputies.distinct_parliamentarygroups()
        groups.remove('')
        for group in groups:
            pg = ParliamentaryGroup(
                    id=generate_id(group),
                    name=group,
                    shortname=group,
                    active=True
                    )
            ParliamentaryGroups.save(pg)

    def __update_active_members(self, chamber='S'):
        response = requests.get(ENDPOINT.format(method='parlamentario/camara/{}'.format(chamber)))
        if response.ok:
            for mp in response.json():
                try:
                    member = Deputies.get(str(mp['idParlamentario']))
                    member.active = True
                    Deputies.save(member)
                except DoesNotExist:
                    log.warning("Extracting members: Deputy does not exists with id {}".format(id))

    def extract(self):
        self.__refresh_members()
        self.__refresh_parliamentarygroups()
        for chamber in ['S', 'D']:
            self.__update_active_members(chamber)
