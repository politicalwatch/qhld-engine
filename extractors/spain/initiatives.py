import time
from logger import get_logger

from bs4 import BeautifulSoup
from concurrent.futures import as_completed

from tipi_data.models.deputy import Deputy
from tipi_data.models.parliamentarygroup import ParliamentaryGroup
from tipi_data.models.place import Place
from tipi_data.repositories.initiatives import Initiatives

from .initiative_types import INITIATIVE_TYPES
from .initiative_extractor_factory import InitiativeExtractorFactory
from .initiative_extractors.initiative_status import NOT_FINAL_STATUS
from .grouped_deputies import GroupedDeputies
from .initiative_extractors.video_extractor import VideoExtractor
from .initiative_extractors.interventions_extractor import InterventionsExtractor
from .initiative_extractors.vote_extractor import VoteExtractor
from .congress_api import CongressApi


log = get_logger(__name__)


class InitiativesExtractor:

    def __init__(self):
        self.INITIATIVES_PER_PAGE = 25
        self.BASE_URL = "https://www.congreso.es/web/guest/indice-de-iniciativas"
        self.SAFETY_EXTRACTION_GAP = 3
        self.totals_by_type = dict()
        self.all_references = list()
        self.deputies = Deputy.objects()
        self.parliamentarygroups = ParliamentaryGroup.objects()
        self.places = Place.objects()
        self.grouped_deputies = GroupedDeputies()
        self.api = CongressApi()

    def get_types(self):
        return INITIATIVE_TYPES

    def sync_totals(self):
        response = self.api.get_initiative_totals()
        if not response.ok:
            raise Exception

        soup = BeautifulSoup(response.json()["content"], "lxml")
        for element in soup.select(".listado_1 li"):
            initiative_type = element.select_one("a").getText().strip()
            if initiative_type[-1] == ".":
                initiative_type = initiative_type[:-1]
            count = int(element.select_one("span").getText().strip("()"))
            self.totals_by_type[initiative_type] = count

    def extract(self):
        start_time = time.perf_counter()
        self.extract_references()

        log.info(f"Getting {len(self.all_references)} initiatives references")
        log.debug(
            "--- %s seconds getting references---" % (time.perf_counter() - start_time)
        )
        log.info("Processing initiatives...")

        start_time = time.perf_counter()
        self.extract_initiatives()
        log.debug(
            "--- %s seconds getting initiatives ---"
            % (time.perf_counter() - start_time)
        )

    def extract_all_references(self):
        self.sync_totals()
        for initiative_type in self.get_types():
            code = initiative_type["code"]
            title = initiative_type["type"]

            db_last_reference = 0
            db_total = 0
            origin_total = (
                self.totals_by_type[title] if title in self.totals_by_type else 0
            )

            new_items = origin_total - db_total
            if not new_items:
                continue

            for extra in range(1, new_items + self.SAFETY_EXTRACTION_GAP + 1):
                self.all_references.append(
                    self.format_reference(db_last_reference + extra, code)
                )

    def extract_all_references_from_type(self, type_code):
        self.sync_totals()
        for initiative_type in self.get_types():
            code = initiative_type["code"]
            if type_code != code:
                continue
            title = initiative_type["type"]

            db_last_reference = 0
            db_total = 0
            origin_total = (
                self.totals_by_type[title] if title in self.totals_by_type else 0
            )

            new_items = origin_total - db_total
            if not new_items:
                continue

            for reference in range(
                db_last_reference, origin_total + self.SAFETY_EXTRACTION_GAP
            ):
                self.all_references.append(self.format_reference(reference, code))

    def extract_references_from_type(self, type_code):
        self.sync_totals()
        initiatives = (
            Initiatives.get_all()
            .filter(initiative_type=type_code)
            .order_by("reference")
            .only("reference", "status")
        )

        last_references = {}
        totals = {}
        previous_ref = 1

        for initiative in initiatives:
            if "reference" not in initiative:
                continue
            title = initiative["initiative_type_alt"]
            if title == "Respuesta":
                continue
            items = initiative["reference"].split("/")
            initiative_type = items[0]
            reference = items[1]

            if initiative_type not in totals:
                totals[initiative_type] = 0

            last_references[initiative_type] = reference
            int_reference = int(reference)
            missing_references = self.calculate_references_between(
                previous_ref, int_reference, initiative_type
            )
            self.all_references += missing_references

            if initiative["status"] in NOT_FINAL_STATUS:
                self.all_references.append(initiative["reference"])

            totals[initiative_type] += 1 + len(missing_references)
            previous_ref = int_reference + 1

        for initiative_type in self.get_types():
            code = initiative_type["code"]
            if type_code != code:
                continue
            title = initiative_type["type"]

            db_last_reference = (
                int(last_references[code]) if code in last_references else 0
            )
            origin_total = (
                self.totals_by_type[title] if title in self.totals_by_type else 0
            )
            db_total = totals[code] if code in totals else 0

            new_items = origin_total - db_total
            if not new_items:
                continue

            for reference in range(
                db_last_reference, origin_total + self.SAFETY_EXTRACTION_GAP
            ):
                self.all_references.append(self.format_reference(reference, code))

    def extract_references(self):
        self.sync_totals()
        initiatives = (
            Initiatives.get_all_without_answers()
            .order_by("reference")
            .only("reference", "status")
        )

        last_references = {}
        totals = {}
        previous_ref = 1

        for initiative in initiatives:
            if "reference" not in initiative:
                continue
            title = initiative["initiative_type_alt"]
            if title == "Respuesta":
                continue
            items = initiative["reference"].split("/")
            initiative_type = items[0]
            reference = items[1]

            if initiative_type not in totals:
                totals[initiative_type] = 0

            last_references[initiative_type] = reference
            int_reference = int(reference)
            missing_references = self.calculate_references_between(
                previous_ref, int_reference, initiative_type
            )
            self.all_references += missing_references

            if initiative["status"] in NOT_FINAL_STATUS:
                self.all_references.append(initiative["reference"])

            totals[initiative_type] += 1
            previous_ref = int_reference + 1

        for initiative_type in self.get_types():
            code = initiative_type["code"]
            title = initiative_type["type"]

            db_last_reference = (
                int(last_references[code]) if code in last_references else 0
            )
            origin_total = (
                self.totals_by_type[title] if title in self.totals_by_type else 0
            )
            db_total = totals[code] if code in totals else 0

            new_items = int(origin_total) - int(db_total)

            if new_items < 0:
                continue

            gap = self.SAFETY_EXTRACTION_GAP

            for extra in range(1, new_items + gap):
                self.all_references.append(
                    self.format_reference(db_last_reference + extra, code)
                )

    def calculate_references_between(self, previous_ref, new_ref, initiative_type):
        missing_references = []
        while previous_ref < new_ref:
            missing_reference = self.format_reference(previous_ref, initiative_type)
            previous_ref += 1
            missing_references.append(missing_reference)

        return missing_references

    def extract_initiatives(self):
        def callback(response):
            InitiativeExtractorFactory.create(
                response,
                self.deputies,
                self.parliamentarygroups,
                self.grouped_deputies,
                self.places,
            ).extract()

        self.process_initiatives(callback)

    def process_initiatives(self, callback):
        futures_requests = list()
        for reference in self.all_references:
            futures_requests.append(self.api.get_initiative(reference))

        for future in as_completed(futures_requests):
            try:
                response = future.result()
            except Exception as e:
                log.error(f"Error {e} processing initiative")
                continue

            if response.ok:
                callback(response)
            else:
                log.error(
                    f"Error {response.status_code} processing initiative on {response.url}"
                )

    def extract_videos(self):
        for reference in self.all_references:
            extractor = VideoExtractor(reference)
            extractor.extract()

    def extract_interventions(self):
        for reference in self.all_references:
            extractor = InterventionsExtractor(reference)
            extractor.extract()

    def extract_votes(self):
        def callback(response):
            extractor = InitiativeExtractorFactory.create(response, [], [], {}, [])
            votes_extractor = VoteExtractor(
                extractor.node_tree, extractor.get_reference()
            )
            votes_extractor.extract()

        self.__skip_oversight_initiatives()
        self.process_initiatives(callback)

    def format_reference(self, ref, initiative_type):
        reference = str(ref)
        missing_zeros = 6 - len(reference)
        return initiative_type + "/" + ("0" * missing_zeros) + reference

    def __skip_oversight_initiatives(self):
        oversight_types = [
            t["code"] for t in INITIATIVE_TYPES if t["group"] == "Función de control"
        ]
        self.all_references = [
            ref
            for ref in self.all_references
            if ref.split("/")[0] not in oversight_types
        ]
