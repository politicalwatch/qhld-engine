import math
import json
import re
import datetime
import sys
import regex
import os

from tipi_data.utils import generate_id
from ..congress_api import CongressApi
from .utils.pdf_parsers import PDFExtractor


class InterventionsExtractor:

    def __init__(self, initiative):
        self.initiative = initiative
        self.api = CongressApi()
        self.url = "/public_oficiales/"
        self.intervention_list = {}
        self.pdfs = {}
        self.headers = []
        self.sesion = ""
        self.regex_patterns = self.get_regex_for_interrupters()
        self.all_matches = {}
        self.sesion_matches = {}

    def extract(self):
        print(f"Getting interventions from {self.initiative}")
        page = 1
        while True:
            interventions = self.retrieve_json(page)

            if (
                not interventions
                or "error" in interventions
                or "intervenciones_encontradas" not in interventions
            ):
                break

            # We order the interventions from the page
            interventions = sorted(
                interventions.get("lista_intervenciones", {}).items(),
                key=lambda x: int(x[1]["doc"]),
            )

            # We create search regex and look for matches in sesion pdfs
            self.get_sesion_speakers(interventions)
            self.get_sesion_pdfs(interventions)
            self.find_all_matches()

            for _, intervention in interventions:
                # To escape votes
                if "tipo_intervencion" in intervention:
                    continue

                self.process_intervention(intervention)

            # TODO: Check pagination on api, not working. Only processing first page (page += 1)
            break
            sys.exit()
        self.save_to_file("sesion.json", self.intervention_list)

    def get_sesion_speakers(self, interventions):
        for _, intervention in interventions:
            if "tipo_intervencion" in intervention:
                continue
            speaker = self.get_speaker_key(intervention)
            regex_pattern = self.get_regex_for_speaker(speaker)
            self.regex_patterns[speaker] = {}
            if regex_pattern not in self.regex_patterns[speaker]:
                self.regex_patterns[speaker] = regex_pattern

        self.regex_patterns.update(self.get_regex_for_interrupters())

    def process_intervention(self, intervention):
        if self.initiative not in self.intervention_list:
            self.intervention_list[self.initiative] = []
        self.intervention_list[self.initiative].append(self._intervention(intervention))

    def find_all_matches(self):
        for key, pdf in self.pdfs.items():
            self.all_matches[key] = []

            for speaker, pattern in self.regex_patterns.items():
                print(pattern)
                matches = regex.finditer(pattern, pdf, regex.IGNORECASE)
                self.all_matches[key].extend(matches)

                for match in matches:
                    print(match.group())

            self.all_matches[key] = sorted(
                self.all_matches[key], key=lambda match: match.start()
            )
            # print(self.all_matches)

    def _intervention(self, intervention):
        speaker_match = re.match(r"(.*)\s+\((.*)\)", intervention["orador"])

        if not speaker_match:
            return

        speaker_name = speaker_match.group(1)
        speaker_surname = speaker_name.split(",")[0]
        legislature = intervention["video_intervencion"]["legislatura"]
        sesion_link = f"{self.url}L{legislature}/{intervention['pdia']}"
        sesion_link_upd = sesion_link.split("#")[0]
        if not self.sesion_matches:
            self.sesion_matches = self.all_matches[sesion_link_upd]

        return {
            "speaker": speaker_name,
            "speaker_surname": speaker_surname,
            "order": intervention["doc"],
            "legislature": legislature,
            "video_link": intervention["video_intervencion"]["enlace_descarga02"],
            "sesion_link": sesion_link,
            "speech": self.extract_speech(speaker_surname, sesion_link_upd),
        }

    def extract_speech(self, speaker_surname, sesion_link):
        speech = ""
        left_matches = self.sesion_matches.copy()
        speaker_found = False

        sesion = self.pdfs[sesion_link]

        for match in self.sesion_matches:
            if not speaker_found:
                if speaker_surname.lower() in match.group().lower():
                    # print("Speaker found 1st time")
                    speech_start = match.end()
                    speaker_found = True

                left_matches.remove(match)
            else:
                if match.re.pattern in self.get_regex_for_interrupters().values():
                    # print("Es interrupción")
                    speech_end = match.start()
                    speech += sesion[speech_start:speech_end]
                    left_matches.remove(match)
                    continue

                if speaker_surname.lower() not in match.group().lower():
                    # print("Siguiente speaker")
                    break
                else:
                    # print("Continua despues de interrupción")
                    speech_start = match.end()
                    left_matches.remove(match)

        self.sesion_matches = left_matches.copy()

        return self.clean_interventions(speech)

    def get_sesion_pdfs(self, interventions):
        for intervention in interventions:
            sesion_link = self.get_sesion_link(intervention)
            self.get_sesion(sesion_link)

    def get_sesion_link(self, intervention):
        legislature = intervention[1]["video_intervencion"]["legislatura"]
        pdia = intervention[1]["pdia"].split("#")[0]
        return f"{self.url}L{legislature}/{pdia}"

    def get_sesion(self, sesion_link):
        sesion = self.pdfs.get(sesion_link) or PDFExtractor(sesion_link).retrieve()
        self.pdfs[sesion_link] = re.sub(r"\s+", " ", sesion)

    def is_same_document(self, current, next_):
        return (
            current["sesion_link"].split("#")[0] == next_["sesion_link"].split("#")[0]
        )

    def clean_interventions(self, text):
        for removable_regex in self.get_regex_for_removables():
            text = re.sub(removable_regex, "", text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    def get_regex_for_speaker(self, speaker_name):
        speaker_name = re.sub(r"\s+", r"\\s+", speaker_name)
        return rf"(La señora|El señor)\s+[a-zá-ú\s+]*[(]?{speaker_name}[)]?:"

    def get_regex_for_interrupters(self):
        return {
            "presidente": r"(La señora presidenta|El señor presidente):",
            "vicepresidente": r"(La señora VICEPRESIDENTA|El señor VICEPRESIDENTE)\s+?([a-zá-ú\s+]*[()])*:",
        }

    def get_speaker_key(self, intervention):
        # speaker_match = re.match(r"(.*)\s+\((.*)\)", intervention["orador"])
        speaker_name = intervention["orador"].split("(")[0].strip()

        if not speaker_name:
            return

        return speaker_name.split(",")[0]

    def get_regex_for_removables(self):
        return self.headers + [r"[(]rumores[)].?", r"[(]aplausos[)].?", r"\n"]

    def retrieve_json(self, page):
        try:
            response = self.api.get_video(self.initiative, page)
            return response.json()
        except json.JSONDecodeError:
            print(f"Error getting video for {self.initiative}")

    def set_headers(self, pdia):
        header_regex = (
            r"\s".join(pdia.split("#")[0].split("/")[-1].split(".")[0][::-1])
            + r"\s*:\se\sv\sc\s?[a-zá-ú-0-9.\s+]*Pág\.\s+*\d*"
        )
        if header_regex not in self.headers:
            self.headers.append(header_regex)

    def save_to_file(self, filename, content):
        updated_interventions = content
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                interventions = json.load(file)
                updated_interventions = {**interventions, **content}

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(updated_interventions, file, indent=4, ensure_ascii=False)
