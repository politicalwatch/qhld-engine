import csv
import json
import re
import sys
import regex
import os

from ..congress_api import CongressApi
from .utils.pdf_parsers import PDFExtractor
from thefuzz import fuzz

class InterventionsExtractor:

    def __init__(self, initiative):
        self.initiative = initiative
        self.api = CongressApi()
        self.url = "/public_oficiales/"
        self.intervention_list = {}
        self.sesions = {}
        self.sesion = ""
        self.regex_removables = self.get_regex_for_removables()
        self.speakers = []
        self.speaker_pattern = r"(El señor|La señora)\s+[a-zá-ú\s]+(\s\([a-zá-ú\s]+\))?:"
        self.cvs_file = 'sesions.csv'

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

            interventions = sorted(
                (
                    (k, v) for k, v in interventions.get("lista_intervenciones", {}).items()
                    if "tipo_intervencion" not in v  # To exclude votes
                ),
                key=lambda x: int(x[1]["doc"]),
            )
            
            self.get_speakers(interventions)
            
            for _, intervention in interventions:
                data = self.process_intervention(intervention)
                if data:
                    self.save_to_csv(self.initiative, data)
                else:
                    print(f'Error en extracción de: {self.initiative}')
                if self.initiative not in self.intervention_list:
                    self.intervention_list[self.initiative] = []
                self.intervention_list[self.initiative].append(data)
                
            break ## Revisar paginación
            
            
    def process_intervention(self, intervention):
        # A REVISAR LA SESIÓN GUARDADA ENTRE INTERVENCIONES E INICIATIVAS
        if not self.sesion:
            self.sesion = self.get_sesion(intervention)
        speaker_regex = self.get_speaker_regex(intervention)
        
        speaker_match = re.match(r"(.*)\s+\((.*)\)", intervention["orador"])
        if not speaker_match:
            return

        speaker_name = speaker_match.group(1)
        speaker_surname = speaker_name.split(",")[0]
        legislature = intervention["video_intervencion"]["legislatura"]
        sesion_link = f"{self.url}L{legislature}/{intervention['pdia']}"

        # TODO: Si el vídeo contiene doble slash //, significa que el enlace no es
        #       En esos casos, se puede recuperar el inicio y fin de intervención y extaraer del de la sesión completa esa extracto
        return {
            "speaker": speaker_name,
            "speaker_surname": speaker_surname,
            "order": intervention["doc"],
            "legislature": legislature,
            "video_link": intervention["video_intervencion"]["enlace_descarga02"],
            "sesion_link": sesion_link,
            "speech": self.extract_speech(speaker_regex),
        }

    def extract_speech(self, speaker_regex):
        print("Extraemos speech de: " + speaker_regex)
        match_speaker = regex.search(speaker_regex, self.sesion, flags=re.IGNORECASE)
        if not match_speaker:
            return 'No encontrado orador'
        self.sesion = self.sesion[match_speaker.end():]
        match_pattern = regex.search(self.speaker_pattern, self.sesion, flags=re.IGNORECASE)
        if not match_pattern:
            return 'No encontrado fin'
        speech = self.sesion[:match_pattern.start()]
        self.sesion = self.sesion[match_pattern.end():]

        while self.is_interrupter(match_pattern):
            match_speaker = regex.search(self.speaker_pattern, self.sesion, flags=re.IGNORECASE)
            print(regex.fullmatch(speaker_regex, match_speaker.group(0), flags=re.IGNORECASE) is None)
            if regex.fullmatch(speaker_regex, match_speaker.group(0), flags=re.IGNORECASE) is None:
                break # No sigue hablando tras interrupción
            self.sesion = self.sesion[match_speaker.end():]
            match_pattern = regex.search(self.speaker_pattern, self.sesion, flags=re.IGNORECASE)
            speech = speech + self.sesion[:match_pattern.start()]
            self.sesion = self.sesion[match_pattern.start():]

        return self.clean_interventions(speech)
    
    def get_speaker_regex(self, intervention):
        speaker_surname = intervention["orador"].split("(")[0].strip()

        if not speaker_surname:
            return

        speaker_surname = re.sub(r"\s+", r"\\s+", speaker_surname.split(",")[0])

        return rf"(?<!\()(La señora|El señor)\s+[a-zá-ú\s+]*[(]?{speaker_surname}[)]?:"    
    
    def get_sesion(self, intervention):
        sesion_link = self.get_sesion_link(intervention)
        if sesion_link not in self.sesions:
            return self.get_sesion_pdf(sesion_link)
        
        return self.sesions[sesion_link]

    def get_sesion_link(self, intervention):
        legislature = intervention["video_intervencion"]["legislatura"]
        pdia = intervention["pdia"].split("#")[0]

        return f"{self.url}L{legislature}/{pdia}"
    
    def get_sesion_pdf(self, sesion_link):
        sesion = self.sesions.get(sesion_link) or PDFExtractor(sesion_link).retrieve()
        pattern = rf"\s\(Número\s+de\s+expediente\s{self.initiative}\)\.\s"
        matches = list(re.finditer(pattern, sesion))

        if matches:
            sesion = sesion[matches[-1].end():]
        self.sesions[sesion_link] = re.sub(r"\s+", " ", sesion).replace("‑", "-").replace("–", "-").replace("—", "-")
        # Fixing of typos on speakers surname
        self.sesions[sesion_link] = self.fix_sesion_typos(self.sesions[sesion_link])

        return self.sesions[sesion_link]
    
    def clean_interventions(self, text):
        for removable_regex in self.get_regex_for_removables():
            text = regex.sub(removable_regex, "", text, 0, flags=re.IGNORECASE | re.MULTILINE)
        return text
    
    def get_speakers(self, interventions):
        for _, intervention in interventions:
            name = intervention["orador"].split("(")[0].strip()
            self.speakers.append(name.split(",")[0].strip().upper())
    
    def retrieve_json(self, page):
        try:
            response = self.api.get_video(self.initiative, page)
            return response.json()
        except json.JSONDecodeError:
            print(f"Error getting video for {self.initiative}")    
    
    def is_interrupter(self, match):
        match_text = match.group(0)  
        interrupters_patterns = self.get_regex_for_interrupters().values()

        return any(regex.fullmatch(pattern, match_text, flags=regex.IGNORECASE) for pattern in interrupters_patterns)
    
    def get_next_interruption(self):
        for pattern in self.get_regex_for_interrupters().values():
            match = regex.search(pattern, self.sesion, flags=regex.IGNORECASE)
            if match:
                return match

        return None
    
    def get_regex_for_interrupters(self):
        return {
            "presidente": r"(La señora presidenta|El señor presidente):",
            "vicepresidente": r"(La señora VICEPRESIDENTA|El señor VICEPRESIDENTE)\s+?([a-zá-ú\s+]*[()])*:",
        }
    
    def get_regex_for_removables(self):
        return [
            r"[(]rumores[)].?",
            r"[(]aplausos[)].?",
            r"\n"
            r"^\d+(?: \d+)* (?:Idem\.( )?)*",
            r"\d+ En aplicación del punto Tercero\.7 del Acuerdo de la Mesa del Congreso de los Diputados relativo al régimen lingüístico de los debates en los órganos parlamentarios\. ",
            r"\d+(\s\d+)*\s-\s?[A-Z\s]+\s-\s\d+(\s\d+)*\s-\s[A-Z\s]+\s[A-Z]+\s[A-Z]+\s?:\s?.*?DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS.*?Pág\.\s?\d+\s"
        ]
        
    def fix_sesion_typos(self, sesion, threshold=80):
        matches = regex.finditer(self.speaker_pattern, sesion, flags=re.IGNORECASE)
        for match in matches:
            if self.is_interrupter(match):
                continue
            for speaker in self.speakers:
                similarity = fuzz.ratio(match.group(2), speaker)
                if similarity > threshold and similarity < 100:
                    sesion = sesion.replace(match.group(2), speaker)
                    break

        return sesion
    
    def save_to_csv(self, initiative_id, data):
        file_exists = os.path.exists(self.cvs_file)

        with open(self.cvs_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["initiative_id", "legislature", "speaker", "status", "error_desc", "video_link", "speech"])

            writer.writerow([
                initiative_id,
                data.get("legislature", ""),
                data.get("speaker", ""),
                "",  # Campo status vacío
                "",  # Campo error_desc vacío
                data.get("video_link", ""),
                data.get("speech", "").strip()  # Eliminar espacios extra en el discurso
            ])