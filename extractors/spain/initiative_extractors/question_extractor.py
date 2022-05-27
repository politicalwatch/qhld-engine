from .initiative_extractor import InitiativeExtractor
from .bulletins_extractor import NonExclusiveBulletinExtractor
from .utils.pdf_parsers import PDFExtractor
from copy import deepcopy
from .initiative_status import NOT_FINAL_STATUS, ON_PROCESS, is_final_status
from tipi_data.models.initiative import Initiative
from tipi_data.utils import generate_id
from logger import get_logger


log = get_logger(__name__)


class QuestionExtractor(InitiativeExtractor):
    QUESTION = 'Pregunta'
    ANSWER = 'Contestación'
    HREF = 'href'
    A = 'a'

    def extract_content(self):
        if not self.has_content():
            self.initiative['content'] = self.retrieve_question()

        try:
            answer = Initiative.objects.get(
                reference=self.get_reference(),
                initiative_type_alt='Respuesta'
            )

            has_content = 'content' in answer
            extract_answer = (not has_content) or (has_content and len(answer['content']) == 0)
        except Exception as e:
            extract_answer = True

        if is_final_status(self.initiative['status']) and self.initiative['status'] != 'Respondida':
            extract_answer = False

        if extract_answer == True:
            answer_content = self.retrieve_answer()
            if answer_content == []:
                self.initiative['status'] = ON_PROCESS
            else:
                self.create_answer_initative(answer_content)

    def should_extract_content(self):
        return True

    def create_answer_initative(self, answer):
        if answer == []:
            return
        try:
            answer_initiative = Initiative.objects.get(
                reference=self.initiative['reference'],
                initiative_type_alt='Respuesta'
            )
            force = False
        except Exception:
            answer_initiative = deepcopy(self.initiative)
            answer_initiative['tagged'] = []
            force = True
        answer_initiative['content'] = answer
        answer_initiative['initiative_type_alt'] = 'Respuesta'
        answer_initiative['author_others'] = ['Gobierno']
        answer_initiative['author_deputies'] = []
        answer_initiative['author_parliamentarygroups'] = []
        answer_initiative['id'] = self.generate_answer_id(answer_initiative)
        answer_initiative['oldid'] = self.generate_answer_oldid(answer_initiative)
        answer_initiative.save(force_insert=force)

    def retrieve_question(self):
        link = self.find_link(self.QUESTION)
        if link == []:
            return []
        link_text = link.text_content()
        if link_text == 'Pregunta':
            return self.retrieve_content(link, True)
        if link_text == 'Pregunta (ver boletín de la iniciativa, según acuerdo de mesa)':
            bulletin_extractor = NonExclusiveBulletinExtractor(self.response, [], [], [])
            bulletin_extractor.initiative['reference'] = self.get_reference()
            bulletin_extractor.extract_content()
            return bulletin_extractor.initiative['content']

        log.error(f"Error, unkown question type found {link_text}")
        return []

    def retrieve_answer(self):
        link = self.find_link(self.ANSWER)
        if link == []:
            return []
        return self.retrieve_content(link)

    def generate_answer_id(self, initiative):
        return initiative['reference'].replace('/', '-') + '-respuesta'

    def generate_answer_oldid(self, initiative):
        return generate_id(
                initiative['reference'],
                initiative['initiative_type_alt']
                )


    def retrieve_content(self, link_tag, is_img = False):
        url = link_tag.get(self.HREF)
        extractor = PDFExtractor(url, is_img)
        return extractor.retrieve()

    def find_link(self, content):
        items = self.node_tree.xpath(f"//section[@id='portlet_iniciativas']//a[contains(normalize-space(text()), '{content}')]")
        if len(items) == 0:
            return []
        return items[0]
