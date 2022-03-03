from datetime import datetime

from logger import get_logger

from tipi_data.models.footprint import FootprintByTopic, \
        FootprintByDeputy, \
        FootprintElement
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.deputies import Deputies

from .footprint_managers import FootprintSumPointOneManager, \
        FootprintSumTwoManager, \
        FootprintSumFiveManager, \
        FootprintSumTwentyManager, \
        FootprintSumFourtyManager, \
        FootprintAdditionalFiveManager, \
        FootprintAdditionalTwentyManager, \
        FootprintInactivityPenalty, \
        FootprintDeputyManager


log = get_logger(__name__)


class ComputeFootprint:

    def __init__(self):
        log.info("Initiatizing footprint...")
        self.topics = {}
        self.knowledgebases = list(KnowledgeBases.get_all())
        for kb in self.knowledgebases:
            self.topics[kb] = Topics.by_kb(kb)
        self.deputies = Deputies.get_all()
        self.footprint_by_deputies = self.__initiatize_footprint_by_deputies()
        log.info("Footprint initiatization finished.")

    def compute(self):
        log.info("Starting footprint computation...")

        for topic in self.topics['politicas']:
            log.info(f"{topic['name'].upper()}: Computing footprint...")
            initial = datetime.now()

            topic_footprint = FootprintByTopic()
            topic_footprint['id'] = topic['id']
            topic_footprint['name'] = topic['name']
            topic_footprint['deputies'] = list()

            for deputy in self.deputies:
                score = self.compute_deputy_by_topic(deputy, topic)
                topic_footprint['deputies'].append(FootprintElement(
                    name=deputy['name'],
                    score=float(score)
                    ))
                self.__add_footprint_by_deputy(deputy, topic, score)

            topic_footprint['deputies'] = self.__sort_scores(topic_footprint['deputies'])
            topic_footprint.save()
            log.info(f"{topic['name'].upper()}: footprint computed in {(datetime.now() - initial).seconds} seconds.")

        self.__save_footprint_by_deputies()
        log.info("Footprint computation finished.")

    def compute_deputy_by_topic(self, deputy, topic):
        score = 0
        deputy_name = deputy['name']
        topic_name = topic['name'] if topic else None

        score += FootprintSumPointOneManager(
                topic_name,
                deputy_name).compute()
        score += FootprintSumTwoManager(
                topic_name,
                deputy_name).compute()
        score += FootprintSumFiveManager(
                topic_name,
                deputy_name).compute()
        score += FootprintSumTwentyManager(
                topic_name,
                deputy_name).compute()
        score += FootprintSumFourtyManager(
                topic_name,
                deputy_name).compute()
        score += FootprintAdditionalFiveManager(
                topic_name,
                deputy_name).compute()
        score += FootprintAdditionalTwentyManager(
                topic_name,
                deputy_name).compute()

        if score > 0:
            fdm = FootprintDeputyManager(deputy)
            score += fdm.compute_email()
            score += fdm.compute_social()

            penalty = FootprintInactivityPenalty(
                    topic_name,
                    deputy_name).compute()
            score = f"{score - (score * penalty):.2f}"

        return score

    def __sort_scores(self, lst):
        return sorted(lst, key=lambda element: float(element['score']), reverse=True)

    def __initiatize_footprint_by_deputies(self):
        global_score = dict()
        for d in self.deputies:
            global_score[d['id']] = self.compute_deputy_by_topic(d, None)

        return [
                FootprintByDeputy(
                    id=d['id'],
                    name=d['name'],
                    score=global_score[d['id']],
                    topics=list())
                for d in self.deputies
                ]

    def __add_footprint_by_deputy(self, deputy, topic, score):
        deputy_footprint = self.__get_deputy__footprint(deputy)
        if not deputy_footprint:
            return
        deputy_footprint['topics'].append(FootprintElement(
            name=topic['name'],
            score=score
            ))

    def __get_deputy__footprint(self, deputy):
        deputy_footprint = [d for d in self.footprint_by_deputies if d['id'] == deputy['id']]
        if len(deputy_footprint) > 0:
            return deputy_footprint[0]
        return None

    def __save_footprint_by_deputies(self):
        for fbd in self.footprint_by_deputies:
            fbd['topics'] = self.__sort_scores(fbd['topics'])
            fbd.save()


if __name__ == "__main__":
    ComputeFootprint().compute()
