from datetime import datetime

from logger import get_logger

from tipi_data.models.footprint import FootprintByTopic, \
        FootprintByDeputy, \
        FootprintByParliamentaryGroup, \
        FootprintElement
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

from .footprint_managers import FootprintSumPointOneManager, \
        FootprintSumFourManager, \
        FootprintSumTenManager, \
        FootprintSumFourtyManager, \
        FootprintSumEightyManager, \
        FootprintAdditionalTwentyManager, \
        FootprintAdditionalSixtyManager, \
        FootprintInactivityPenalty, \
        FootprintDeputyManager


log = get_logger(__name__)


class ComputeFootprint:

    def __init__(self):
        log.info("Initiatizing footprint...")
        self.topics = []
        self.knowledgebases = list(KnowledgeBases.get_all())
        for kb in self.knowledgebases:
            self.topics += Topics.by_kb(kb)
        self.deputies = Deputies.get_all()
        self.footprint_by_deputies = self.__initialize_footprint_by_deputies()
        self.parliamentarygroups = ParliamentaryGroups.get_all()
        self.footprint_by_parliamentarygroups = self.__initialize_footprint_by_parliamentarygroups()
        log.info("Footprint initiatization finished.")

    def compute(self):
        log.info("Starting footprint computation...")

        for topic in self.topics:
            log.info(f"{topic['name'].upper()}: Computing footprint...")
            initial = datetime.now()

            topic_footprint = FootprintByTopic()
            topic_footprint['id'] = topic['id']
            topic_footprint['name'] = topic['name']
            topic_footprint['deputies'] = list()

            for group in self.parliamentarygroups:
                score = self.compute_by_topic(group, topic, 'parliamentarygroup')
                topic_footprint['parliamentarygroups'].append(FootprintElement(
                    name=group['name'],
                    score=float(score)
                    ))
                self.__add_footprint_by_parliamentarygroup(group, topic, score)
            topic_footprint['parliamentarygroups'] = self.__sort_scores(topic_footprint['parliamentarygroups'])

            for deputy in self.deputies:
                score = self.compute_by_topic(deputy, topic, 'deputy')
                topic_footprint['deputies'].append(FootprintElement(
                    name=deputy['name'],
                    score=float(score)
                    ))
                self.__add_footprint_by_deputy(deputy, topic, score)

            topic_footprint['deputies'] = self.__sort_scores(topic_footprint['deputies'])

            topic_footprint.save()
            log.info(f"{topic['name'].upper()}: footprint computed in {(datetime.now() - initial).seconds} seconds.")

        self.__save_footprint_by_parliamentarygroups()
        self.__save_footprint_by_deputies()
        log.info("Footprint computation finished.")

    def compute_by_topic(self, entity, topic, typeof):
        score = 0
        entity_name = entity['name']
        topic_name = topic['name'] if topic else None

        score += FootprintSumPointOneManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintSumFourManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintSumTenManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintSumFourtyManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintSumEightyManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintAdditionalTwentyManager(
                topic_name,
                entity_name,
                typeof).compute()
        score += FootprintAdditionalSixtyManager(
                topic_name,
                entity_name,
                typeof).compute()

        if score > 0 and typeof == 'deputy':
            fdm = FootprintDeputyManager(entity)
            score += fdm.compute_email()
            score += fdm.compute_social()

        if score > 0:
            penalty = FootprintInactivityPenalty(
                    topic_name,
                    entity_name,
                    typeof).compute()
            score = f"{score - (score * penalty):.2f}"


        return score

    def __sort_scores(self, lst):
        return sorted(lst, key=lambda element: float(element['score']), reverse=True)

    def __initialize_footprint_by_deputies(self):
        global_score = dict()
        for d in self.deputies:
            global_score[d['id']] = self.compute_by_topic(d, None, 'deputy')

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

    def __initialize_footprint_by_parliamentarygroups(self):
        global_score = dict()
        for g in self.parliamentarygroups:
            global_score[g['id']] = self.compute_by_topic(g, None, 'parliamentarygroup')

        return [
                FootprintByParliamentaryGroup(
                    id=g['id'],
                    name=g['name'],
                    score=global_score[g['id']],
                    topics=list())
                for g in self.parliamentarygroups
                ]

    def __add_footprint_by_parliamentarygroup(self, group, topic, score):
        group_footprint = self.__get_group__footprint(group)
        if not group_footprint:
            return
        group_footprint['topics'].append(FootprintElement(
            name=topic['name'],
            score=score
            ))

    def __get_group__footprint(self, group):
        group_footprint = [g for g in self.footprint_by_parliamentarygroups if g['id'] == group['id']]
        if len(group_footprint) > 0:
            return group_footprint[0]
        return None

    def __save_footprint_by_parliamentarygroups(self):
        for fbpg in self.footprint_by_parliamentarygroups:
            fbpg['topics'] = self.__sort_scores(fbpg['topics'])
            fbpg.save()


if __name__ == "__main__":
    ComputeFootprint().compute()
