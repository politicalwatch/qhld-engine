from datetime import datetime

from logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from tipi_data.models.footprint import FootprintByTopic, \
        FootprintByDeputy, \
        FootprintByParliamentaryGroup, \
        FootprintElement
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups

from .footprint_managers import FootprintSumFourManager, \
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
        log.info("Initializing footprint...")
        self.topics = []
        self.knowledgebases = list(KnowledgeBases.get_all())
        for kb in self.knowledgebases:
            self.topics += Topics.by_kb(kb)
        self.deputies = Deputies.get_all()
        self.footprint_by_deputies = self.__initialize_footprint_by_deputies()
        self.parliamentarygroups = ParliamentaryGroups.get_all()
        self.footprint_by_parliamentarygroups = self.__initialize_footprint_by_parliamentarygroups()
        log.info("Footprint initialization finished.")

    def compute(self):
        log.info("Starting footprint computation...")

        for topic in self.topics:
            log.info(f"{topic['name'].upper()}: Computing footprint...")
            initial = datetime.now()

            topic_footprint = FootprintByTopic()
            topic_footprint['id'] = topic['id']
            topic_footprint['name'] = topic['name']

            self.__compute_topic_by_entity(
                    topic,
                    topic_footprint,
                    entity_list=self.deputies,
                    entity_keys=('deputy', 'deputies')
                    )
            self.__compute_topic_by_entity(
                    topic,
                    topic_footprint,
                    entity_list=self.parliamentarygroups,
                    entity_keys=('parliamentarygroup', 'parliamentarygroups')
                    )

            topic_footprint.save()
            log.info(f"{topic['name'].upper()}: footprint computed in {(datetime.now() - initial).seconds} seconds.")

        self.__save_footprint_by_deputies()
        self.__save_footprint_by_parliamentarygroups()
        log.info("Footprint computation finished.")

    def __compute_topic_by_entity(self, topic, topic_footprint, entity_list, entity_keys):
        topic_footprint[entity_keys[1]] = list()
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_entity = {
                    executor.submit(self.__compute_by_topic, e, topic, entity_keys[0]): e
                    for e in entity_list
                    }
            for future in as_completed(future_to_entity):
                e = future_to_entity[future]
                try:
                    topic_footprint[entity_keys[1]].append(FootprintElement(
                        name=e['name'],
                        score=float(future.result())
                        ))
                except Exception as e:
                    log.error(f"Cannot generate footprint by {entity_keys[0]} {e} for topic {topic}: {e}")

        self.__normalize_topic_scores(topic_footprint[entity_keys[1]])
        topic_footprint[entity_keys[1]] = self.__sort_scores(topic_footprint[entity_keys[1]])

        add_method = f"__add_footprint_by_{entity_keys[0]}"
        for entity in entity_list:
            getattr(self, f"_{self.__class__.__name__}{add_method}", None)(
                entity,
                topic,
                list(filter(
                    lambda x: x['name'] == entity['name'],
                    topic_footprint[entity_keys[1]]
                    ))[0]['score']
                )

    def __compute_by_topic(self, entity, topic, typeof):
        score = 0
        entity_name = entity['name']
        topic_name = topic['name'] if topic else None

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

    def __initialize_footprint_by_deputies(self):
        global_score = dict()

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_deputy = {
                    executor.submit(self.__compute_by_topic, d, None, 'deputy'): d
                    for d in self.deputies
                    }
            for future in as_completed(future_to_deputy):
                d = future_to_deputy[future]
                try:
                    global_score[d['id']] = float(future.result())
                except Exception as e:
                    log.error(f"Cannot generate footprint by deputy {d}: {e}")

        self.__normalize_scores(global_score)

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

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_group = {
                    executor.submit(self.__compute_by_topic, g, None, 'parliamentarygroup'): g
                    for g in self.parliamentarygroups
                    }
            for future in as_completed(future_to_group):
                g = future_to_group[future]
                try:
                    global_score[g['id']] = float(future.result())
                except Exception as e:
                    log.error(f"Cannot generate footprint by group {d}: {e}")

        self.__normalize_scores(global_score)

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

    def __normalize_scores(self, scores):
        max_score = max(score for score in scores.values())
        min_score = min(score for score in scores.values())
        distance = max_score - min_score
        if distance == 0:
            distance = 1
        for key, score in scores.items():
            normalized_score = (score - min_score) * 100 / distance
            scores[key] = round(normalized_score, 2)

    def __normalize_topic_scores(self, scores):
        max_score = max(item['score'] for item in scores)
        min_score = min(item['score'] for item in scores)
        distance = max_score - min_score
        if distance == 0:
            distance = 1
        for item in scores:
            normalized_score = (item['score'] - min_score) * 100 / distance
            item['score'] = round(normalized_score, 2)

    def __sort_scores(self, lst):
        return sorted(lst, key=lambda element: float(element['score']), reverse=True)


if __name__ == "__main__":
    ComputeFootprint().compute()
