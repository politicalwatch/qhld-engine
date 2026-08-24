from datetime import datetime

from qhld_engine.logger import get_logger

from tipi_data.models.footprint import FootprintByTopic, \
        FootprintByDeputy, \
        FootprintByParliamentaryGroup, \
        FootprintElement
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.deputies import Deputies
from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
from tipi_data.repositories.initiatives import Initiatives
from tipi_data.repositories.footprints import Footprints

from .footprint_managers import FootprintDeputyManager, \
        inactivity_penalty, \
        topic_scores_pipeline, \
        global_scores_pipeline, \
        topic_last_dates_pipeline


log = get_logger(__name__)


class ComputeFootprint:

    def __init__(self):
        log.info("Initializing footprint...")
        self.today = datetime.today()
        self.topics = []
        for kb in KnowledgeBases.get_all():
            self.topics += Topics.by_kb(kb)
        self.deputies = Deputies.get_all()
        self.parliamentarygroups = ParliamentaryGroups.get_all()
        log.info("Footprint initialization finished.")

    def compute(self):
        log.info("Starting footprint computation...")

        # --- 1. Aggregate everything in a fixed number of passes ---------
        log.info("Aggregating scores...")
        dep_topic = self.__topic_scores('deputy')
        grp_topic = self.__topic_scores('parliamentarygroup')
        dep_topic_last = self.__topic_last_dates('deputy')
        grp_topic_last = self.__topic_last_dates('parliamentarygroup')
        dep_global = self.__global_scores('deputy')
        grp_global = self.__global_scores('parliamentarygroup')

        # --- 2. Per-topic footprints (+ feed per-entity topic lists) -----
        deputy_fp = {
                d['id']: FootprintByDeputy(id=d['id'], name=d['name'], score=0.0, topics=list())
                for d in self.deputies
                }
        deputy_by_name = {d['name']: deputy_fp[d['id']] for d in self.deputies}
        group_fp = {
                g['id']: FootprintByParliamentaryGroup(id=g['id'], name=g['name'], score=0.0, topics=list())
                for g in self.parliamentarygroups
                }
        group_by_name = {g['name']: group_fp[g['id']] for g in self.parliamentarygroups}

        for topic in self.topics:
            initial = datetime.now()
            topic_footprint = FootprintByTopic(id=topic['id'], name=topic['name'])

            dep_els = self.__topic_elements(
                    self.deputies, topic['name'], dep_topic, dep_topic_last, 'deputy')
            grp_els = self.__topic_elements(
                    self.parliamentarygroups, topic['name'], grp_topic, grp_topic_last,
                    'parliamentarygroup')

            for el in dep_els:
                deputy_by_name[el['name']]['topics'].append(
                        FootprintElement(name=topic['name'], score=el['score']))
            for el in grp_els:
                group_by_name[el['name']]['topics'].append(
                        FootprintElement(name=topic['name'], score=el['score']))

            topic_footprint['deputies'] = self.__sort_scores(dep_els)
            topic_footprint['parliamentarygroups'] = self.__sort_scores(grp_els)

            Footprints.save_topic(topic_footprint)
            log.info(f"{topic['name'].upper()}: footprint computed in "
                     f"{(datetime.now() - initial).seconds} seconds.")

        # --- 3. Global (topic-agnostic) scores per entity ----------------
        self.__finalize_global(self.deputies, deputy_fp, dep_global, 'deputy')
        self.__finalize_global(
                self.parliamentarygroups, group_fp, grp_global, 'parliamentarygroup')

        for fp in deputy_fp.values():
            fp['topics'] = self.__sort_scores(fp['topics'])
            Footprints.save_deputy(fp)
        for fp in group_fp.values():
            fp['topics'] = self.__sort_scores(fp['topics'])
            Footprints.save_parliamentarygroup(fp)

        log.info("Footprint computation finished.")

    # --- Aggregation wrappers ---------------------------------------------

    def __topic_scores(self, typeof):
        """{(entity_name, topic_name): raw_score}"""
        return {
                (r['_id']['e'], r['_id']['t']): r['score']
                for r in Initiatives.aggregate(topic_scores_pipeline(typeof))
                if r['_id'].get('e') is not None and r['_id'].get('t') is not None
                }

    def __topic_last_dates(self, typeof):
        """{(entity_name, topic_name): last_valid_created}"""
        return {
                (r['_id']['e'], r['_id']['t']): r['last']
                for r in Initiatives.aggregate(topic_last_dates_pipeline(typeof))
                if r['_id'].get('e') is not None and r['_id'].get('t') is not None
                }

    def __global_scores(self, typeof):
        """{entity_name: {'score': raw_score, 'last': last_valid_created}}"""
        return {
                r['_id']: {'score': r['score'], 'last': r.get('last')}
                for r in Initiatives.aggregate(global_scores_pipeline(typeof))
                if r['_id'] is not None
                }

    # --- Score finalization -----------------------------------------------

    def __finalize(self, raw, typeof, entity, last_date):
        """Apply the deputy contact bonus and the inactivity penalty to a raw score,
        replicating the original per-cell logic (bonus/penalty only when raw > 0)."""
        if raw <= 0:
            return 0.0
        score = raw
        if typeof == 'deputy':
            fdm = FootprintDeputyManager(entity)
            score += fdm.compute_email()
            score += fdm.compute_social()
        penalty = inactivity_penalty(last_date, self.today)
        return round(score - (score * penalty), 2)

    def __topic_elements(self, entities, topic_name, scores, last_dates, typeof):
        elements = [
                FootprintElement(
                    name=e['name'],
                    score=self.__finalize(
                        scores.get((e['name'], topic_name), 0.0),
                        typeof,
                        e,
                        last_dates.get((e['name'], topic_name)),
                        ))
                for e in entities
                ]
        self.__normalize_topic_scores(elements)
        return elements

    def __finalize_global(self, entities, footprints, global_scores, typeof):
        raw = dict()
        for e in entities:
            data = global_scores.get(e['name'], {})
            raw[e['id']] = self.__finalize(
                    data.get('score', 0.0), typeof, e, data.get('last'))
        if raw:
            self.__normalize_scores(raw)
        for e in entities:
            footprints[e['id']]['score'] = raw[e['id']]

    # --- Normalization / sorting (unchanged) ------------------------------

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
        if not scores:
            return
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
