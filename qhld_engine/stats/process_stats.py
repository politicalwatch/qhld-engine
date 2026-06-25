from os import environ as env
from datetime import datetime, timedelta

from tipi_data.models.stats import Stats as StatsModel
from tipi_data.repositories.stats import Stats
from tipi_data.repositories.initiatives import Initiatives
from tipi_data.repositories.initiativetypes import InitiativeTypes
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.footprints import Footprints

from qhld_engine.extractors.config import MODULE_EXTRACTOR


class GenerateStats(object):

    def __init__(self):
        self.topics = {}
        self.subtopics = {}
        self.knowledgebases = list(KnowledgeBases.get_all())

        for kb in self.knowledgebases:
            self.topics[kb] = Topics.by_kb(kb)
            self.subtopics[kb] = Topics.get_subtopics_by_kb(kb)
        self.stats = StatsModel()

    def generate(self):
        self.overall()
        if MODULE_EXTRACTOR == 'spain':
            self.last_days()
        self.deputies_by_topics()
        self.deputies_by_subtopics()
        self.parliamentarygroups_by_topics()
        self.parliamentarygroups_by_subtopics()
        self.places_by_topics()
        self.places_by_subtopics()
        self.by_week()
        self.topics_by_week()

        Stats.delete_all()
        Stats.save(self.stats)

    def overall(self):
        self.stats['overall'] = {
                'allinitiatives': Initiatives.count_by_query({}),
                'topics': {},
                'subtopics': {},
                }
        for kb in self.knowledgebases:
            self.stats['overall'][kb] = Initiatives.count_by_kb(kb)
            self.stats['overall']['topics'][kb] = list()
            self.stats['overall']['subtopics'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$match': {'tagged.tags.topic': topic.name}},
                    {'$group': {'_id': topic.name, 'initiatives': {'$sum': 1}}}
                ]
                results = Initiatives.aggregate_by_kb(kb, pipeline)
                for item in results:
                    self.stats['overall']['topics'][kb].append(item)

            for subtopic in self.subtopics[kb]:
                pipeline = [
                    {'$match': {'tagged.tags.subtopic': subtopic}},
                    {'$group': {'_id': subtopic, 'initiatives': {'$sum': 1}}}
                    ]
                results = Initiatives.aggregate_by_kb(kb, pipeline)
                for item in results:
                    self.stats['overall']['subtopics'][kb].append(item)
            self.stats['overall']['subtopics'][kb].sort(key=lambda x: x['initiatives'], reverse=True)

    def last_days(self):
        GROUPED_TYPES = [
                ('legislative', 'Función legislativa'),
                ('orientation', 'Función de orientación política'),
                ('oversight', 'Función de control'),
                ]
        UP = 'up'
        DOWN = 'down'
        EQUAL = 'equal'
        DAYS_INTERVAL = 15
        TODAY = datetime.today()

        self.stats['lastdays'] = dict()
        for gtk, gt in GROUPED_TYPES:
            initiative_types = InitiativeTypes.get_names_by_group(gt)

            total = Initiatives.count_by_query({
                'initiative_type_alt': {
                    '$in': initiative_types},
                'created': {
                    '$gt': TODAY - timedelta(days=DAYS_INTERVAL),
                    '$lte': TODAY},
                })

            total_prev = Initiatives.count_by_query({
                'initiative_type_alt': {
                    '$in': initiative_types},
                'created': {
                    '$gt': TODAY - timedelta(days=DAYS_INTERVAL*2),
                    '$lte': TODAY - timedelta(days=DAYS_INTERVAL)},
                })

            trend = UP if total > total_prev else DOWN if total < total_prev else EQUAL
            self.stats['lastdays'][gtk] = {
                    'initiatives': total,
                    'trend': trend
                    }

    def deputies_by_topics(self):
        self.stats['deputiesByTopics'] = {}
        for kb in self.knowledgebases:
            self.stats['deputiesByTopics'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$unwind': '$topics'},
                    {'$match': {'topics.name': topic['name']}},
                    {'$group': {'_id': '$name', 'footprint': {'$sum': '$topics.score'}}},
                    {'$sort': {'footprint': -1}},
                    {'$limit': 10}
                    ]
                results = Footprints.aggregate_deputies(pipeline)
                if len(results) > 0:
                    self.stats['deputiesByTopics'][kb].append({
                        '_id': topic['name'],
                        'deputies': results
                    })

    def parliamentarygroups_by_topics(self):
        self.stats['parliamentarygroupsByTopics'] = {}
        for kb in self.knowledgebases:
            self.stats['parliamentarygroupsByTopics'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$match': {'tagged.topics': topic['name']}},
                    {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
                results = list(Initiatives.aggregate_by_kb(kb, pipeline))
                if len(results) > 0:
                    self.stats['parliamentarygroupsByTopics'][kb].append({
                        '_id': topic['name'],
                        'parliamentarygroups': results
                    })

    def places_by_topics(self):
        self.stats['placesByTopics'] = {}
        for kb in self.knowledgebases:
            self.stats['placesByTopics'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$match': {'tagged.topics': topic['name'], 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]
                results = list(Initiatives.aggregate_by_kb(kb, pipeline))

                if len(results) > 0:
                    self.stats['placesByTopics'][kb].append({
                        '_id': topic['name'],
                        'places': results
                    })

    def deputies_by_subtopics(self):
        self.stats['deputiesBySubtopics'] = {}
        for kb in self.knowledgebases:
            self.stats['deputiesBySubtopics'][kb] = list()

            for subtopic in self.subtopics[kb]:
                pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic}},
                    {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]

                results = list(Initiatives.aggregate_by_kb(kb, pipeline))
                if len(results) > 0:
                    self.stats['deputiesBySubtopics'][kb].append({
                        '_id': subtopic,
                        'deputies': results
                    })

    def parliamentarygroups_by_subtopics(self):
        self.stats['parliamentarygroupsBySubtopics'] = {}
        for kb in self.knowledgebases:
            self.stats['parliamentarygroupsBySubtopics'][kb] = list()

            for subtopic in self.subtopics[kb]:
                pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic}},
                    {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]

                results = list(Initiatives.aggregate_by_kb(kb, pipeline))
                if len(results) > 0:
                    self.stats['parliamentarygroupsBySubtopics'][kb].append({
                        '_id': subtopic,
                        'parliamentarygroups': results
                    })

    def places_by_subtopics(self):
        self.stats['placesBySubtopics'] = {}
        for kb in self.knowledgebases:
            self.stats['placesBySubtopics'][kb] = list()

            for subtopic in self.subtopics[kb]:
                pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic, 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]

                results = list(Initiatives.aggregate_by_kb(kb, pipeline))
                if len(results) > 0:
                    self.stats['placesBySubtopics'][kb].append({
                        '_id': subtopic,
                        'places': results
                    })

    def by_week(self):
        start_date = env.get('LEGISLATURE_START_DATE', '')
        end_date = env.get('LEGISLATURE_END_DATE', '')
        pipeline = [
            {'$match': {'created': {'$exists': True, '$gte': self.__convert_to_date(start_date)}}},
            {'$project': {'yearWeek': {'$dateToString': {'format': '%G-%V', 'date': '$created' }}}},
            {'$group': {'_id': '$yearWeek', 'initiatives': {'$sum': 1}}},
            {'$project': {'week': '$_id', 'initiatives': 1, '_id': 0}},
            {'$sort': {'week': 1}}
            ]
        results = Initiatives.aggregate(pipeline)
        if len(results) > 0:
            results = self.__generate_remaining_weeks(start_date, end_date, results)
            self.stats['byWeek'] = results

    def topics_by_week(self):
        start_date = env.get('LEGISLATURE_START_DATE', '')
        end_date = env.get('LEGISLATURE_END_DATE', '')
        self.stats['topicsByWeek'] = {}
        for kb in self.knowledgebases:
            self.stats['topicsByWeek'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$match': {'tagged.topics': topic['name'], 'created': {'$exists': True, '$gte': self.__convert_to_date(start_date)}}},
                    {'$project': {'yearWeek': {'$dateToString': {'format': '%G-%V', 'date': '$created' }}}},
                    {'$group': {'_id': '$yearWeek', 'initiatives': {'$sum': 1}}},
                    {'$project': {'week': '$_id', 'initiatives': 1, '_id': 0}},
                    {'$sort': {'week': 1}}
                    ]
                results = list(Initiatives.aggregate_by_kb(kb, pipeline))
                if len(results) > 0:
                    results = self.__generate_remaining_weeks(start_date, end_date, results)
                    self.stats['topicsByWeek'][kb].append({
                        '_id': topic['name'],
                        'byWeek': results
                    })

    def __generate_remaining_weeks(self, start_date_param, end_date_param, data):
        if end_date_param == '':
            now = datetime.now()
        else:
            now = self.__convert_to_date(end_date_param)
        start_date = self.__convert_to_date(start_date_param)
        remaining_weeks = []
        date_it = start_date
        while date_it <= now:
            week_it = date_it.strftime('%G-%V')
            if not any(d['week'] == week_it for d in data):
                remaining_weeks.append({'initiatives': 0, 'week': week_it})
            date_it += timedelta(weeks=1)
        data += remaining_weeks
        data = sorted(data, key=lambda d: d['week'])
        return data

    def __convert_to_date(self, str_date, separator='-'):
        if str_date == '':
            return None
        str_date_parts = str_date.split(separator)
        return datetime(
                int(str_date_parts[0]),
                int(str_date_parts[1]),
                int(str_date_parts[2])
                )



if __name__ == "__main__":
    GenerateStats().generate()
