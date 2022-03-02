from datetime import datetime
from datetime import timedelta

from tipi_data.models.stats import Stats
from tipi_data.repositories.initiatives import Initiatives
from tipi_data.models.initiative_type import InitiativeType
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.topics import Topics

from extractors.config import MODULE_EXTRACTOR


class GenerateStats(object):

    def __init__(self):
        self.topics = {}
        self.subtopics = {}
        self.knowledgebases = list(KnowledgeBases.get_all())

        for kb in self.knowledgebases:
            self.topics[kb] = Topics.by_kb(kb)
            self.subtopics[kb] = Topics.by_kb(kb).distinct('tags.subtopic')
        self.stats = Stats()

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

        Stats.objects().delete()
        self.stats.save()

    def overall(self):
        self.stats['overall'] = {
                'allinitiatives': Initiatives.get_all().count(),
                'topics': {},
                'subtopics': {},
                }
        for kb in self.knowledgebases:
            self.stats['overall'][kb] = Initiatives.by_kb(kb).count()
            self.stats['overall']['topics'][kb] = list()
            self.stats['overall']['subtopics'][kb] = list()

            for topic in self.topics[kb]:
                pipeline = [
                    {'$match': {'tagged.tags.topic': topic.name}},
                    {'$group': {'_id': topic.name, 'initiatives': {'$sum': 1}}}
                ]
                results = Initiatives.by_kb(kb).aggregate(*pipeline)
                for item in results:
                    self.stats['overall']['topics'][kb].append(item)

            for subtopic in self.subtopics[kb]:
                pipeline = [
                    {'$match': {'tagged.tags.subtopic': subtopic}},
                    {'$group': {'_id': subtopic, 'initiatives': {'$sum': 1}}}
                    ]
                results = Initiatives.by_kb(kb).aggregate(*pipeline)
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
        DAYS_INTERVAL = 7
        TODAY = datetime.today()

        self.stats['lastdays'] = dict()
        for gtk, gt in GROUPED_TYPES:
            initiative_types = list(map(
                lambda x: x.name,
                InitiativeType.objects.filter(group=gt).only('name')))

            total = Initiatives.by_query({
                'initiative_type_alt': {
                    '$in': initiative_types},
                'created': {
                    '$gt': TODAY - timedelta(days=DAYS_INTERVAL),
                    '$lte': TODAY},
                }).count()

            total_prev = Initiatives.by_query({
                'initiative_type_alt': {
                    '$in': initiative_types},
                'created': {
                    '$gt': TODAY - timedelta(days=DAYS_INTERVAL*2),
                    '$lte': TODAY - timedelta(days=DAYS_INTERVAL)},
                }).count()

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
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]
                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
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
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
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
                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))

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
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]

                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
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
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]

                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
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

                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
                if len(results) > 0:
                    self.stats['placesBySubtopics'][kb].append({
                        '_id': subtopic,
                        'places': results
                    })


if __name__ == "__main__":
    GenerateStats().generate()
