from tipi_data.models.stats import Stats
from tipi_data.repositories.topics import Topics
from tipi_data.repositories.initiatives import Initiatives


class GenerateStats(object):

    def __init__(self):
        self.topics = Topics.get_all()
        self.subtopics = self.topics.distinct('tags.subtopic')
        self.knowledgebases = list(Topics.get_kbs())
        self.stats = Stats()

    def generate(self):
        Stats.objects().delete()
        self.overall()
        self.deputies_by_topics()
        self.deputies_by_subtopics()
        self.parliamentarygroups_by_topics()
        self.parliamentarygroups_by_subtopics()
        self.places_by_topics()
        self.places_by_subtopics()
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

        pipeline = [
            {'$match': {'tagged': {'$exists': True, '$not': {'$size': 0}}}},
            {'$unwind': '$tagged'},
            {'$group': {'_id': '$tagged', 'initiatives': {'$sum': 1}}},
            {'$sort': {'initiatives': -1}}
            ]
        for kb in self.knowledgebases:
            results = Initiatives.by_kb(kb).aggregate(*pipeline)
            for item in results:
                self.stats['overall']['topics'][kb].append(item)

        for subtopic in self.subtopics:
            pipeline = [
                {'$match': {'tagged.tags.subtopic': subtopic}},
                {'$group': {'_id': subtopic, 'initiatives': {'$sum': 1}}}
                ]
            for kb in self.knowledgebases:
                results = Initiatives.by_kb(kb).aggregate(*pipeline)
                for item in results:
                    self.stats['overall']['subtopics'][kb].append(item)

        for kb in self.knowledgebases:
            self.stats['overall']['subtopics'][kb].sort(key=lambda x: x['initiatives'], reverse=True)

    def deputies_by_topics(self):
        self.stats['deputiesByTopics'] = {}
        for kb in self.knowledgebases:
            self.stats['deputiesByTopics'][kb] = list()

        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]
            for kb in self.knowledgebases:
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

        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
            for kb in self.knowledgebases:
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

        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name'], 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]
            for kb in self.knowledgebases:
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

        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]

            for kb in self.knowledgebases:
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

        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
            for kb in self.knowledgebases:
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

        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic, 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]

            for kb in self.knowledgebases:
                results = list(Initiatives.by_kb(kb).aggregate(*pipeline))
                if len(results) > 0:
                    self.stats['placesBySubtopics'][kb].append({
                        '_id': subtopic,
                        'places': results
                    })


if __name__ == "__main__":
    GenerateStats().generate()
