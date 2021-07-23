from tipi_data.models.stats import Stats
from tipi_data.models.topic import Topic
from tipi_data.models.initiative import Initiative


class GenerateStats(object):

    def __init__(self):
        self.topics = Topic.objects()
        self.subtopics = self.topics.distinct('tags.subtopic')
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
                'initiatives': Initiative.objects.count(),
                'allinitiatives': Initiative.all.count(),
                'topics': list(),
                'subtopics': list()
                }
        pipeline = [
            {'$match': {'tagged': {'$exists': True, '$not': {'$size': 0}}}},
            {'$unwind': '$tagged'},
            {'$group': {'_id': '$tagged', 'initiatives': {'$sum': 1}}},
            {'$sort': {'initiatives': -1}}
            ]
        result = Initiative.objects().aggregate(*pipeline)
        for item in result:
            self.stats['overall']['topics'].append(item)
        for subtopic in self.subtopics:
            pipeline = [
                {'$match': {'tagged.tags.subtopic': subtopic}},
                {'$group': {'_id': subtopic, 'initiatives': {'$sum': 1}}}
                ]
            result = Initiative.objects().aggregate(*pipeline)
            if result._has_next():
                self.stats['overall']['subtopics'].append(result.next())
        self.stats['overall']['subtopics'].sort(key=lambda x: x['initiatives'], reverse=True)

    def deputies_by_topics(self):
        self.stats['deputiesByTopics'] = list()
        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['deputiesByTopics'].append({
                    '_id': topic['name'],
                    'deputies': result
                    })

    def parliamentarygroups_by_topics(self):
        self.stats['parliamentarygroupsByTopics'] = list()
        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name']}}, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['parliamentarygroupsByTopics'].append({
                    '_id': topic['name'],
                    'parliamentarygroups': result
                    })

    def places_by_topics(self):
        self.stats['placesByTopics'] = list()
        for topic in self.topics:
            pipeline = [
                    {'$match': {'tagged.topics': topic['name'], 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['placesByTopics'].append({
                    '_id': topic['name'],
                    'places': result
                    })

    def deputies_by_subtopics(self):
        self.stats['deputiesBySubtopics'] = list()
        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_deputies'},
                    {'$group': {'_id': '$author_deputies', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 10}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['deputiesBySubtopics'].append({
                    '_id': subtopic,
                    'deputies': result
                    })

    def parliamentarygroups_by_subtopics(self):
        self.stats['parliamentarygroupsBySubtopics'] = list()
        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic } }, {'$unwind': '$author_parliamentarygroups'},
                    {'$group': {'_id': '$author_parliamentarygroups', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['parliamentarygroupsBySubtopics'].append({
                    '_id': subtopic,
                    'parliamentarygroups': result
                    })

    def places_by_subtopics(self):
        self.stats['placesBySubtopics'] = list()
        for subtopic in self.subtopics:
            pipeline = [
                    {'$match': { 'tagged.tags.subtopic': subtopic, 'place': {'$not': {'$eq': ""}, '$exists': True}}},
                    {'$group': {'_id': '$place', 'initiatives': {'$sum': 1}}}, {'$sort': {'initiatives': -1}},
                    {'$limit': 5}
                    ]
            result = list(Initiative.objects().aggregate(*pipeline))
            if len(result) > 0:
                self.stats['placesBySubtopics'].append({
                    '_id': subtopic,
                    'places': result
                    })


if __name__ == "__main__":
    GenerateStats().generate()
