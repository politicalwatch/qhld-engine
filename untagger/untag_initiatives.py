from tipi_data.repositories.initiatives import Initiatives
from logger import get_logger
from tagger.topic_alignment import calculate_single_topic_alignment


log = get_logger(__name__)


class UntagInitiatives:

    def untag_all(self):
        log.info('Untagging all initiatives')
        initiatives = Initiatives.get_all()
        initiatives.update(unset__tagged=1)
        for initiative in initiatives:
            calculate_single_topic_alignment(initiative, True)

    def by_kb(self, kb):
        log.info(f'Untagging knowledge base "{kb}"')
        initiatives = Initiatives.get_all()
        initiatives.update(pull__tagged__knowledgebase=kb)
        for initiative in initiatives:
            calculate_single_topic_alignment(initiative, True)

    def by_topic(self, topic):
        log.info(f'Untagging topic "{topic}"')
        initiatives = Initiatives.by_query({
            'tagged.topics': topic
            })
        for initiative in initiatives:
            for kb in initiative.tagged:
                kb.topics = [t for t in kb.topics if t != topic]
                kb.tags = [t for t in kb.tags if t.topic != topic]
            calculate_single_topic_alignment(initiative, False)
            initiative.save()

    def by_tag(self, topic, tag):
        log.info(f'Untagging tag "{tag}" from topic "{topic}"')
        initiatives = Initiatives.by_tag(topic, tag)
        for initiative in initiatives:
            for kb in initiative.tagged:
                kb.tags = [t for t in kb.tags if t.topic != topic or t.tag != tag]
                kb.topics = list({t.topic for t in kb.tags})
            calculate_single_topic_alignment(initiative, False)
            initiative.save()

    def by_reference(self, reference):
        log.info(f'Untagging initiative "{reference}"')
        initiatives = Initiatives.by_reference(reference)
        initiatives.update(unset__tagged=1)
        for initiative in initiatives:
            calculate_single_topic_alignment(initiative, True)
