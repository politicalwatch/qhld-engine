from tipi_data.repositories.initiatives import Initiatives
from qhld_engine.logger import get_logger
from qhld_engine.tagger.topic_alignment import calculate_single_topic_alignment


log = get_logger(__name__)


class UntagInitiatives:

    def untag_all(self):
        log.info('Untagging all initiatives')
        Initiatives.unset_tagged_all()
        for initiative in Initiatives.get_all():
            calculate_single_topic_alignment(initiative, True)

    def by_kb(self, kb):
        log.info(f'Untagging knowledge base "{kb}"')
        Initiatives.pull_tagged_by_kb(kb)
        for initiative in Initiatives.get_all():
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
            Initiatives.save(initiative)

    def by_tag(self, topic, tag):
        log.info(f'Untagging tag "{tag}" from topic "{topic}"')
        initiatives = Initiatives.by_tag(topic, tag)
        for initiative in initiatives:
            for kb in initiative.tagged:
                kb.tags = [t for t in kb.tags if t.topic != topic or t.tag != tag]
                kb.topics = list({t.topic for t in kb.tags})
            calculate_single_topic_alignment(initiative, False)
            Initiatives.save(initiative)

    def by_reference(self, reference):
        log.info(f'Untagging initiative "{reference}"')
        Initiatives.unset_tagged_by_reference(reference)
        for initiative in Initiatives.by_reference(reference):
            calculate_single_topic_alignment(initiative, True)
