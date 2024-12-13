import pickle
import codecs

import tipi_tasks
from tipi_data.models.initiative import Tag
from tipi_data.repositories.amendments import Amendments
from tipi_data.repositories.initiatives import Initiatives
from tipi_data.repositories.knowledgebases import KnowledgeBases
from tipi_data.repositories.tags import Tags
from tipi_data.repositories.alerts import InitiativeAlerts

from logger import get_logger
from alerts.settings import USE_ALERTS, REASONS
from tagger.topic_alignment import calculate_single_topic_alignment


log = get_logger(__name__)


class TagInitiatives:

    def __same_field(self, tag1, tag2, field):
        return tag1[field] == tag2[field]

    def __same_tag(self, tag1, tag2):
        return self.__same_field(tag1, tag2, 'topic') \
                and self.__same_field(tag1, tag2, 'subtopic') \
                and self.__same_field(tag1, tag2, 'tag')

    def __merge_results(self, title_tags, body_tags):
        if len(title_tags) == 0:
            if len(body_tags) > 0:
                return body_tags
            return []
        merged_tags = body_tags.copy()
        for title_tag in title_tags:
            added = False
            for body_tag in body_tags:
                if self.__same_tag(title_tag, body_tag):
                    body_tag['times'] += title_tag['times']
                    added = True
                    break
            if not added:
                merged_tags.append(title_tag.copy())
        return merged_tags

    def tag_initiatives(self, initiatives, tags, merge=False, send_alerts=True, kb=None):
        total = len(initiatives)
        for index, initiative in enumerate(initiatives):
            log.info(f"Tagging initiative {index+1} of {total}: {initiative['reference']} {initiative['initiative_type_alt']}")
            self.tag_initiative(initiative, tags, merge, send_alerts, kb)

    def tag_initiative(self, initiative, tags, merge=False, send_alerts=True, kb=False):
        try:
            self.untag_when_required(initiative, merge, kb)

            if kb:
                initiative.init_tagged_kb(kb)

            tags = self.get_tags(initiative, tags)

            for tag in tags:
                initiative.add_tag(tag['knowledgebase'], tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

            initiative.remove_single_occurences()

            calculate_single_topic_alignment(initiative, False)

            initiative.save()

            if initiative.has_tags() and USE_ALERTS and send_alerts:
                InitiativeAlerts.create_alert(initiative, REASONS['published'])

        except Exception as e:
            log.error(f"Error tagging {initiative['id']}: {e}")

    def get_tags(self, initiative, tags):
        tipi_tasks.init()
        title_result = tipi_tasks.tagger.extract_tags_from_text(initiative['title'], tags)
        title_tags = []
        if 'result' in title_result.keys():
            title_tags = title_result['result']['tags']

        if 'content' not in initiative:
            return title_tags

        text = ' '.join(initiative['content'])
        body_result = tipi_tasks.tagger.extract_tags_from_text(text, tags)
        if 'result' not in body_result.keys():
            return title_tags

        body_tags = body_result['result']['tags']
        return self.__merge_results(title_tags, body_tags)

    def untag_when_required(self, initiative, merge, kb):
        if merge:
            return

        if kb:
            initiative.untag_kb(kb)
        else:
            initiative.untag()

    def run(self):
        self.tag_untagged()

        kbs = KnowledgeBases.get_all()
        for kb in kbs:
            log.info(f"Tagging kb {kb}")
            self.tag_kb(kb)

    def tag_untagged(self):
        log.info("Tagging completely untagged initiatives")
        tags = Tags.get_all()
        tags = codecs.encode(pickle.dumps(tags), "base64").decode()
        initiatives = list(Initiatives.get_all_short_untagged())
        self.tag_initiatives(initiatives, tags, True, True)

    def tag_kb(self, kb):
        log.info(f'Tagging knowledge base "{kb}"')
        tags = Tags.by_kb(kb)
        tags = codecs.encode(pickle.dumps(tags), "base64").decode()
        initiatives = list(Initiatives.by_kb_short_untagged(kb))
        self.tag_initiatives(initiatives, tags, True, True, kb)

    def tag_long(self):
        log.info("Tagging long initiatives")
        self.tag_long_untagged()
        kbs = KnowledgeBases.get_all()
        for kb in kbs:
            log.info(f"Tagging kb {kb}")
            self.tag_long_by_kb(kb)

    def tag_long_untagged(self):
        log.info("Tagging completely untagged initiatives")
        tags = Tags.get_all()
        tags = codecs.encode(pickle.dumps(tags), "base64").decode()
        initiatives = list(Initiatives.get_all_long_untagged())
        self.tag_initiatives(initiatives, tags, True, True)

    def tag_long_by_kb(self, kb):
        log.info(f'Tagging completely untagged initiatives by knowledge base "{kb}"')
        tags = Tags.by_kb(kb)
        tags = codecs.encode(pickle.dumps(tags), "base64").decode()
        initiatives = list(Initiatives.by_kb_long_untagged(kb))
        self.tag_initiatives(initiatives, tags, True, True, kb)

    def new_topic(self, topic):
        log.info(f'Tagging topic "{topic}"')
        tags = codecs.encode(pickle.dumps(Tags.by_topic(topic)), "base64").decode()
        initiatives = list(Initiatives.get_all())
        self.tag_initiatives(initiatives, tags, True, False)

    def new_tag(self, topic, tag):
        log.info(f'Tagging tag "{tag}" from topic "{topic}"')
        tag = Tags.by_name(topic, tag)
        tags = codecs.encode(pickle.dumps(tag), "base64").decode()
        initiatives = list(Initiatives.get_all())
        self.tag_initiatives(initiatives, tags, True, False)

    def by_reference(self, reference):
        log.info(f'Tagging initiative "{reference}"')
        tags = codecs.encode(pickle.dumps(Tags.get_all()), "base64").decode()
        initiatives = list(Initiatives.by_reference(reference))
        self.tag_initiatives(initiatives, tags, False, False)

    def rename(self, topic, old_tag, new_tag):
        log.info(f'Renaming tag "{old_tag}" by "{new_tag}" from topic "{topic}"')
        initiatives = Initiatives.by_tag(topic, old_tag)
        for initiative in initiatives:
            for kb in initiative.tagged:
                for tag in kb.tags:
                    if tag.topic == topic and tag.tag == old_tag:
                        tag.tag = new_tag
            initiative.save()

    def merge_tags(self, old_tags, new_tags):
        for new_tag in new_tags:
            if any(old_tag.tag == new_tag.tag for old_tag in old_tags):
                continue
            old_tags.append(new_tag)
        return old_tags

    def tag_amendments(self):
        amendments = list(Amendments.get_all_untagged())
        tags = Tags.get_all()
        tags = codecs.encode(pickle.dumps(tags), "base64").decode()

        total = len(amendments)
        for index, amendment in enumerate(amendments):
            log.info(f"Tagging amendment {index+1} of {total}: {amendment['id']} {amendment['type']}")
            self.tag_amendment(amendment, tags)

    def tag_amendment(self, amendment, tags):
        try:
            result_tags = self.get_amendment_tags(amendment['justification'], tags)

            for tag in result_tags:
                amendment.add_justification_tag(tag['knowledgebase'], tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

            if 'propossed_change' in amendment:
                result_tags = self.get_amendment_tags(amendment['propossed_change'], tags)

                for tag in result_tags:
                    amendment.add_propossed_change_tag(tag['knowledgebase'], tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

            amendment.save()
        except Exception as e:
            log.error(f"Error tagging {amendment['id']}: {e}")

    def get_amendment_tags(self, text, tags):
        tipi_tasks.init()
        result = tipi_tasks.tagger.extract_tags_from_text(' '.join(text), tags)

        tags = []
        if 'result' in result.keys():
            tags = result['result']['tags']

        return tags
