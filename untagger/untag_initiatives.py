from tipi_data.repositories.initiatives import Initiatives


class UntagInitiatives:

    def untag_all(self):
        print('Untagging all initiatives')
        Initiatives.get_all().update(unset__tagged=1)

    def by_kb(self, kb):
        print('Untagging knowledge base "' + kb + '"')
        Initiatives.get_all().update(pull_tagged__knowledgebase=kb)

    def by_topic(self, topic):
        print('Untagging topic "' + topic + '"')
        Initiatives.get_all().update(pull_tagged__topics=topic)

    def by_tag(self, tag):
        print('Untagging tag "' + tag + '"')
        Initiatives.get_all().update(pull_tagged__tags__tag=tag)

    def by_reference(self, reference):
        print('Untagging initiative "' + reference + '"')
        Initiatives.by_reference(reference).update(unset__tagged=1)
