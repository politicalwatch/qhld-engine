"""Integration tests for ``UntagInitiatives`` against a throwaway MongoDB.

Each test seeds tagged initiatives, runs an untag operation, and reads the result
back through the ``Initiatives`` repository. Characterization: these pin the
behaviour the engine currently has in production.
"""

import pytest

from tipi_data.models.initiative import Initiative, Tag, Tagged
from tipi_data.repositories.initiatives import Initiatives

from qhld_engine.untagger.untag_initiatives import UntagInitiatives

pytestmark = pytest.mark.integration


def _tagged(kb, tag_tuples):
    tags = [Tag(topic=t, subtopic=s, tag=name, times=times)
            for (t, s, name, times) in tag_tuples]
    topics = sorted({tag.topic for tag in tags})
    return Tagged(knowledgebase=kb, topics=topics, tags=tags)


def _seed(id, reference, tagged):
    Initiatives.save(Initiative(id=id, reference=reference, tagged=tagged))


def test_untag_all_clears_every_initiative(mongo_db):
    _seed("i1", "R1", [_tagged("kb1", [("A", "s", "t1", 2)])])
    _seed("i2", "R2", [_tagged("kb2", [("B", "s", "t2", 3)])])

    UntagInitiatives().untag_all()

    assert all(len(i.tagged) == 0 for i in Initiatives.get_all())


def test_by_kb_pulls_only_that_knowledgebase(mongo_db):
    _seed("i1", "R1", [
        _tagged("kb1", [("A", "s", "t1", 2)]),
        _tagged("kb2", [("B", "s", "t2", 3)]),
    ])

    UntagInitiatives().by_kb("kb1")

    reloaded = Initiatives.get("i1")
    assert [t.knowledgebase for t in reloaded.tagged] == ["kb2"]


def test_by_reference_clears_only_matching_reference(mongo_db):
    _seed("i1", "R1", [_tagged("kb1", [("A", "s", "t1", 2)])])
    _seed("i2", "R2", [_tagged("kb1", [("B", "s", "t2", 3)])])

    UntagInitiatives().by_reference("R1")

    assert len(Initiatives.get("i1").tagged) == 0
    assert len(Initiatives.get("i2").tagged) == 1


def test_by_topic_strips_topic_keeps_others(mongo_db):
    _seed("i1", "R1", [_tagged("kb1", [
        ("A", "s", "t1", 2),
        ("B", "s2", "t2", 3),
    ])])

    UntagInitiatives().by_topic("A")

    kb = Initiatives.get("i1").tagged[0]
    assert kb.topics == ["B"]
    assert {t.tag for t in kb.tags} == {"t2"}


def test_by_tag_removes_single_tag_and_recomputes_topics(mongo_db):
    _seed("i1", "R1", [_tagged("kb1", [
        ("A", "s", "t1", 2),
        ("A", "s2", "t2", 1),
    ])])

    UntagInitiatives().by_tag("A", "t1")

    kb = Initiatives.get("i1").tagged[0]
    assert {t.tag for t in kb.tags} == {"t2"}
    assert kb.topics == ["A"]
