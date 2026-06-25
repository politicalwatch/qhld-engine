"""Unit tests for topic-alignment computation — pure, no DB.

``calculate_single_topic_alignment(initiative, needs_to_be_saved=False)`` operates
purely in memory on the model instance, so it needs no Mongo. It buckets each
knowledgebase's tags by topic, sums their ``times``, turns those into percentages
of the kb total, and stores them on ``tagged[*].topic_alignment`` sorted by
percentage descending.
"""

import pytest

from tipi_data.models.initiative import Initiative, Tag, Tagged

from qhld_engine.tagger.topic_alignment import calculate_single_topic_alignment

pytestmark = pytest.mark.unit


def _initiative_with_tags(tags):
    tagged = Tagged(
        knowledgebase="kb1",
        topics=sorted({t.topic for t in tags}),
        tags=tags,
    )
    return Initiative(id="init-1", tagged=[tagged])


def test_percentages_sum_to_topic_share_and_sorted_desc():
    initiative = _initiative_with_tags([
        Tag(topic="A", subtopic="s1", tag="t1", times=3),
        Tag(topic="B", subtopic="s2", tag="t2", times=1),
    ])

    calculate_single_topic_alignment(initiative, needs_to_be_saved=False)

    alignment = initiative.tagged[0].topic_alignment
    # total times = 4 -> A=75%, B=25%, ordered descending by percentage.
    assert [(a.topic, a.percentage) for a in alignment] == [("A", 75.0), ("B", 25.0)]


def test_times_are_summed_per_topic():
    initiative = _initiative_with_tags([
        Tag(topic="A", subtopic="s1", tag="t1", times=2),
        Tag(topic="A", subtopic="s2", tag="t2", times=1),
        Tag(topic="B", subtopic="s3", tag="t3", times=1),
    ])

    calculate_single_topic_alignment(initiative, needs_to_be_saved=False)

    by_topic = {a.topic: a.percentage for a in initiative.tagged[0].topic_alignment}
    # A = (2+1)/4 = 75%, B = 1/4 = 25%
    assert by_topic == {"A": 75.0, "B": 25.0}


def test_handles_multiple_knowledgebases_independently():
    initiative = Initiative(
        id="init-2",
        tagged=[
            Tagged(knowledgebase="kb1", topics=["A"],
                   tags=[Tag(topic="A", subtopic="s", tag="t", times=5)]),
            Tagged(knowledgebase="kb2", topics=["B", "C"],
                   tags=[Tag(topic="B", subtopic="s", tag="t", times=3),
                         Tag(topic="C", subtopic="s", tag="u", times=1)]),
        ],
    )

    calculate_single_topic_alignment(initiative, needs_to_be_saved=False)

    assert [(a.topic, a.percentage) for a in initiative.tagged[0].topic_alignment] == [("A", 100.0)]
    assert [(a.topic, a.percentage) for a in initiative.tagged[1].topic_alignment] == [("B", 75.0), ("C", 25.0)]
