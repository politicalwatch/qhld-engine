"""Integration test for ``TagInitiatives`` orchestration against throwaway MongoDB.

We stub only ``tipi_tasks.tagger.extract_tags_from_text`` (the actual tag
extraction is qhld-tasks' own, separately tested domain). What this exercises is
the *engine's* orchestration: applying the returned tags, dropping single
occurrences, computing topic alignment, and persisting via ``Initiatives``.
"""

import pytest

import tipi_tasks
import tipi_tasks.tagger
from tipi_data.models.initiative import Initiative
from tipi_data.repositories.initiatives import Initiatives

from tagger.tag_initiatives import TagInitiatives

pytestmark = pytest.mark.integration


# Two tags on topic A (kept) and one lone tag on topic B (dropped by
# remove_single_occurences since its topic total is 1).
CANNED_TAGS = [
    {"knowledgebase": "kb1", "topic": "A", "subtopic": "s1", "tag": "t1", "times": 3},
    {"knowledgebase": "kb1", "topic": "A", "subtopic": "s2", "tag": "t2", "times": 2},
    {"knowledgebase": "kb1", "topic": "B", "subtopic": "s3", "tag": "t3", "times": 1},
]


def test_tag_initiative_applies_tags_and_alignment(mongo_db, monkeypatch):
    initiative = Initiative(
        id="i1", reference="R1", title="A title", content=["body text"])
    Initiatives.save(initiative)

    def fake_extract(text, tags):
        # Return tags for the title only; the body call yields no result, so
        # get_tags returns the title tags unchanged (no double-counting).
        if text == "A title":
            return {"result": {"tags": [dict(t) for t in CANNED_TAGS]}}
        return {}

    monkeypatch.setattr(tipi_tasks, "init", lambda *a, **k: None)
    monkeypatch.setattr(tipi_tasks.tagger, "extract_tags_from_text", fake_extract)

    TagInitiatives().tag_initiative(initiative, "unused-tags-blob", send_alerts=False)

    reloaded = Initiatives.get("i1")
    assert [t.knowledgebase for t in reloaded.tagged] == ["kb1"]
    kb = reloaded.tagged[0]
    # topic B had a single occurrence -> removed; topic A's two tags kept.
    assert {t.tag for t in kb.tags} == {"t1", "t2"}
    assert kb.topics == ["A"]
    assert [(a.topic, a.percentage) for a in kb.topic_alignment] == [("A", 100.0)]
