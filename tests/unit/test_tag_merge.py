"""Unit tests for ``TagInitiatives`` tag-merge logic — pure dict math, no DB.

``__merge_results`` combines tags found in an initiative's title with those found
in its body: when a title tag matches a body tag (same topic/subtopic/tag) their
``times`` are summed; otherwise the title tag is appended. Accessed through the
name-mangled attribute since it's a private method.
"""

import pytest

from qhld_engine.tagger.tag_initiatives import TagInitiatives

pytestmark = pytest.mark.unit


def _merge(title_tags, body_tags):
    return TagInitiatives()._TagInitiatives__merge_results(title_tags, body_tags)


def _tag(topic, subtopic, tag, times):
    return {"topic": topic, "subtopic": subtopic, "tag": tag, "times": times}


def test_empty_title_returns_body():
    body = [_tag("A", "s", "t", 3)]
    assert _merge([], body) == body


def test_empty_title_and_body_returns_empty():
    assert _merge([], []) == []


def test_matching_tags_sum_times():
    title = [_tag("A", "s", "t", 2)]
    body = [_tag("A", "s", "t", 3)]

    merged = _merge(title, body)

    assert len(merged) == 1
    assert merged[0]["times"] == 5


def test_distinct_title_tag_is_appended():
    title = [_tag("A", "s1", "t1", 1)]
    body = [_tag("B", "s2", "t2", 4)]

    merged = _merge(title, body)

    assert {(t["topic"], t["tag"]) for t in merged} == {("A", "t1"), ("B", "t2")}


def test_same_topic_but_different_tag_not_merged():
    # __same_tag requires topic AND subtopic AND tag to match.
    title = [_tag("A", "s", "t1", 1)]
    body = [_tag("A", "s", "t2", 1)]

    merged = _merge(title, body)

    assert len(merged) == 2
