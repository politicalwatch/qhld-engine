"""Offline tests for the gate benchmark's probe-set wiring.

No network: these cover how the two arms are assembled, which is where a silent
mistake would be worst — a junk arm that quietly loads nothing still prints a
suppression number.
"""

import json

import pytest

from qhld_engine.application.evaluation import gate_benchmark
from qhld_ai.domain.errors import NotASpeechQuery
from qhld_ai.domain.ports.query_parser import ParsedQuery
from qhld_ai.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def test_junk_arm_is_read_from_the_retrieval_queryset_not_copied():
    """The junk probes have exactly one home. They were hardened 5 -> 24 once
    without the floor being recalibrated; a second copy would let that recur."""
    junk = gate_benchmark.load_junk()
    assert len(junk) == 24
    assert {row["id"] for row in junk} == {f"J{i}" for i in range(1, 25)}
    assert all(row["expected"] == "refuse" for row in junk)


def test_junk_classes_are_derived_from_the_queryset_own_notes():
    classes = {row["id"]: row["class"] for row in gate_benchmark.load_junk()}
    assert classes["J3"] == "injection"       # "injection-shaped probe"
    assert classes["J4"] == "gibberish"       # "gibberish probe: keyboard mash"
    assert classes["J18"] == "near-domain"    # "NEAR-DOMAIN junk, hardest class"
    assert classes["J1"] == "off-domain"      # "off-domain probe"


def test_a_note_is_classified_by_its_opening_marker_not_by_what_it_discusses():
    """J5's real note, which declares itself off-domain and then explains that
    sport would have been a near-domain trap. Matching anywhere files it under the
    class it exists to contrast itself with."""
    note = ("off-domain probe: leisure/lifestyle. Deliberately NOT sports/football — "
            "parliament genuinely debates sports policy, so that would be a near-domain "
            "trap, not junk.")
    assert gate_benchmark._junk_class(note) == "off-domain"
    assert {row["id"]: row["class"] for row in gate_benchmark.load_junk()}["J5"] == "off-domain"


def test_an_unrecognised_note_falls_back_to_the_dimension_rather_than_vanishing():
    assert gate_benchmark._junk_class("something nobody wrote a pattern for") == "offdomain"
    assert gate_benchmark._junk_class("") == "offdomain"


def test_the_probe_set_carries_four_classes_split_by_expected_reason():
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    assert len(runner.legitimate) == 35
    assert all(row["expected"] == "pass" for row in runner.legitimate)
    assert len(runner.non_search) == 10
    assert len(runner.unsupported_language) == 5
    assert len(runner.hostile) == 20
    for row in (runner.non_search + runner.unsupported_language + runner.hostile):
        assert row["expected"] == "refuse"


def test_the_legitimate_arm_keeps_its_adversarial_half():
    """The arm was widened 23 -> 35 specifically with queries whose SURFACE looks
    hostile — 'instrucciones del Ministerio', 'ley de memoria', 'saltarse el
    reglamento'. That is where a hostile classifier's false positives come from, so
    an arm that drifted back to easy searches would report a comfortable number and
    measure nothing."""
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    adversarial = [row for row in runner.legitimate if row["class"] == "adversarial"]
    assert len(adversarial) >= 10


def test_no_hostile_probe_is_reused_from_the_junk_arm():
    """J3/J21/J22 are injection-shaped but stay in the junk arm, where their notes
    say they measure the retrieval backstop. Reusing one here would put the same
    query in two arms of one run — the duplication this harness already refuses."""
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    junk_queries = {row["query"] for row in runner.junk}
    assert not junk_queries & {row["query"] for row in runner.hostile}


def test_every_refuse_row_states_which_refusal_it_deserves():
    """Without a reason a row can only be scored as refused/not, which cannot see
    a correct verdict delivered with the wrong explanation."""
    rows = gate_benchmark.load_queryset()["queries"]
    assert all(row.get("expected_reason")
               for row in rows if row["expected"] == "refuse")
    assert all(row.get("expected_reason") is None
               for row in rows if row["expected"] == "pass")


def test_every_probe_names_a_distinct_hazard():
    """The set's admission rule: a row earns its place only when the MECHANISM
    that might refuse it differs from every other row."""
    hazards = [row["hazard"] for row in gate_benchmark.load_queryset()["queries"]]
    assert len(set(hazards)) == len(hazards)


def test_most_non_searches_carry_a_real_topic_because_that_is_the_point():
    """A non-search with a real parliamentary topic is the one thing the relevance
    floor cannot save us from — it would retrieve genuinely relevant speeches. If
    this arm drifted to only topicless probes it would stop testing that."""
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    in_domain = [row for row in runner.non_search if row["in_domain"]]
    assert len(in_domain) >= len(runner.non_search) / 2


def test_arms_resolve_to_their_own_rows_and_an_unknown_arm_is_an_error():
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    assert runner.entries("legitimate") is runner.legitimate
    assert runner.entries("non-search") is runner.non_search
    assert runner.entries("unsupported-language") is runner.unsupported_language
    assert runner.entries("junk") is runner.junk
    with pytest.raises(ValueError):
        runner.entries("offdomain")


def test_a_language_refusal_is_not_filed_under_the_intent_gate(monkeypatch):
    """UnsupportedLanguage and NotASpeechQuery share a base class, so catching them
    in the wrong order would silently score every language refusal as an intent
    refusal — and the arms would still look healthy."""
    from qhld_ai.domain.errors import UnsupportedLanguage

    class _RefusesLanguage:
        def _prepare(self, query, parsed):
            raise UnsupportedLanguage(query, "en")

    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    runner._service = _RefusesLanguage()
    monkeypatch.setattr(
        runner, "_parser",
        lambda *a, **k: _StubParser(ParsedQuery(semantic_query="x", query_language="en")))
    runner.legitimate = [{"id": "U1", "class": "language", "query": "what did they say"}]
    assert runner.run()[0]["outcome"] == gate_benchmark.REFUSED_LANGUAGE


def test_ids_cannot_collide_across_arms():
    """The arms are scored separately but reported together; overlapping ids would
    silently merge two queries in the per-query table."""
    own = {row["id"] for row in gate_benchmark.load_queryset()["queries"]}
    junk = {row["id"] for row in gate_benchmark.load_junk()}
    assert not own & junk
    assert len(own) == len(gate_benchmark.load_queryset()["queries"])


class _StubService:
    """Stands in for NaturalSearchSpeeches: ``_prepare`` is the gate."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def _prepare(self, query, parsed):
        if self.behaviour == "refuse":
            raise NotASpeechQuery(query)
        return None, {"speaker": "X"}, self.behaviour, True


def _runner(monkeypatch, parsed, behaviour):
    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    runner._service = _StubService(behaviour)
    monkeypatch.setattr(runner, "_parser", lambda *a, **k: _StubParser(parsed))
    runner.legitimate = [{"id": "L1", "class": "shape", "query": "una consulta"}]
    return runner


class _StubParser:
    def __init__(self, parsed):
        self.parsed = parsed

    def parse(self, query, today):
        return self.parsed


def test_a_refusal_is_attributed_to_the_flag_when_the_parser_said_not_a_search(monkeypatch):
    parsed = ParsedQuery(semantic_query="", is_speech_search=False)
    rows = _runner(monkeypatch, parsed, "refuse").run()
    assert rows[0]["outcome"] == gate_benchmark.REFUSED_FLAG


def test_a_refusal_is_attributed_to_the_empty_parse_when_the_flag_allowed_it(monkeypatch):
    """The other raise site: the parser judged it a real search, and it died on a
    parse with no topic, no filters and nothing blocking. Never exercised by the
    live run so far, which is exactly why it is pinned here."""
    parsed = ParsedQuery(semantic_query="", is_speech_search=True)
    rows = _runner(monkeypatch, parsed, "refuse").run()
    assert rows[0]["outcome"] == gate_benchmark.REFUSED_EMPTY


def test_a_passing_query_with_no_topic_left_is_recorded_as_a_browse(monkeypatch):
    parsed = ParsedQuery(semantic_query="", is_speech_search=True)
    rows = _runner(monkeypatch, parsed, "").run()
    assert (rows[0]["outcome"], rows[0]["route"]) == (gate_benchmark.PASS, "browse")


def test_a_passing_query_with_a_topic_is_recorded_as_a_search(monkeypatch):
    parsed = ParsedQuery(semantic_query="vivienda", is_speech_search=True)
    rows = _runner(monkeypatch, parsed, "vivienda").run()
    assert (rows[0]["outcome"], rows[0]["route"]) == (gate_benchmark.PASS, "search")


def test_a_parser_that_cannot_emit_valid_output_is_scored_as_the_product_would_behave(
        monkeypatch):
    """Not an aborted run and not a skipped row: the product would carry an empty
    parse forward and the empty-parse rule would fire, so that is what is scored."""
    class _Exploding:
        def parse(self, query, today):
            raise ValueError("no json")

    runner = gate_benchmark.RunGateBenchmark(settings=Settings(_env_file=None))
    runner._service = _StubService("refuse")
    monkeypatch.setattr(runner, "_parser", lambda *a, **k: _Exploding())
    runner.legitimate = [{"id": "L1", "class": "shape", "query": "una consulta"}]
    row = runner.run()[0]
    assert row["parse_error"] == "ValueError"
    assert row["outcome"] == gate_benchmark.REFUSED_EMPTY


def test_load_junk_ignores_every_other_dimension(tmp_path):
    path = tmp_path / "queryset.json"
    path.write_text(json.dumps([
        {"id": "T1", "dimension": "topical", "query": "real", "notes": ""},
        {"id": "J1", "dimension": "offdomain", "query": "junk", "notes": "gibberish probe"},
    ]), encoding="utf-8")
    rows = gate_benchmark.load_junk(str(path))
    assert [row["id"] for row in rows] == ["J1"]
