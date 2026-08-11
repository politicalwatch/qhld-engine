"""Unit tests for bitext scoring.

The behaviours pinned here are the ones that were got WRONG while building the gold set,
each of which changed a headline figure:

- a rendering belonging to another source is a NEGATIVE, not an exclusion (excluding
  them dropped 227 of 323 negatives and turned Galician from not-separable to separable);
- a rendering the gold set could not attribute at all is NEITHER class;
- RANK and GATE are different questions and a good score on one says nothing about the
  other.
"""

import pytest

from qhld_engine.application.evaluation.bitext_benchmark import RunBitextBenchmark
from qhld_engine.domain.evaluation import bitext_scoring

pytestmark = pytest.mark.unit


def _row(lang, source, candidate, score, is_pair, is_rendering=False, video="s1"):
    return {"lang": lang, "video_id": video, "source": source, "candidate": candidate,
            "score": score, "is_pair": is_pair, "is_rendering": is_rendering}


def test_gate_is_positive_only_when_every_pair_beats_every_non_pair():
    clear = [_row("ca", 0, 1, 0.90, True), _row("ca", 0, 2, 0.40, False)]
    assert bitext_scoring.score(clear)["ca"]["gate"] == pytest.approx(0.50)

    overlapping = [_row("ca", 0, 1, 0.60, True), _row("ca", 0, 2, 0.70, False)]
    assert bitext_scoring.score(overlapping)["ca"]["gate"] < 0


def test_an_instrument_can_rank_perfectly_and_still_gate_nothing():
    """Basque in one test: the partner is always top, and no threshold separates."""
    rows = [
        _row("eu", 0, 1, 0.80, True), _row("eu", 0, 2, 0.79, False),
        _row("eu", 3, 4, 0.66, True), _row("eu", 3, 5, 0.65, False, video="s2"),
    ]
    report = bitext_scoring.score(rows)["eu"]
    assert report["rank_top1"] == report["rank_total"] == 2   # ranks perfectly
    assert report["gate"] < 0                                  # separates nothing


def test_a_rendering_of_another_source_counts_as_a_negative():
    # The hardest negative there is: a real translation, of the neighbouring paragraph.
    rows = [_row("gl", 0, 5, 0.90, True),
            _row("gl", 0, 6, 0.95, False, is_rendering=False)]
    report = bitext_scoring.score(rows)["gl"]
    assert report["n_non"] == 1
    assert report["gate"] < 0


def test_an_unattributable_rendering_is_neither_class():
    rows = [_row("gl", 0, 5, 0.90, True),
            _row("gl", 0, 6, 0.99, False, is_rendering=True)]
    report = bitext_scoring.score(rows)["gl"]
    assert report["n_true"] == 1
    assert report["n_non"] == 0
    assert report["gate"] is None  # nothing to gate against, rather than a false +0.99


def test_specificity_is_reported_because_accuracy_hides_it():
    # 1 pair against 9 negatives: answering "no" to everything is 90% accurate.
    rows = [{"lang": "eu", "video_id": "s1", "source": 0, "candidate": 1,
             "decision": False, "is_pair": True, "is_rendering": False}]
    rows += [{"lang": "eu", "video_id": "s1", "source": 0, "candidate": c,
              "decision": False, "is_pair": False, "is_rendering": False}
             for c in range(2, 11)]
    cell = bitext_scoring.score_decisions(rows)["eu"]
    assert cell["specificity"] == 1.0
    assert cell["recall"] == 0.0


def test_speech_level_needs_one_confirmation_not_all_of_them():
    """Existence is a per-speech question, so mediocre per-pair recall can still be
    right about every speech."""
    rows = [{"lang": "eu", "video_id": "s1", "source": 0, "candidate": 1,
             "decision": True, "is_pair": True, "has_renderings": True},
            {"lang": "eu", "video_id": "s1", "source": 2, "candidate": 3,
             "decision": False, "is_pair": True, "has_renderings": True}]
    assert bitext_scoring.speech_level(rows) == {"speeches": 1, "detected": 1,
                                                 "missed": []}


def test_split_scoring_separates_the_two_ways_a_block_can_be_wrong():
    """They are not symmetric. A rendering left in the as-delivered block is the
    self-translation defect; a spoken paragraph removed from it is worse — words nobody
    can get back."""
    left_in = [{"video_id": "s1", "lang": "ca", "gold": {4, 5}, "predicted": {4},
                "undecided": False, "blocks": 2}]
    report = bitext_scoring.score_split(left_in)
    assert report["micro"]["precision"] == 1.0      # nothing wrongly removed
    assert report["micro"]["recall"] == 0.5
    assert report["misses"][0]["missed"] == [5]

    removed = [{"video_id": "s1", "lang": "ca", "gold": {4}, "predicted": {4, 9},
                "undecided": False, "blocks": 2}]
    report = bitext_scoring.score_split(removed)
    assert report["micro"]["recall"] == 1.0
    assert report["micro"]["precision"] == 0.5
    assert report["misses"][0]["spurious"] == [9]


def test_exact_block_match_is_all_or_nothing_per_speech():
    rows = [{"video_id": "a", "lang": "gl", "gold": {1, 2}, "predicted": {1, 2},
             "undecided": False, "blocks": 2},
            {"video_id": "b", "lang": "gl", "gold": {1, 2}, "predicted": {1},
             "undecided": False, "blocks": 2}]
    report = bitext_scoring.score_split(rows)
    assert report["exact"] == 1 and report["speeches"] == 2


# ---- the frozen gold set itself ----------------------------------------------------

def test_the_goldset_is_self_contained_and_consistent():
    runner = RunBitextBenchmark()
    assert len(runner.entries) >= 11
    for speech in runner.entries:
        paragraphs = speech["paragraphs"]
        assert paragraphs and all(p.strip() for p in paragraphs)
        for source, candidate in speech["pairs"]:
            assert 0 <= source < len(paragraphs)
            assert 0 <= candidate < len(paragraphs)
            # Every paired Spanish paragraph must also be declared a rendering: the two
            # fields are the same truth at different granularity.
            assert candidate in speech["renderings"], speech["video_id"]
        assert all(0 <= r < len(paragraphs) for r in speech["renderings"])


def test_the_goldset_keeps_the_hard_negatives():
    """Guards the exclusion bug: most comparisons must be usable negatives."""
    runner = RunBitextBenchmark()
    comparisons = list(runner.comparisons())
    pairs = sum(1 for c in comparisons if c[3])
    excluded = sum(1 for c in comparisons if not c[3] and c[4])
    negatives = len(comparisons) - pairs - excluded
    assert pairs >= 46
    assert negatives > 300           # was 96 while the exclusion was too broad
    assert excluded < negatives / 5
