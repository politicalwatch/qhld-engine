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


def _split_row(video="s1", lang="ca", gold=(), sources=(), delivered=(), spanish=(),
               absorbed=(), total=None):
    placed = set(delivered) | set(spanish) | set(gold) | set(sources) | set(absorbed)
    return {"video_id": video, "lang": lang, "gold": set(gold), "sources": set(sources),
            "delivered": set(delivered), "spanish": set(spanish),
            "absorbed": set(absorbed),
            "total": total if total is not None else (max(placed) + 1 if placed else 0),
            "predicted": set(spanish) - set(delivered), "undecided": False, "blocks": 2}


def test_a_paragraph_in_both_blocks_is_invisible_to_the_split_score():
    """Why score_pairs has to exist. Paragraph 0's translation was missed so the original
    stands in the Spanish block too — and `predicted` subtracts the as-delivered block, so
    the defect leaves no trace in the figures score_split reports."""
    row = _split_row(gold={2}, sources={0}, delivered={0, 1}, spanish={0, 2})

    assert bitext_scoring.score_split([row])["micro"]["f1"] == 1.0
    assert bitext_scoring.score_pairs([row])["missed"] == 1


def test_the_duplication_census_leaves_no_paragraph_unclassified():
    # 0 is a gold source, so a translation exists and was not found. 3 is a rendering, so
    # it was never spoken and does not belong in the delivered block either. 7 is neither:
    # spoken, never translated, correctly in both.
    row = _split_row(gold={3}, sources={0}, delivered={0, 3, 7}, spanish={0, 3, 7})
    census = bitext_scoring.score_pairs([row])["census"]

    assert [c["paragraph"] for c in census["defect"]] == [0]
    assert [c["paragraph"] for c in census["es_side"]] == [3]
    assert [c["paragraph"] for c in census["expected"]] == [7]


def test_an_original_the_detector_read_as_spanish_still_counts_as_missed():
    """The first rejected definition asked the aligner which ORIGINALS it had paired, which
    cannot see a source whose own paragraph the detector called Spanish — the entire
    code-mixed class, and the cases most worth seeing."""
    row = _split_row(gold={5}, sources={2}, delivered={2, 5}, spanish={2, 5})

    report = bitext_scoring.score_pairs([row])
    assert report["missed"] == 1
    assert [c["paragraph"] for c in report["census"]["defect"]] == [2]


def test_an_original_with_no_spanish_block_to_stand_in_is_unanswerable():
    """Neither a pass nor a failure. The second rejected definition dropped these speeches
    silently; counting them as passes would flatter the figure and move the denominator the
    day the one-block shape is fixed. score_split already fails them on its own side."""
    row = _split_row(gold={3, 4}, sources={0, 1}, delivered={0, 1, 3, 4}, spanish=())

    report = bitext_scoring.score_pairs([row])
    assert report["unanswerable"] == 2
    assert report["answerable"] == 0 and report["recall"] is None
    assert bitext_scoring.score_split([row])["micro"]["recall"] == 0.0


# ---- block coverage: the absorbed-paragraph defect ---------------------------------

def test_an_absorbed_paragraph_is_told_apart_from_an_alignment_over_claim():
    """Both leave the as-delivered block although they were spoken, so score_split adds
    them into one `fp` count and neither can then be graded. Paragraph 1 had no language of
    its own, so it was never a pairing unit and left only because the run around it did;
    paragraph 4 was read and over-claimed."""
    row = _split_row(gold={3}, delivered={0, 2}, spanish={0, 1, 3, 4}, absorbed={1},
                     total=5)

    assert bitext_scoring.score_split([row])["micro"]["fp"] == 2
    lost = bitext_scoring.score_coverage([row])["spoken_lost"]
    assert [r["paragraph"] for r in lost["absorbed"]] == [1]
    assert [r["paragraph"] for r in lost["claimed"]] == [4]


def test_a_paragraph_in_neither_block_is_counted_although_fp_cannot_express_it():
    """`predicted` is `spanish - delivered`, so a paragraph missing from BOTH blocks is
    absent from `predicted` and scores as a true negative. It is the worst case there is —
    the text is in no block at all — and it is why this figure is a superset of `fp`."""
    row = _split_row(delivered={0}, spanish={0}, absorbed={1}, total=2)

    assert bitext_scoring.score_split([row])["micro"]["fp"] == 0
    lost = bitext_scoring.score_coverage([row])["spoken_lost"]
    assert [r["paragraph"] for r in lost["absorbed"]] == [1]
    assert lost["absorbed"][0]["in_spanish"] is False


def test_a_rendering_absent_from_the_delivered_block_is_not_a_loss():
    """The whole point of the delivered block. Gold `renderings` is complete, so its
    complement is what was spoken — nothing else may be subtracted from that figure."""
    row = _split_row(gold={2, 3}, delivered={0, 1}, spanish={0, 1, 2, 3}, total=4)

    assert bitext_scoring.score_coverage([row])["spoken_lost"] == {"absorbed": [],
                                                                   "claimed": []}


def test_the_mirror_direction_excuses_a_gold_pair_source_and_nothing_else():
    """A co-official original that something rendered belongs only in the as-delivered
    block, so its absence from the Spanish one is correct. Paragraph 2 is not a gold source,
    so its absence is reported — as a count, because gold pairs is partial and an
    unattributed original would look identical."""
    row = _split_row(gold={3}, sources={0}, delivered={0, 1, 2}, spanish={1, 3},
                     absorbed={2}, total=4)

    es_lost = bitext_scoring.score_coverage([row])["es_lost"]
    assert [r["paragraph"] for r in es_lost] == [2]
    assert es_lost[0]["absorbed"] is True
    assert "precision" not in es_lost[0] and "recall" not in es_lost[0]


# ---- the frozen gold set itself ----------------------------------------------------

def test_no_speech_carries_the_same_paragraph_twice():
    """Both scorers key on paragraph text, so a repeated paragraph would be counted in
    every position it appears."""
    for speech in RunBitextBenchmark().entries:
        paragraphs = speech["paragraphs"]
        assert len(set(paragraphs)) == len(paragraphs), speech["video_id"]


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
