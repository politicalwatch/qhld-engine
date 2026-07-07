"""Benchmark service for mention extraction.

Runs the ``MentionTagger`` over a frozen, hand-labelled gold set (speeches keyed by
``_id``), timing each speech, and returns rows the pure ``mentions_scoring`` module
aggregates. Runs live (needs Mongo for the speeches + deputies catalog + the spaCy
model); no Qdrant/LLM. The gold set lives in the repo (``mentions_goldset.json``)
beside the parse/retrieval sets.
"""

import json
import os
import time

DEFAULT_GOLDSET = os.path.join(os.path.dirname(__file__), "mentions_goldset.json")


def load_goldset(path=DEFAULT_GOLDSET):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["speeches"]


class RunMentionsBenchmark:
    def __init__(self, goldset_path=DEFAULT_GOLDSET, tagger=None):
        self.entries = load_goldset(goldset_path)
        self._tagger = tagger

    def _tagger_obj(self):
        if self._tagger is None:
            from qhld_engine.application.speeches.mention_tagging import MentionTagger
            from tipi_data.repositories.deputies import Deputies

            self._tagger = MentionTagger(Deputies.get_all())
        return self._tagger

    def run(self):
        """Return a scored row per gold-set speech: predicted vs gold names, split into
        deputies and non-deputies. Predictions are split by ``person_type`` so the deputy
        metric is scored on exactly the deputy predictions (unchanged basis) and the new
        non-deputy figures are scored separately."""
        from tipi_data.repositories.speeches import Speeches

        tagger = self._tagger_obj()
        rows = []
        for entry in self.entries:
            speech = Speeches.get(entry["speech_id"])
            start = time.perf_counter()
            mentions = tagger.tag_speech(speech)
            latency = time.perf_counter() - start
            rows.append({
                **entry,
                "pred_deputies": [m.name for m in mentions if m.person_type == "deputy"],
                "gold_deputies": entry["expected_deputies"],
                "pred_non_deputies": [
                    m.name for m in mentions if m.person_type != "deputy"],
                "gold_non_deputies": entry.get("expected_non_deputies", []),
                "latency": latency,
            })
        return rows
