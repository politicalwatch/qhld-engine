"""Benchmark service for the intent gate.

Isolates the gate the way the ``--floors`` sweep isolates the relevance floor: it
runs ``parse`` and then the gate itself, and stops. No retrieval, no reranker,
one parser call per query per repeat. Going through ``execute`` instead would
drag the whole search in — slow, paid, and nondeterministic in its own right
(grouped pool membership is not monotonic in ``limit``), which would mix noise
that is not the gate's into the gate's number.

Three arms, scored identically and read in different directions:

- ``legitimate`` — real searches (``gate_queryset.json``, ``expected: pass``).
  Every refusal is a FALSE POSITIVE.
- ``non-search`` — inputs that are not searches at all (same file,
  ``expected: refuse``): an instruction, a request to generate content, a
  question to the assistant, an attempt to change its behaviour. Every pass is a
  MISS, and an expensive one: these carry real parliamentary topics, so nothing
  downstream stops them. A miss here answers a request we never honoured with
  genuinely relevant speeches.
- ``junk`` — the off-domain probes, read straight out of the ``offdomain``
  dimension of ``queryset.json``. A pass here is also a miss, but a cheaper one:
  the relevance floor is a backstop (measured 2026-08-16 — all seven of the
  gate's misses return zero results), so the cost is a paid pipeline and the
  wrong error message rather than visible junk.

The junk rows are deliberately NOT copied here — they were hardened 5 -> 24 once
already and nobody re-ran the floor calibration against the new set, so a second
copy is exactly the mistake that is already on record.

Calls ``NaturalSearchSpeeches._prepare`` directly, which is private on purpose:
that method IS the gate, and the point of this harness is to measure it without
anything downstream of it running. It also builds the FULL resolver the product
builds — ``RunParseBenchmark`` deliberately omits the deputies catalog (mentions
are not a scored slot there), and reusing that here would make every mention and
speaker query resolve to nothing and land at the empty-parse raise as a harness
artifact rather than a finding.

Runs live: Qdrant for the corpus vocabulary, Mongo for the catalogs, and the LLM
for the parse. Run it on the host per the standing benchmark-on-host preference:

    MONGO_HOST=localhost MONGO_PORT=27018 QDRANT_HOST=localhost \\
    uv run --no-sync qhld eval gate --repeats 5

Every setting that names the collection must be in the environment — see
``check_vocabulary``.
"""

import json
import os
import re
import time
from datetime import date

from qhld_engine.domain.evaluation.gate_scoring import (
    ARMS, JUNK, LEGITIMATE, NON_SEARCH, PASS, REFUSED_EMPTY, REFUSED_FLAG,
    REFUSED_LANGUAGE, UNSUPPORTED_LANGUAGE)

DEFAULT_QUERYSET = os.path.join(os.path.dirname(__file__), "gate_queryset.json")
JUNK_QUERYSET = os.path.join(os.path.dirname(__file__), "queryset.json")
JUNK_DIMENSION = "offdomain"

# The junk set records its own taxonomy in prose, as a marker each note OPENS
# with ("off-domain probe:", "NEAR-DOMAIN junk,", "injection-shaped probe:",
# "gibberish probe:"). Deriving the class from that keeps queryset.json the single
# source of truth — the alternative is a lookup table here that silently goes
# stale the next time the probes are hardened.
#
# Anchored to the start on purpose: the notes also DISCUSS the other classes in
# passing. J5 declares itself off-domain and then explains that sport would have
# been "a near-domain trap, not junk" — an unanchored search files it under the
# class it exists to contrast itself with. Anything unrecognised falls back to
# the dimension name rather than being dropped.
_JUNK_CLASSES = (
    ("injection", re.compile(r"^\s*injection", re.I)),
    ("gibberish", re.compile(r"^\s*gibberish", re.I)),
    ("near-domain", re.compile(r"^\s*near-domain", re.I)),
    ("off-domain", re.compile(r"^\s*off-domain", re.I)),
)


def _junk_class(notes: str) -> str:
    for name, pattern in _JUNK_CLASSES:
        if pattern.search(notes or ""):
            return name
    return JUNK_DIMENSION


def load_queryset(path=DEFAULT_QUERYSET):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_junk(path=JUNK_QUERYSET, dimension=JUNK_DIMENSION):
    """The junk arm, read from the retrieval query set rather than duplicated."""
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [
        {"id": row["id"], "query": row["query"], "expected": "refuse",
         "class": _junk_class(row.get("notes", "")), "notes": row.get("notes", "")}
        for row in rows if row.get("dimension") == dimension
    ]


def _parse_date(iso):
    year, month, day = (int(part) for part in iso.split("-"))
    return date(year, month, day)


class VocabularyError(RuntimeError):
    """The resolver read an empty corpus vocabulary, so no number it produces means
    anything."""


class RunGateBenchmark:
    def __init__(self, queryset_path=DEFAULT_QUERYSET, junk_path=JUNK_QUERYSET,
                 settings=None):
        from qhld_ai.infrastructure.config.settings import get_settings

        data = load_queryset(queryset_path)
        self.today = _parse_date(data["today"])
        rows = data["queries"]
        self.legitimate = [row for row in rows if row["expected"] == "pass"]
        self.non_search = [row for row in rows if row.get("expected_reason")
                           == "not_a_speech_search"]
        self.unsupported_language = [row for row in rows if row.get("expected_reason")
                                     == "unsupported_language"]
        self.junk = load_junk(junk_path)
        base = settings or get_settings()
        # Nothing is ever retrieved, so the reranker is irrelevant — and pinning it
        # to noop guarantees no local cross-encoder is loaded just to be unused.
        # Safe for collection identity: the name carries provider, model, dimension,
        # sparse branch and compression, never the reranker.
        self.settings = base.model_copy(update={"reranker_provider": "noop"})
        self._service = None

    def service(self):
        if self._service is None:
            from qhld_ai.application.search.natural_search import NaturalSearchSpeeches

            self._service = NaturalSearchSpeeches(settings=self.settings)
        return self._service

    def collection(self) -> str:
        from qhld_ai.infrastructure.vectorstore.naming import collection_name

        service = self.service()
        dim = len(service.search.embedder.embed_query("probe"))
        return collection_name(self.settings, dim)

    def check_vocabulary(self) -> tuple[str, int]:
        """Fail loudly when the resolver has nothing to resolve against.

        A reader that omits any setting the collection name is built from resolves
        a collection the indexer never wrote, and Qdrant answers with an empty set
        rather than an error. Here that does not merely weaken the result, it
        MANUFACTURES it: with an empty vocabulary nothing resolves, every
        legitimate query reaches the empty-parse raise, and the harness reports a
        false-positive rate near 100% that is entirely its own doing.
        """
        collection = self.collection()
        hint = ("Check EMBEDDING_MODEL, SPARSE_PROVIDER and QDRANT_QUANTIZATION against "
                "the collections that actually exist.")
        try:
            speakers = self.service().search.store.distinct_values(collection, "speaker")
        except Exception as exc:  # noqa: BLE001 - any read failure is the same problem here
            # A missing collection surfaces as a 404 from the store. Same cause as the
            # empty-vocabulary case below, so it gets the same explanation rather than a
            # bare traceback.
            raise VocabularyError(
                f"could not read the vocabulary of collection {collection!r} "
                f"({type(exc).__name__}: {exc}). {hint}") from exc
        if not speakers:
            raise VocabularyError(
                f"collection {collection!r} yielded no speaker values — the resolver has "
                "an empty vocabulary, so every query would fail to resolve and the gate "
                f"would look far worse than it is. {hint}")
        return collection, len(speakers)

    def entries(self, arm):
        try:
            return {LEGITIMATE: self.legitimate, NON_SEARCH: self.non_search,
                    UNSUPPORTED_LANGUAGE: self.unsupported_language,
                    JUNK: self.junk}[arm]
        except KeyError:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}") from None

    def run(self, arm=LEGITIMATE, llm_provider=None, llm_model=None,
            reasoning_effort=None):
        """One pass over one arm; a row per query carrying the gate's outcome."""
        from qhld_ai.domain.errors import NotASpeechQuery, UnsupportedLanguage
        from qhld_ai.domain.ports.query_parser import ParsedQuery

        service = self.service()
        parser = self._parser(llm_provider, llm_model, reasoning_effort)
        entries = self.entries(arm)
        rows = []
        for entry in entries:
            start = time.perf_counter()
            try:
                parsed = parser.parse(entry["query"], self.today)
                parse_error = None
            except Exception as exc:  # noqa: BLE001
                # A parser that cannot emit schema-valid output must not abort the
                # run. An empty parse is what the product would carry forward, and
                # the gate's own empty-parse rule is what would then fire — so score
                # it exactly as the product would behave.
                parsed = ParsedQuery(semantic_query="")
                parse_error = type(exc).__name__
            try:
                _, filters, semantic, _ = service._prepare(entry["query"], parsed)
                outcome = PASS
                # Free diagnostic: a legitimate query that passes the gate but has
                # no topic left browses instead of searching. Not a refusal, but
                # not the same answer either.
                route = "search" if semantic else "browse"
            except UnsupportedLanguage:
                # Caught BEFORE NotASpeechQuery: both descend from SearchRefused,
                # and an ordering mistake here would silently file every language
                # refusal under the intent gate.
                outcome = REFUSED_LANGUAGE
                route = None
                filters = None
            except NotASpeechQuery:
                # Both intent-gate raise sites throw the same error, so the site is
                # read off the parse: the flag is the parser's own verdict, and
                # anything else that raises got through the flag and died on an
                # empty parse.
                outcome = REFUSED_FLAG if not parsed.is_speech_search else REFUSED_EMPTY
                route = None
                filters = None
            rows.append({
                **entry,
                "outcome": outcome,
                "route": route,
                "latency": time.perf_counter() - start,
                "parse_error": parse_error,
                "pred_topic": parsed.semantic_query,
                "pred_filters": filters,
                "is_speech_search": parsed.is_speech_search,
            })
        return rows

    def _parser(self, llm_provider=None, llm_model=None, reasoning_effort=None):
        from qhld_ai.infrastructure.queryparsing.factory import create_query_parser_from_env

        update = {}
        if llm_provider:
            update["query_parser_llm_provider"] = llm_provider
        if llm_model:
            update["query_parser_llm_model"] = llm_model
        if reasoning_effort:
            update["query_parser_llm_reasoning_effort"] = reasoning_effort
        return create_query_parser_from_env(
            self.settings.model_copy(update=update) if update else self.settings)

    def model_label(self) -> str:
        return (f"{self.settings.query_parser_llm_provider}:"
                f"{self.settings.query_parser_llm_model}")
