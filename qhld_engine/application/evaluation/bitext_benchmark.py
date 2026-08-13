"""Run an instrument over the bitext gold set.

The gold set asks one question per (original paragraph, candidate) comparison: is this
Spanish paragraph a rendering of that co-official one. Instruments answer it in two
different currencies — a score (lexical overlap, embedding cosine) or a yes/no (an LLM) —
so the runner emits rows in both shapes and ``bitext_scoring`` reports whichever applies.

Everything the gold set needs is inside the JSON, including the paragraph texts, so this
runs offline against a frozen corpus. That matters more than it sounds: the source text is
otherwise reproducible only by re-extracting from the Congress API, and an evaluation whose
inputs can move underneath it cannot be used to compare two runs.
"""

import json
from pathlib import Path

GOLDSET = Path(__file__).with_name("bitext_goldset.json")


class RunBitextBenchmark:
    def __init__(self, goldset: str | Path | None = None):
        self.path = Path(goldset) if goldset else GOLDSET
        data = json.loads(self.path.read_text())
        self.policy = data["policy"]
        self.entries = data["speeches"]

    def comparisons(self):
        """Every comparison the gold set defines, as
        ``(speech, source index, candidate index, is_pair, is_rendering)``.

        Candidates are the paragraphs that are NOT sources — a rendering never renders
        another rendering, and comparing a paragraph with itself is not a question anyone
        asks.

        ``is_rendering`` marks only the renderings the gold set could not attribute to a
        source, because those are the ones that are neither a positive nor a negative. A
        rendering belonging to a DIFFERENT source is a perfectly good negative for this
        one — and one of the hardest, since it is a real translation of a neighbouring
        paragraph in the same speech. Excluding those too (an earlier version of this
        method did) throws away 227 of 323 negatives, and they are exactly the ones that
        separate a working instrument from a plausible one.
        """
        for speech in self.entries:
            pairs = {(a, b) for a, b in speech["pairs"]}
            attributed = {b for _, b in speech["pairs"]}
            unattributed = set(speech["renderings"]) - attributed
            sources = sorted({a for a, _ in speech["pairs"]})
            for source in sources:
                for candidate in range(len(speech["paragraphs"])):
                    if candidate in sources:
                        continue
                    yield (speech, source, candidate,
                           (source, candidate) in pairs, candidate in unattributed)

    def run_scored(self, score_pair):
        """``score_pair(original_text, candidate_text, lang) -> float``."""
        rows = []
        for speech, source, candidate, is_pair, is_rendering in self.comparisons():
            paragraphs = speech["paragraphs"]
            rows.append({
                "lang": speech["lang"], "video_id": speech["video_id"],
                "source": source, "candidate": candidate,
                "score": float(score_pair(paragraphs[source], paragraphs[candidate],
                                          speech["lang"])),
                "is_pair": is_pair, "is_rendering": is_rendering,
                "has_renderings": bool(speech["renderings"]),
            })
        return rows

    def run_decided(self, decide_pair, workers: int = 6):
        """``decide_pair(original_text, candidate_text, lang) -> bool``.

        Concurrent because a hosted model is latency-bound, and the gold set is small
        enough that ordering does not matter.
        """
        from concurrent.futures import ThreadPoolExecutor

        items = list(self.comparisons())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decisions = list(pool.map(
                lambda item: decide_pair(item[0]["paragraphs"][item[1]],
                                         item[0]["paragraphs"][item[2]],
                                         item[0]["lang"]),
                items))
        return [{
            "lang": speech["lang"], "video_id": speech["video_id"],
            "source": source, "candidate": candidate,
            "decision": bool(decision),
            "is_pair": is_pair, "is_rendering": is_rendering,
            "has_renderings": bool(speech["renderings"]),
        } for (speech, source, candidate, is_pair, is_rendering), decision
            in zip(items, decisions)]


    def run_classifier(self, detect, **split_kwargs):
        """Run the real classifier and compare the blocks it produces against gold.

        The measure that matters, and the simplest one available: the gold set already
        records which paragraphs were never spoken, and the product question is whether
        the classifier puts those — and only those — in the rendering block. It needs no
        ordering rule of its own, because the alignment inside ``split_languages`` is
        already monotone, and no negative-class bookkeeping, which is where the pairwise
        scoring twice went wrong.

        Each row also carries where every paragraph *landed* — ``delivered`` and
        ``spanish`` as paragraph-index sets — plus the gold ``sources``, which is what
        ``score_pairs`` needs to ask the other half of the question: not only whether the
        renderings left the record of what was said, but whether the classifier saw that
        each original HAS a rendering. Both blocks are needed for that, because the answer
        is a paragraph sitting in both of them.

        ``absorbed`` records which paragraphs had no language of their own — under
        ``MIN_VOTING_CHARS`` once quotations and annotations are removed, so
        ``paragraph_language`` returns ``None`` and ``paragraph_spans`` folds them into a
        neighbouring run. Such a paragraph is never a pairing unit, so the alignment never
        decided anything about it: it leaves a block only because the run around it did.
        That is what lets ``score_coverage`` tell this defect apart from an alignment that
        really did over-claim a paragraph it had read.

        Keys on paragraph text, so a speech may not carry the same paragraph twice; the
        gold-set consistency test asserts that.
        """
        from qhld_engine.domain.speeches.language_runs import paragraph_language
        from qhld_engine.domain.speeches.language_split import split_languages

        rows = []
        for speech in self.entries:
            paragraphs = speech["paragraphs"]
            split = split_languages("\n\n".join(paragraphs), detect, speech["duration"],
                                    **split_kwargs)
            delivered = split.blocks[0].text if split.blocks else ""
            rendering = split.blocks[1].text if len(split.blocks) > 1 else ""
            in_delivered = {i for i, p in enumerate(paragraphs) if p in delivered}
            in_spanish = {i for i, p in enumerate(paragraphs) if p in rendering}
            # In the rendering block and NOT in the as-delivered one. Option B copies
            # delivered Spanish into both, so presence in the Spanish block alone says
            # nothing about whether it was spoken.
            rows.append({
                "video_id": speech["video_id"], "lang": speech["lang"],
                "gold": set(speech["renderings"]),
                "predicted": in_spanish - in_delivered,
                "sources": {a for a, _ in speech["pairs"]},
                "delivered": in_delivered, "spanish": in_spanish,
                "absorbed": {i for i, p in enumerate(paragraphs)
                             if paragraph_language(p, detect) is None},
                "total": len(paragraphs),
                "undecided": split.undecided, "blocks": len(split.blocks),
            })
        return rows


# ---- the instruments ---------------------------------------------------------------

def token_overlap():
    """What ships today: containment over whole tokens of four characters or more.

    Kept as the baseline every replacement is measured against, not because it works —
    it ranks 25/42 and gates nothing.
    """
    from qhld_engine.domain.speeches.language_runs import renders
    return lambda source, candidate, lang: renders(source, candidate)


def embedding_cosine(settings=None):
    """Cosine between the two paragraphs, through whatever embedder is configured.

    Uses the same factory the indexer does, so the evaluation measures the lane the
    product would actually run on rather than a model chosen for the benchmark.
    """
    from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env

    embedder = create_embedder_from_env(settings)
    cache: dict[str, list[float]] = {}

    def vector(text):
        if text not in cache:
            raw = embedder.embed_documents([text])[0]
            norm = sum(v * v for v in raw) ** 0.5 or 1.0
            cache[text] = [v / norm for v in raw]
        return cache[text]

    def score(source, candidate, lang):
        a, b = vector(source), vector(candidate)
        return sum(x * y for x, y in zip(a, b))

    return score


DEFAULT_PROMPT = """Eres un lingüista revisando el Diario de Sesiones del Congreso \
español. El Diario publica cada discurso en lengua cooficial junto con su interpretación \
al castellano, que reformula, reordena y a veces resume.

Texto A ({lang}):
{a}

Texto B (castellano):
{b}

¿Es el texto B una traducción o interpretación al castellano del texto A?
Responde SOLO "SI" si B reproduce el contenido de A, aunque lo reformule o reordene.
Responde SOLO "NO" si B trata el mismo tema pero dice cosas distintas, o si es otra parte \
del discurso. Tratar el mismo asunto NO basta: tiene que decir lo mismo."""

_LANG_NAMES = {"ca": "catalán", "gl": "gallego", "eu": "euskera"}


def llm_judgement(prompt: str | None = None, settings=None, max_chars: int = 2500):
    """Ask a model whether one paragraph renders the other.

    The prompt is a parameter because it, not the model, is the operating point: the same
    model on the same pairs moves from recall 0.70 / specificity 0.98 to recall 0.98 /
    specificity 0.54 on a rewrite. Tuning it is the point of running this benchmark.
    """
    from qhld_ai.infrastructure.llm.factory import create_llm_from_env

    llm = create_llm_from_env(settings)
    template = prompt or DEFAULT_PROMPT

    def decide(source, candidate, lang):
        reply = llm.invoke(template.format(a=source[:max_chars], b=candidate[:max_chars],
                                           lang=_LANG_NAMES.get(lang, lang)))
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        return text.strip().upper().startswith(("SI", "SÍ"))

    return decide
