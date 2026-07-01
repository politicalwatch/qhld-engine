"""Pure passage chunking — no I/O, no deps.

Speeches are embedded as passages rather than whole: some run to ~20k chars
(beyond several embedding models' context windows) and passage-level vectors give
finer "where was X discussed" retrieval. Splitting is sentence-aware and budgeted
by characters — deliberately not tokens, since there is no single tokenizer across
the pluggable embedding providers, so a char budget is the model-agnostic choice.

Each ``SpeechText`` block is chunked independently by the caller so a chunk never
straddles a language boundary.
"""

import re


# Same sentence splitter used by the language split, kept local so this module has
# no cross-domain import.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?»])\s+")


def chunk_text(text, target_chars, overlap_chars):
    """Split ``text`` into sentence-aligned passages of about ``target_chars``.

    Sentences are packed greedily up to ``target_chars``; when a chunk closes, a
    tail of roughly ``overlap_chars`` (whole sentences) is carried into the next
    chunk to preserve context across the cut. A single sentence longer than the
    target becomes its own chunk (never mid-sentence split).
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > target_chars:
            chunks.append(" ".join(current))
            current, current_len = _overlap_tail(current, overlap_chars)
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences, overlap_chars):
    """The trailing whole sentences whose combined length is within
    ``overlap_chars``, to seed the next chunk. Returns ``(tail, tail_len)``."""
    if overlap_chars <= 0:
        return [], 0
    tail = []
    length = 0
    for sentence in reversed(sentences):
        addition = len(sentence) + (1 if tail else 0)
        if length + addition > overlap_chars:
            break
        tail.insert(0, sentence)
        length += addition
    return tail, length
