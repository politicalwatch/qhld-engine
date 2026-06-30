"""Pure language-split logic — no HTTP, no DB, no detector dependency.

Co-official-language speeches are published in the Diario de Sesiones as the full
original (Galician/Catalan/Basque) immediately followed by its full Spanish
interpretation, concatenated under one speaker heading with no delimiter. The
structure is reliably two clean blocks ``[original][castellano]``, so a speech is
split at the *single* major language boundary into an original block and (when a
translation is present) a Spanish block.

The language detector is injected as a callable — ``detect(str) -> str | None`` —
so this module stays a pure, trivially-testable string transform; the py3langid
adapter that backs it lives in the infrastructure layer.
"""

import re


# Co-official languages of Spain that appear (alongside Spanish) in the Diario.
CO_LANGS = frozenset({"ca", "eu", "gl"})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?»])\s+")

# The detector is unreliable on very short fragments; below this, a sentence
# contributes no language vote (it still occupies its position in the sequence).
_MIN_CHARS = 12

# A side (co-official original, or Spanish translation) shorter than this fraction
# of the speech is treated as absent rather than as a real block.
_MIN_RATIO = 0.15


def split_languages(text, detect):
    """Split ``text`` into per-language blocks.

    Returns ``(original_language, blocks)`` where ``blocks`` is a list of
    ``(lang, text, original)`` tuples. Monolingual text yields a single block
    ``[(dominant, text, True)]``; a bilingual co-official speech yields
    ``[(orig_lang, original, True), ("es", castellano, False)]``.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    if not sentences:
        return "es", []

    langs = [detect(s) if len(s) >= _MIN_CHARS else None for s in sentences]
    total = sum(len(s) for s in sentences)

    co_chars = sum(len(s) for s, lang in zip(sentences, langs) if lang in CO_LANGS)
    if not co_chars or co_chars / total < _MIN_RATIO:
        dominant = _dominant_lang(sentences, langs)
        return dominant, [(dominant, text, True)]

    orig_lang = _argmax_co_lang(sentences, langs)
    k = _boundary(sentences, langs)
    original = " ".join(sentences[:k])
    castellano = " ".join(sentences[k:])

    # k == 0 means the objective put everything on the Spanish side (an
    # es-dominant speech with no real original boundary) → treat as monolingual.
    if not original or len(castellano) / total < _MIN_RATIO:
        return orig_lang, [(orig_lang, text, True)]
    return orig_lang, [(orig_lang, original, True), ("es", castellano, False)]


def _char_weights(sentences, langs):
    weights = {}
    for sentence, lang in zip(sentences, langs):
        if lang:
            weights[lang] = weights.get(lang, 0) + len(sentence)
    return weights


def _dominant_lang(sentences, langs):
    weights = _char_weights(sentences, langs)
    if not weights:
        return "es"
    return max(weights, key=weights.get)


def _argmax_co_lang(sentences, langs):
    weights = {
        lang: chars
        for lang, chars in _char_weights(sentences, langs).items()
        if lang in CO_LANGS
    }
    return max(weights, key=weights.get)


def _boundary(sentences, langs):
    """The split point ``k`` maximizing (co-official chars in ``sents[:k]``) plus
    (Spanish chars in ``sents[k:]``).

    This is the key trick: it finds the single cut that best separates the
    original (front-loaded co-official) from its trailing Spanish interpretation.
    A naive "earliest k where the suffix is ≥85% Spanish" rule fails — the large
    trailing Spanish block pulls the cut into the original.
    """
    lengths = [len(s) for s in sentences]
    is_co = [lang in CO_LANGS for lang in langs]
    is_es = [lang == "es" for lang in langs]

    co_prefix = 0
    es_suffix = sum(length for length, es in zip(lengths, is_es) if es)
    best_k, best_score = 0, es_suffix
    for k in range(1, len(sentences) + 1):
        length = lengths[k - 1]
        if is_co[k - 1]:
            co_prefix += length
        if is_es[k - 1]:
            es_suffix -= length
        score = co_prefix + es_suffix
        if score > best_score:
            best_score, best_k = score, k
    return best_k
