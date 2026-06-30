"""Language detection adapter backed by py3langid.

py3langid (a pure-Python fork of langid.py, Lui & Baldwin 2012) is the only
evaluated detector that natively and cleanly separates Galician from Spanish — the
one distinction that matters for splitting co-official Diario speeches. lingua-py has
no Galician model (Galician leaks into Spanish), and fasttext is deprecated. The
identifier is restricted to the languages of the Spanish Parliament; Portuguese is
included only so its probability mass is captured and folded back into Galician
(Portuguese never legitimately appears in this corpus).

Exposes a single ``detect(text) -> str | None`` callable, injected into the pure
``domain.speeches.language_split`` logic.
"""

# Restrict the model to the languages that occur in the Diario. Portuguese is a
# decoy: py3langid sometimes reads Galician as Portuguese, so we keep it in the
# set and fold pt -> gl rather than let it bleed into another language.
_LANGUAGES = ["es", "ca", "eu", "gl", "pt"]

# Below this length py3langid is too unreliable to trust.
_MIN_CHARS = 12

_identifier = None


def _get_identifier():
    global _identifier
    if _identifier is None:
        from py3langid.langid import LanguageIdentifier, MODEL_FILE

        identifier = LanguageIdentifier.from_pickled_model(
            MODEL_FILE, norm_probs=True)
        identifier.set_languages(_LANGUAGES)
        _identifier = identifier
    return _identifier


def detect(text):
    """The ISO-639-1 code of ``text``'s language, or ``None`` if too short.

    Portuguese is folded to Galician — it never legitimately appears in the
    Diario, so a ``pt`` reading is really a misclassified Galician fragment."""
    if not text or len(text) < _MIN_CHARS:
        return None
    lang = _get_identifier().classify(text)[0]
    return "gl" if lang == "pt" else lang
