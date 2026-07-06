"""Rule-based query parser — the baseline against the LLM parser.

A conventional NLP pipeline: spaCy ``es_core_news_lg`` NER (PER → speaker,
ORG/MISC → group/party) + light regex for titles/languages/legislature + a
relative-date regex backed by ``dateparser`` for absolute dates. It is
deliberately *not* a re-implementation of the LLM's reasoning — it shows where an
off-the-shelf rule/NER stack trails a structured-output LLM: it cannot tell a
speaker from a merely-mentioned person, cannot classify a title vs a name, and
``dateparser`` misses most Spanish relative *ranges* ("los últimos tres meses").

spaCy/dateparser are lazy-imported so this module stays importable (and the
factory registration stays cheap) even when they are not installed; only ``parse``
needs them.
"""

import re
from datetime import date

from qhld_engine.domain.ports.query_parser import ParsedQuery

from .factory import _register

# Capture the whole office phrase (multi-word ministries), stopping at a topic/date
# connector so "ministra de transición ecológica sobre energía" → "ministra de
# transición ecológica".
_TITLE_RE = re.compile(
    r"\b(?:ministr[oa]|vicepresident[ae]|president[ae]|secretari[oa])\s+de[l]?\s+"
    r".+?(?=\s+(?:sobre|acerca|en\s|que|durante|desde|,)|$)", re.IGNORECASE)

_LANGS = {
    "gallego": "gl", "galego": "gl", "catalán": "ca", "catalan": "ca",
    "euskera": "eu", "vasco": "eu", "vascuence": "eu", "castellano": "es",
    "español": "es",
}
_LANG_RE = re.compile(r"\ben\s+(" + "|".join(_LANGS) + r")\b", re.IGNORECASE)
_LEG_RE = re.compile(r"\blegislatura\s+(\d+)\b", re.IGNORECASE)

_NUMBER_WORDS = {
    "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}
_UNITS = {"día": "days", "dia": "days", "semana": "weeks", "mes": "months", "año": "years"}
# Unit suffix allows the irregular Spanish plural "meses" as well as "-s".
_REL_RANGE_RE = re.compile(
    r"\búltim[oa]s?\s+(?:(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+)?"
    r"(día|dia|semana|mes|año)(?:es|s)?\b", re.IGNORECASE)

# Query scaffolding stripped from the residual semantic query.
_SCAFFOLD_RE = re.compile(
    r"\b(intervenciones?|discursos?|debates?|qué|que|ha|han|dicho|dijo|sobre|acerca|"
    r"de|del|la|el|los|las|en|un|una)\b", re.IGNORECASE)


class RuleBasedQueryParser:
    def __init__(self, settings=None):
        self.settings = settings
        self._nlp = None

    def _model(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load("es_core_news_lg")
        return self._nlp

    def parse(self, query: str, today: date) -> ParsedQuery:
        doc = self._model()(query)
        spans = []
        speaker = group = None
        for ent in doc.ents:
            if ent.label_ == "PER" and speaker is None:
                speaker, _ = ent.text, spans.append(ent.text)
            elif ent.label_ in ("ORG", "MISC") and group is None:
                group, _ = ent.text, spans.append(ent.text)

        title = self._match(_TITLE_RE, query, spans)
        lang_word = self._match(_LANG_RE, query, spans, group=1)
        lang = _LANGS.get(lang_word.lower()) if lang_word else None
        legislature = self._match(_LEG_RE, query, spans, group=1)
        date_from, date_to = self._dates(query, today, spans)

        return ParsedQuery(
            semantic_query=self._residual(query, spans),
            speaker=speaker,
            speaker_title=title,
            group_or_party=group,
            date_from=date_from,
            date_to=date_to,
            lang=lang,
            legislature=legislature)

    @staticmethod
    def _match(regex, query, spans, group=0):
        """Return the matched text (or capture ``group``) and record the full match
        span for stripping; ``None`` if no match."""
        m = regex.search(query)
        if not m:
            return None
        spans.append(m.group(0))
        return m.group(group)

    def _dates(self, query, today, spans):
        rel = _REL_RANGE_RE.search(query)
        if rel:
            spans.append(rel.group(0))
            count = self._count(rel.group(1))
            unit = _UNITS[rel.group(2).lower()]
            return self._shift(today, count, unit).isoformat(), today.isoformat()
        # No relative range → off-the-shelf dateparser for absolute dates.
        found = self._search_dates(query, today)
        if found:
            spans.extend(text for text, _ in found)
            dates = sorted(dt.date() for _, dt in found)
            lo, hi = dates[0], dates[-1]
            return lo.isoformat(), (hi.isoformat() if hi != lo else None)
        return None, None

    @staticmethod
    def _count(token):
        if not token:
            return 1
        return int(token) if token.isdigit() else _NUMBER_WORDS.get(token.lower(), 1)

    @staticmethod
    def _shift(today, count, unit):
        from dateutil.relativedelta import relativedelta

        return today - relativedelta(**{unit: count})

    @staticmethod
    def _search_dates(query, today):
        from datetime import datetime

        from dateparser.search import search_dates

        return search_dates(
            query, languages=["es"],
            settings={
                "RELATIVE_BASE": datetime(today.year, today.month, today.day),
                "PREFER_DATES_FROM": "past"}) or []

    @staticmethod
    def _residual(query, spans):
        residual = query
        for span in spans:
            if span:
                residual = re.sub(re.escape(span), " ", residual, flags=re.IGNORECASE)
        residual = _SCAFFOLD_RE.sub(" ", residual)
        return re.sub(r"\s+", " ", residual).strip(" ,.")


@_register("rule_based")
def create(settings) -> RuleBasedQueryParser:
    return RuleBasedQueryParser(settings)
