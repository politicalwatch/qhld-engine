"""Port for query understanding: parse a natural-language search query into
structured filters plus the residual semantic query.

``ParsedQuery`` is a Pydantic model so it doubles as the LLM structured-output
schema (the LLM adapter binds it via ``with_structured_output``) and the port's
return type. The field descriptions are the extraction spec the LLM reads, so
keep them precise.

Entity *resolution* (fuzzy-matching ``speaker`` to a corpus value, a party name
to a group token, ISO dates to the YYYYMMDD range) is a separate application
concern — this port only extracts what the user asked for, verbatim-ish.
"""

from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    semantic_query: str = Field(
        description=(
            "The thematic content to search for — what the speech is ABOUT — with "
            "speaker, mentioned-person, group/party and date constraints removed. "
            "Empty string if the query is purely a filter with no topic."))
    speakers: list[str] | None = Field(
        default=None,
        description=(
            "Every person named as one who SPEAKS/intervenes, each given by proper "
            "name (e.g. 'María Jesús Montero'), one list item per person. Null if "
            "speakers are only referred to by office/title, or if none is specified."))
    mentioned_persons: list[str] | None = Field(
        default=None,
        description=(
            "Every person that the speech must MENTION or refer to, who is NOT the "
            "speaker (e.g. 'discursos que mencionen a Zapatero y a Rajoy' → "
            "['Zapatero', 'Rajoy']). Each given by proper name, one list item per "
            "person. Null if the query does not ask for mentioned persons."))
    mentions_mode: Literal["all", "any"] = Field(
        default="all",
        description=(
            "How multiple mentioned persons combine: 'all' when the speech must "
            "mention every one of them (connective 'y'/'e', or a single person), "
            "'any' when mentioning one suffices (connective 'o'/'u')."))
    speaker_title: str | None = Field(
        default=None,
        description=(
            "The speaker's office or role when referred to by title instead of name "
            "(e.g. 'ministra de economía', 'presidente del gobierno'). Null otherwise."))
    groups_or_parties: list[str] | None = Field(
        default=None,
        description=(
            "Every parliamentary group or political party named as a filter (e.g. "
            "'PSOE', 'Grupo Socialista', 'Partido Popular'), one list item per "
            "group/party. Null if none."))
    date_from: str | None = Field(
        default=None,
        description=(
            "Start of the date range in ISO format YYYY-MM-DD. Resolve relative "
            "expressions ('el último año', 'últimos tres meses', 'en 2024') against "
            "the provided current date. Null if the query has no time constraint."))
    date_to: str | None = Field(
        default=None,
        description="End of the date range in ISO format YYYY-MM-DD. Null if open-ended.")
    lang: str | None = Field(
        default=None,
        description=(
            "Language code (es/ca/gl/eu) only if the user explicitly asks for a "
            "language. Null otherwise."))
    legislature: str | None = Field(
        default=None,
        description="Legislature number only if explicitly stated. Null otherwise.")


class QueryParserPort(Protocol):
    def parse(self, query: str, today: date) -> ParsedQuery:
        """Extract structured filters + residual semantic query from ``query``.
        ``today`` anchors relative-date resolution (injected, never a wall-clock)."""
        ...
