"""LLM query parser.

Binds the ``ParsedQuery`` schema to a chat model via ``with_structured_output`` so
the model returns a validated object, not free text. The chat model is built from
the existing ``create_llm_from_env`` factory on a Settings copy whose ``llm_*`` are
overridden by the decoupled ``query_parser_llm_*`` (empty => fall back to the main
llm settings), so the parser can run on a different model than any future
answer-synthesis. The model is built lazily on first ``parse`` so importing this
module stays cheap.
"""

from datetime import date

from qhld_engine.domain.ports.query_parser import ParsedQuery
from qhld_engine.infrastructure.config.settings import Settings

from .factory import _register

_SYSTEM = """\
Eres un analizador de consultas para un buscador de intervenciones parlamentarias \
del Congreso de los Diputados de España. Dada una consulta en lenguaje natural, \
extrae los filtros estructurados y la consulta semántica residual.

Reglas:
- semantic_query: SOLO el tema o contenido de la intervención (de qué trata), sin \
las restricciones de orador, persona mencionada, grupo/partido ni fechas. Cadena \
vacía si la consulta no tiene tema.
- speakers: TODAS las personas que INTERVIENEN, indicadas por su nombre propio, un \
elemento de la lista por persona (p. ej. 'de Pedro Sánchez y Yolanda Díaz' → \
['Pedro Sánchez', 'Yolanda Díaz']). Null si se refiere a ellas por su cargo, o si \
no se especifica orador.
- mentioned_persons: TODAS las personas que la intervención debe MENCIONAR o \
nombrar, que NO son quien interviene, un elemento por persona (p. ej. 'discursos \
que mencionen a Zapatero y a Rajoy' → ['Zapatero', 'Rajoy']). Null si la consulta \
no pide personas mencionadas.
- mentions_mode: cómo combinar varias personas mencionadas: 'all' si deben \
aparecer todas (conjunción 'y'/'e', o una sola persona: 'que mencionen a Ayuso y \
Putin' → 'all'); 'any' si basta con una (disyunción 'o'/'u': 'que mencionen a \
Ayuso o Putin' → 'any').
- speaker_title: el cargo del orador cuando se le nombra por su cargo en lugar de \
por su nombre (p. ej. 'ministra de economía'). Null en otro caso.
- groups_or_parties: TODOS los grupos parlamentarios o partidos políticos como \
filtro, un elemento por grupo/partido (p. ej. 'del PSOE y del PP' → ['PSOE', \
'PP']; 'del Grupo Socialista' → ['Grupo Socialista']). Null si no hay.
- date_from / date_to: rango de fechas en formato ISO YYYY-MM-DD. Resuelve las \
expresiones relativas ('el último año', 'últimos tres meses', 'en 2024') tomando \
como fecha actual {today}. Null si no hay restricción temporal.
- lang, legislature: solo si se indican explícitamente.
"""


class LLMQueryParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._structured = None

    def _model(self):
        if self._structured is None:
            from qhld_engine.infrastructure.llm.factory import create_llm_from_env

            overrides = {}
            if self.settings.query_parser_llm_provider:
                overrides["llm_provider"] = self.settings.query_parser_llm_provider
            if self.settings.query_parser_llm_model:
                overrides["llm_model"] = self.settings.query_parser_llm_model
            llm_settings = self.settings.model_copy(update=overrides)
            chat = create_llm_from_env(llm_settings)
            self._structured = chat.with_structured_output(ParsedQuery)
        return self._structured

    def parse(self, query: str, today: date) -> ParsedQuery:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SYSTEM.format(today=today.isoformat())),
            HumanMessage(content=query),
        ]
        return self._model().invoke(messages)


@_register("llm")
def create(settings: Settings) -> LLMQueryParser:
    return LLMQueryParser(settings)
