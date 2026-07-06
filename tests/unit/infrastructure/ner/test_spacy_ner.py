"""Light test for the spaCy NER adapter — loads the real es_core_news_lg model.

The model is a main dependency (mention NER runs in the daily extract), so this
runs by default; guarded by find_spec so a stripped env skips instead of erroring.
"""

from importlib.util import find_spec

import pytest

pytestmark = pytest.mark.unit

_HAS_MODEL = find_spec("spacy") and find_spec("es_core_news_lg")
pytestmark = [pytest.mark.unit,
              pytest.mark.skipif(not _HAS_MODEL, reason="spaCy model not installed")]


def test_person_spans_extracts_people_not_orgs():
    from qhld_engine.infrastructure.config.settings import Settings
    from qhld_engine.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    text = ("El señor Feijóo criticó a Pedro Sánchez. El Gobierno y el "
            "Partido Popular no se pusieron de acuerdo.")
    spans = ner.person_spans(text)
    # People are captured; the ORG mentions (Gobierno, Partido Popular) are not.
    assert any("Feijóo" in s for s in spans)
    assert any("Sánchez" in s for s in spans)
    assert not any("Partido Popular" == s for s in spans)


def test_empty_text_returns_no_spans():
    from qhld_engine.infrastructure.config.settings import Settings
    from qhld_engine.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    assert ner.person_spans("") == []
