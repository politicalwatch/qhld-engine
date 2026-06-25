"""Unit regression tests for the Paraguay extractor's pydantic-migration bugs.

The mongoengine→pydantic move (qhld-data v2.0.0) changed two semantics on the
``Initiative`` model that the Paraguay extractor still relied on the old way:

- ``x in initiative`` no longer works (pydantic ``__iter__`` yields ``(k, v)``
  tuples and there is no ``__contains__``), so membership checks are always False.
- assigning an *undeclared* field raises ``ValueError`` (the model is fixed, no
  ``extra="allow"``).

These tests exercise the extractor's private helpers directly — no DB, no network.
``LegislativePeriod.get`` is stubbed so ``__init__`` does not hit the API.
"""

import pytest

from tipi_data.models.initiative import Initiative

from qhld_engine.extractors.paraguay import initiatives as paraguay_initiatives
from qhld_engine.extractors.paraguay.initiatives import InitiativesExtractor

pytestmark = pytest.mark.unit


@pytest.fixture
def extractor(monkeypatch):
    monkeypatch.setattr(
        paraguay_initiatives.LegislativePeriod, "get", lambda self: "2023-2028"
    )
    return InitiativesExtractor()


# --- Bug A: __untag set undeclared topics/tags -> ValueError ------------------

def test_untag_resets_tagged_to_empty_list(extractor):
    """``__untag`` should clear tagging via the model's own ``untag()``; pre-fix it
    set the undeclared ``topics``/``tags`` fields and raised ``ValueError``."""
    initiative = Initiative(id="1")
    initiative.tagged = []  # baseline; would normally hold Tagged entries

    extractor._InitiativesExtractor__untag(initiative)

    assert initiative.tagged == []


# --- Bug B: __has_content membership check always False -----------------------

def test_has_content_true_when_initiative_has_loaded_content(extractor, monkeypatch):
    """Content just loaded onto the in-memory initiative must count; pre-fix
    ``"content" in initiative`` was always False so the method returned False and
    the caller wiped the content back to ``[""]``."""
    monkeypatch.setattr(
        paraguay_initiatives.Initiatives,
        "get",
        lambda _id: (_ for _ in ()).throw(Exception("does not exist")),
    )
    initiative = Initiative(id="1", content=["algo de contenido"])

    assert extractor._InitiativesExtractor__has_content(initiative) is True


def test_has_content_false_when_no_content_anywhere(extractor, monkeypatch):
    """No saved doc and no in-memory content -> the no-content branch should fire."""
    monkeypatch.setattr(
        paraguay_initiatives.Initiatives,
        "get",
        lambda _id: (_ for _ in ()).throw(Exception("does not exist")),
    )
    initiative = Initiative(id="1")

    assert extractor._InitiativesExtractor__has_content(initiative) is False
