"""Unit tests for the interventions-search URLs.

The endpoint accepts every filter in the query string only. Sent in the POST body
they are accepted and silently ignored, and page 1 is served again — which for
pagination means an initiative with more than one page of interventions is
truncated at the page size with no error to show for it. These tests pin the page
number, and the date range, into the URL.
"""

from types import SimpleNamespace

import pytest

from qhld_engine.extractors.spain import congress_api as mod

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _legislature(monkeypatch):
    monkeypatch.setattr(mod, "get_settings",
                        lambda: SimpleNamespace(id_legislatura=15))


def _builder():
    return mod.CongressUrlBuilder()


def test_initiative_url_carries_the_page_number():
    # The regression: without this the page lives only in the form data, the
    # portal ignores it, and every page after the first repeats the first.
    url = _builder().for_video("210/000039", 2)

    assert "_intervenciones_paginaActual=2" in url
    assert "_intervenciones_id_iniciativa=210/000039" in url


def test_initiative_url_defaults_to_the_first_page():
    assert "_intervenciones_paginaActual=1" in _builder().for_video("172/000001")


def test_each_page_of_an_initiative_gets_a_distinct_url():
    builder = _builder()
    pages = {builder.for_video("210/000039", n) for n in (1, 2, 3)}

    assert len(pages) == 3


