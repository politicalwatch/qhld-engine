import pytest

from qhld_engine.extractors.spain.initiative_extractors.vote_extractor import extract_json_link

pytestmark = pytest.mark.unit


def test_extract_json_link_returns_none_for_unparseable_querystring_href():
    href = "https://www.congreso.es/opendata/votaciones?targetDate=24/09/2024&targetLegislatura=XV"
    assert extract_json_link(f'<a href="{href}">JSON</a>') is None


def test_extract_json_link_parses_clean_json_href():
    href = "https://www.congreso.es/webpublica/opendata/votaciones/Leg15/Sesion061/20240924/Votacion001/VOT_20240924203510.json"
    assert extract_json_link(f'<a href="{href}" download="">JSON</a>') == href
