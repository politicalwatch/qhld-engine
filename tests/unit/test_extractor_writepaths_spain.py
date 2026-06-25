"""Unit regression tests for the extractor write-path ``_id`` bug — no DB.

The pymongo/pydantic migration made ``tipi_data`` ``MongoModel.id`` a *required*
field (``Field(alias="_id")``, no default). The extractors keep the old
mongoengine habit of constructing the model empty and setting ``id`` last, which
now raises ``ValidationError: _id Field required`` at construction time. These
two paths reach the bug without any DB or network, so they live in the unit tier.
"""

from types import SimpleNamespace

import pytest

from tipi_data.utils import generate_id

from qhld_engine.extractors.spain.deputy_extractors.deputy_extractor import (
    DeputyExtractor,
)
from qhld_engine.extractors.spain.initiative_extractors import video_extractor
from qhld_engine.extractors.spain.initiative_extractors.video_extractor import (
    VideoExtractor,
)

pytestmark = pytest.mark.unit


# --- Spain DeputyExtractor: builds the Deputy in __init__ ---------------------

def test_deputy_extractor_builds_saveable_deputy():
    """``DeputyExtractor.__init__`` constructs a ``Deputy`` from the scraped name;
    pre-fix it raises because no ``_id`` is supplied."""
    response = SimpleNamespace(
        text='<html><body><span class="nombre-dip">Juan Pérez</span></body></html>',
        url="https://www.congreso.es/dummy",
    )

    extractor = DeputyExtractor(response, [])

    assert extractor.deputy.name == "Juan Pérez"
    assert extractor.deputy.id == ""  # placeholder; real id set in extract()


# --- Spain VideoExtractor: builds the Video in extract_interventions ----------

def test_video_extractor_builds_saveable_video(monkeypatch):
    """``extract_interventions`` builds a ``Video`` and sets its id from the link;
    pre-fix the empty ``Video()`` construction raises."""
    saved = []
    monkeypatch.setattr(video_extractor.Videos, "save", lambda video: saved.append(video))

    link = "https://video.congreso.es/intervencion/0001.mp4"
    interventions = {
        "0": {
            "video_intervencion": {"enlace_descarga02": link},
            "fecha": 1700000000,
            "sesion": {"nombre_sesion": "Sesión plenaria"},
            "tipo_intervencion": "Pregunta",
            "orador": "Diputada X",
        }
    }

    VideoExtractor("184/000001").extract_interventions(interventions)

    assert len(saved) == 1
    video = saved[0]
    assert video.id == generate_id(link)
    assert video.link == link
    assert video.reference == "184/000001"
