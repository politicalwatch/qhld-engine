"""Unit regression for the footprint write-path ``_id`` bug — no DB, no threads.

Same root cause as the extractor ``_id`` write-paths: ``FootprintByTopic`` is a
``MongoModel`` whose ``id`` (``alias="_id"``) is required with no default, so the old
``FootprintByTopic()`` empty-construct-then-assign pattern raised
``ValidationError: _id Field required`` under the pydantic data layer. ``compute()``
must build it with ``id`` set at construction.

The earlier write-path sweep missed ``compute_footprint.py``; this test drives the
real ``compute()`` loop body (aggregations stubbed) so a revert to the empty construct
is caught, not just a model-level smoke test.
"""

from datetime import datetime

import pytest

from tipi_data.models.footprint import FootprintByTopic

from qhld_engine.footprint import compute_footprint as cf_module
from qhld_engine.footprint.compute_footprint import ComputeFootprint

pytestmark = pytest.mark.unit


def test_compute_builds_saveable_topic_footprint(monkeypatch):
    # Bypass the DB-touching __init__ and feed one topic through compute().
    cf = ComputeFootprint.__new__(ComputeFootprint)
    cf.today = datetime.today()
    cf.topics = [{"id": "topic-1", "name": "AUTISMO"}]
    cf.deputies = []
    cf.parliamentarygroups = []

    # The aggregation passes hit the DB; not under test here.
    monkeypatch.setattr(cf_module.Initiatives, "aggregate", lambda pipeline: [])

    saved = []
    monkeypatch.setattr(cf_module.Footprints, "save_topic", lambda fp: saved.append(fp))
    monkeypatch.setattr(cf_module.Footprints, "save_deputy", lambda fp: None)
    monkeypatch.setattr(
        cf_module.Footprints, "save_parliamentarygroup", lambda fp: None)

    cf.compute()  # pre-fix: raises _id-required at FootprintByTopic()

    assert len(saved) == 1
    fp = saved[0]
    assert isinstance(fp, FootprintByTopic)
    assert fp.id == "topic-1"          # set at construction, not lost
    assert fp.name == "AUTISMO"
    assert fp.to_bson()["_id"] == "topic-1"
