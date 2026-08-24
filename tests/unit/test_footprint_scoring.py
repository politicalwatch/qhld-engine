"""Unit tests for the aggregation-based footprint scoring — no DB, no threads.

Covers the weight table (:func:`weight_of`, the Python reference for the Mongo
``$switch`` expression) and the in-memory assembly (penalty + contact bonus +
normalization) driven from simulated aggregation results.
"""

from datetime import datetime, timedelta

import pytest

from qhld_engine.footprint import compute_footprint as cf_module
from qhld_engine.footprint.compute_footprint import ComputeFootprint
from qhld_engine.footprint.footprint_managers import (
    inactivity_penalty,
    weight_of,
)

pytestmark = pytest.mark.unit


# --- Weight table ----------------------------------------------------------

def test_weight_base_tiers():
    assert weight_of("Pregunta oral en Pleno", "En tramitación") == 10
    assert weight_of("Proposición no de Ley ante el Pleno", "En tramitación") == 40
    assert weight_of("Proposición de ley de Diputados", "En tramitación") == 80
    assert weight_of("Interpelación urgente", "En tramitación") == 4


def test_weight_unknown_type_is_zero():
    # Written questions are intentionally not scored.
    assert weight_of("Pregunta al Gobierno con respuesta escrita", "En tramitación") == 0
    assert weight_of("Respuesta", "Contestada") == 0


def test_weight_approved_bonus_stacks_on_base():
    # Approved PNL: base 40 + bonus 20.
    assert weight_of("Proposición no de Ley ante el Pleno", "Aprobada") == 60
    # Approved law: base 80 + bonus 60.
    assert weight_of("Proposición de ley de Diputados", "Aprobada") == 140
    # Bonus only applies when approved.
    assert weight_of("Proposición no de Ley ante el Pleno", "Rechazada") == 40


# --- Inactivity penalty ----------------------------------------------------

def test_inactivity_penalty_tiers():
    today = datetime(2026, 8, 19)
    assert inactivity_penalty(None, today) == 0
    assert inactivity_penalty(today - timedelta(days=10), today) == 0
    assert inactivity_penalty(today - timedelta(days=30 * 4), today) == 0.10
    assert inactivity_penalty(today - timedelta(days=30 * 8), today) == 0.25
    assert inactivity_penalty(today - timedelta(days=30 * 13), today) == 0.50


# --- Assembly from simulated aggregation results ---------------------------

def _make_cf(deputies, groups, aggregate_map):
    """Build a ComputeFootprint without touching the DB. ``aggregate_map`` maps a
    marker embedded in each pipeline (via its last $group id) to a canned result."""
    cf = ComputeFootprint.__new__(ComputeFootprint)
    cf.today = datetime(2026, 8, 19)
    cf.topics = [{"id": "t1", "name": "SANIDAD"}]
    cf.deputies = deputies
    cf.parliamentarygroups = groups
    return cf


def test_compute_end_to_end_with_stubbed_aggregations(monkeypatch):
    deputies = [
        {"id": "d1", "name": "Ana", "email": "ana@x.es", "twitter": "@ana"},
        {"id": "d2", "name": "Beto", "email": "", "twitter": ""},
    ]
    groups = [{"id": "g1", "name": "GP1"}]

    # Route each pipeline to a canned result by inspecting its shape:
    #  - topic scores group by {e, t} and sum "_contrib"-derived "score"
    #  - topic last-dates group by {e, t} with only "last"
    #  - global group by author string with "score" + "last"
    recent = cf_module.datetime(2026, 8, 1)

    def fake_aggregate(pipeline):
        group = pipeline[-1]["$group"]
        gid = group["_id"]
        is_topic = isinstance(gid, dict)
        author_field = None
        for stage in pipeline:
            if "$unwind" in stage and stage["$unwind"].startswith("$author"):
                author_field = stage["$unwind"]
        is_deputy = author_field == "$author_deputies"
        has_score = "score" in group

        if is_topic and has_score:  # topic_scores_pipeline
            if is_deputy:
                return [{"_id": {"e": "Ana", "t": "SANIDAD"}, "score": 10.0}]
            return [{"_id": {"e": "GP1", "t": "SANIDAD"}, "score": 5.0}]
        if is_topic and not has_score:  # topic_last_dates_pipeline
            if is_deputy:
                return [{"_id": {"e": "Ana", "t": "SANIDAD"}, "last": recent}]
            return [{"_id": {"e": "GP1", "t": "SANIDAD"}, "last": recent}]
        # global_scores_pipeline
        if is_deputy:
            return [
                {"_id": "Ana", "score": 100.0, "last": recent},
                {"_id": "Beto", "score": 50.0, "last": recent},
            ]
        return [{"_id": "GP1", "score": 20.0, "last": recent}]

    monkeypatch.setattr(cf_module.Initiatives, "aggregate", fake_aggregate)

    saved_topics, saved_deputies, saved_groups = [], [], []
    monkeypatch.setattr(cf_module.Footprints, "save_topic", saved_topics.append)
    monkeypatch.setattr(cf_module.Footprints, "save_deputy", saved_deputies.append)
    monkeypatch.setattr(
        cf_module.Footprints, "save_parliamentarygroup", saved_groups.append)

    cf = _make_cf(deputies, groups, {})
    cf.compute()

    # One topic footprint, with every entity represented (even zero-scorers).
    assert len(saved_topics) == 1
    tfp = saved_topics[0]
    assert tfp.id == "t1"
    assert {e.name for e in tfp.deputies} == {"Ana", "Beto"}
    # Ana scored on the topic, Beto did not -> Ana ranks first and normalizes to 100.
    assert tfp.deputies[0].name == "Ana"
    assert tfp.deputies[0].score == 100.0
    assert tfp.deputies[-1].score == 0.0

    # Global deputy scores: Ana (raw 100 + email/social 80 = 180) > Beto (raw 50).
    by_id = {fp.id: fp for fp in saved_deputies}
    assert by_id["d1"].score == 100.0   # normalized max
    assert by_id["d2"].score == 0.0     # normalized min
    # Each deputy carries the topic list.
    assert any(t.name == "SANIDAD" for t in by_id["d1"].topics)

    assert len(saved_groups) == 1
    assert saved_groups[0].id == "g1"
