"""Unit tests for the Typer CLI wiring — no DB.

Each command is a thin shim that instantiates a task class (lazily, inside the
command body) and calls one method. We patch the task class at its source module
and assert the command dispatches to the right method with the right arguments.
The task classes are never really constructed, so no Mongo is touched.
"""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from qhld_engine.cli import app
from qhld_engine.domain.ports.vector_store import SearchHit, SpeechGroup

pytestmark = pytest.mark.unit

runner = CliRunner()


def _patch_class(monkeypatch, dotted_path):
    """Patch a class at its source module with a MagicMock; return the mock instance."""
    mock_cls = MagicMock()
    monkeypatch.setattr(dotted_path, mock_cls)
    return mock_cls.return_value


# --- extractor -------------------------------------------------------------

# (subcommand args, method called on ExtractorTask, expected call args)
EXTRACTOR_CASES = [
    (["members"], "members", ()),
    (["load-groups", "groups.json"], "load_groups", ("groups.json",)),
    (["calculate-composition-groups"], "calculate_composition_groups", ()),
    (["initiatives"], "initiatives", ()),
    (["totals"], "totals", ()),
    (["references"], "references", ()),
    (["votes"], "votes", ()),
    (["interventions"], "interventions", ()),
    (["all-initiatives"], "all_initiatives", ()),
    (["all-references"], "all_references", ()),
    (["all-votes"], "all_votes", ()),
    (["all-interventions"], "all_interventions", ()),
    (["single-initiative", "REF"], "single_initiatives", ("REF",)),
    (["single-intervention", "REF"], "single_interventions", ("REF",)),
    (["single-vote", "REF"], "single_votes", ("REF",)),
    (["type-initiative", "C"], "type_initiatives", ("C",)),
    (["type-references", "C"], "type_references", ("C",)),
    (["type-interventions", "C"], "type_interventions", ("C",)),
    (["type-votes", "C"], "type_votes", ("C",)),
    (["type-all-initiative", "C"], "type_all_initiatives", ("C",)),
    (["type-all-references", "C"], "type_all_references", ("C",)),
    (["type-all-interventions", "C"], "type_all_interventions", ("C",)),
    (["type-all-votes", "C"], "type_all_votes", ("C",)),
]


@pytest.mark.parametrize("argv, method, call_args", EXTRACTOR_CASES)
def test_extractor_commands(monkeypatch, argv, method, call_args):
    task = _patch_class(monkeypatch, "qhld_engine.extractors.extractor.ExtractorTask")
    result = runner.invoke(app, ["extractor", *argv])
    assert result.exit_code == 0, result.output
    getattr(task, method).assert_called_once_with(*call_args)


def test_extractor_default_runs(monkeypatch):
    task = _patch_class(monkeypatch, "qhld_engine.extractors.extractor.ExtractorTask")
    result = runner.invoke(app, ["extractor"])
    assert result.exit_code == 0, result.output
    task.run.assert_called_once_with()


# --- tagger ----------------------------------------------------------------

TAGGER_CASES = [
    (["all"], "run", ()),
    (["all-long"], "tag_long", ()),
    (["amendments"], "tag_amendments", ()),
    (["kb", "KB"], "tag_kb", ("KB",)),
    (["new-topic", "T"], "new_topic", ("T",)),
    (["new-tag", "T", "G"], "new_tag", ("T", "G")),
    (["rename-tag", "T", "O", "N"], "rename", ("T", "O", "N")),
    (["reference", "REF"], "by_reference", ("REF",)),
]


@pytest.mark.parametrize("argv, method, call_args", TAGGER_CASES)
def test_tagger_commands(monkeypatch, argv, method, call_args):
    task = _patch_class(monkeypatch, "qhld_engine.tagger.tag_initiatives.TagInitiatives")
    result = runner.invoke(app, ["tagger", *argv])
    assert result.exit_code == 0, result.output
    getattr(task, method).assert_called_once_with(*call_args)


def test_tagger_default_runs(monkeypatch):
    task = _patch_class(monkeypatch, "qhld_engine.tagger.tag_initiatives.TagInitiatives")
    result = runner.invoke(app, ["tagger"])
    assert result.exit_code == 0, result.output
    task.run.assert_called_once_with()


def test_tagger_modify_regex_untags_then_tags(monkeypatch):
    tagger = _patch_class(monkeypatch, "qhld_engine.tagger.tag_initiatives.TagInitiatives")
    untagger = _patch_class(monkeypatch, "qhld_engine.untagger.untag_initiatives.UntagInitiatives")
    result = runner.invoke(app, ["tagger", "modify-regex", "T", "G"])
    assert result.exit_code == 0, result.output
    untagger.by_tag.assert_called_once_with("T", "G")
    tagger.new_tag.assert_called_once_with("T", "G")


# --- untagger --------------------------------------------------------------

UNTAGGER_CASES = [
    (["all"], "untag_all", ()),
    (["kb", "KB"], "by_kb", ("KB",)),
    (["topic", "T"], "by_topic", ("T",)),
    (["tag", "T", "G"], "by_tag", ("T", "G")),
    (["reference", "REF"], "by_reference", ("REF",)),
]


@pytest.mark.parametrize("argv, method, call_args", UNTAGGER_CASES)
def test_untagger_commands(monkeypatch, argv, method, call_args):
    task = _patch_class(monkeypatch, "qhld_engine.untagger.untag_initiatives.UntagInitiatives")
    result = runner.invoke(app, ["untagger", *argv])
    assert result.exit_code == 0, result.output
    getattr(task, method).assert_called_once_with(*call_args)


def test_untagger_default_untags_all(monkeypatch):
    task = _patch_class(monkeypatch, "qhld_engine.untagger.untag_initiatives.UntagInitiatives")
    result = runner.invoke(app, ["untagger"])
    assert result.exit_code == 0, result.output
    task.untag_all.assert_called_once_with()


# --- leaf commands ---------------------------------------------------------

def test_stats(monkeypatch):
    task = _patch_class(monkeypatch, "qhld_engine.stats.process_stats.GenerateStats")
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0, result.output
    task.generate.assert_called_once_with()


def test_footprint(monkeypatch):
    task = _patch_class(monkeypatch, "qhld_engine.footprint.compute_footprint.ComputeFootprint")
    result = runner.invoke(app, ["footprint"])
    assert result.exit_code == 0, result.output
    task.compute.assert_called_once_with()


def test_send_alerts(monkeypatch):
    mock_cls = MagicMock()
    monkeypatch.setattr("qhld_engine.alerts.send_alerts.SendAlerts", mock_cls)
    result = runner.invoke(app, ["send-alerts"])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once_with()


def test_topic_alignment_with_id(monkeypatch):
    fn = MagicMock()
    monkeypatch.setattr("qhld_engine.tagger.topic_alignment.calculate_topic_alignment", fn)
    result = runner.invoke(app, ["topic-alignment", "abc123"])
    assert result.exit_code == 0, result.output
    fn.assert_called_once_with("abc123")


def test_topic_alignment_without_id(monkeypatch):
    fn = MagicMock()
    monkeypatch.setattr("qhld_engine.tagger.topic_alignment.calculate_topic_alignment", fn)
    result = runner.invoke(app, ["topic-alignment"])
    assert result.exit_code == 0, result.output
    fn.assert_called_once_with(None)


# --- search ----------------------------------------------------------------

def test_search_default_is_passage_level(monkeypatch):
    svc = _patch_class(
        monkeypatch, "qhld_engine.application.search.search_speeches.SearchSpeeches")
    svc.search.return_value = []
    result = runner.invoke(app, ["search", "speeches", "financiación autonómica", "--k", "7"])
    assert result.exit_code == 0, result.output
    svc.search.assert_called_once_with(
        "financiación autonómica", k=7,
        filters={"group": None, "legislature": None, "lang": None, "speaker": None})
    svc.search_grouped.assert_not_called()  # --grouped absent → passage path (baseline A/B)


def test_search_grouped_flag_calls_grouped(monkeypatch):
    svc = _patch_class(
        monkeypatch, "qhld_engine.application.search.search_speeches.SearchSpeeches")
    svc.search_grouped.return_value = [
        SpeechGroup(speech_id="A", score=0.8, highlights=[
            SearchHit(id="p1", score=0.8, payload={
                "speaker": "X", "reference": "172/000001", "lang": "gl", "text": "hola"})])]
    result = runner.invoke(app, [
        "search", "speeches", "q", "--grouped", "--k", "5", "--highlights", "2", "--lang", "gl"])
    assert result.exit_code == 0, result.output
    svc.search_grouped.assert_called_once_with(
        "q", page_size=5, highlights=2,
        filters={"group": None, "legislature": None, "lang": "gl", "speaker": None})
    svc.search.assert_not_called()
    assert "172/000001" in result.output


# --- debug -----------------------------------------------------------------

def test_debug_generate_alert(monkeypatch):
    initiatives = MagicMock()
    initiatives.by_reference.return_value = ["INIT"]
    alerts = MagicMock()
    monkeypatch.setattr("tipi_data.repositories.initiatives.Initiatives", initiatives)
    monkeypatch.setattr("tipi_data.repositories.alerts.InitiativeAlerts", alerts)
    result = runner.invoke(app, ["debug", "generate-alert", "REF"])
    assert result.exit_code == 0, result.output
    initiatives.by_reference.assert_called_once_with("REF")
    alerts.create_alert.assert_called_once_with("INIT", "Nueva iniciativa")
