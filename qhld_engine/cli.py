"""QHLD engine CLI — the `qhld` command (entrypoint: ``qhld_engine.cli:app``).

Composition root: wires the command groups defined under
``qhld_engine.presentation.cli`` and hosts the single-shot leaf commands
(stats, footprint, send-alerts, topic-alignment) inline. Task classes are
imported lazily inside each command so ``qhld --help`` stays fast and side-effect
free.
"""

import typer

from qhld_engine.presentation.cli import (
    debug, embeddings, evaluation, extractor, search, speeches, tagger, untagger,
)

app = typer.Typer(
    name="qhld",
    help="QHLD engine CLI — extract, tag, stats, footprint.",
    no_args_is_help=True,
)

app.add_typer(extractor.app, name="extractor")
app.add_typer(tagger.app, name="tagger")
app.add_typer(untagger.app, name="untagger")
app.add_typer(embeddings.app, name="embeddings")
app.add_typer(speeches.app, name="speeches")
app.add_typer(search.app, name="search")
app.add_typer(evaluation.app, name="eval")
app.add_typer(debug.app, name="debug")


@app.command("stats")
def stats():
    """Generate aggregated stats by topic, deputy and parliamentary group."""
    from qhld_engine.stats.process_stats import GenerateStats

    GenerateStats().generate()


@app.command("footprint")
def footprint():
    """Compute the legislative footprint for every topic and entity."""
    from qhld_engine.footprint.compute_footprint import ComputeFootprint

    ComputeFootprint().compute()


@app.command("send-alerts")
def send_alerts():
    """Trigger the (async) send-alerts task."""
    from qhld_engine.alerts.send_alerts import SendAlerts

    SendAlerts()


@app.command("topic-alignment")
def topic_alignment(
    id: str | None = typer.Argument(None, help="Initiative id; omit to recompute for all."),
):
    """Recompute topic alignment for one initiative, or all when no id is given."""
    from qhld_engine.tagger.topic_alignment import calculate_topic_alignment

    calculate_topic_alignment(id)


if __name__ == "__main__":
    app()
