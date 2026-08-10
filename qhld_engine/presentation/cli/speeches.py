"""`qhld speeches` — enrichment over already-extracted speeches.

``tag-mentions``, ``tag-entities`` and ``probe-durations`` are the one-shot backfills
that populate ``Speech.mentions``, ``Speech.entities`` and ``Speech.duration`` for the
existing corpus (new speeches get all three at extract time). The backfill services
are imported lazily so ``--help`` never connects to Mongo or loads the spaCy model.
"""

import typer

app = typer.Typer(help="Enrich already-extracted speeches (mentions, entities, …).")


@app.command("tag-mentions")
def tag_mentions(
    reference: str | None = typer.Option(
        None, "--reference", "-r",
        help="Tag only this initiative's speeches (always re-tagged); "
             "omit to tag the whole corpus."),
    tag_all: bool = typer.Option(
        False, "--all",
        help="Re-tag every speech, not just those without mentions. Needed after a "
             "threshold or NER-model change. By default only untagged speeches run."),
):
    """Extract mentioned deputies from stored speech text and persist them to
    ``Speech.mentions``. Incremental by default; ``--reference`` targets one initiative."""
    from qhld_engine.application.speeches.backfill_mentions import BackfillMentions

    BackfillMentions().execute(
        [reference] if reference else None, incremental=not tag_all)


@app.command("probe-durations")
def probe_durations(
    reference: str | None = typer.Option(
        None, "--reference", "-r",
        help="Probe only this initiative's speeches (always re-probed); "
             "omit to probe the whole corpus."),
    probe_all: bool = typer.Option(
        False, "--all",
        help="Re-probe every speech, not just those without a duration. Only "
             "needed if the Congress republishes a clip. By default a speech that "
             "already carries one is skipped."),
):
    """Read how long each intervention's video runs and persist it to
    ``Speech.duration``. Headers only — nothing is downloaded. Incremental by
    default; ``--reference`` targets one initiative."""
    from qhld_engine.application.speeches.backfill_durations import BackfillDurations

    BackfillDurations().execute(
        [reference] if reference else None, incremental=not probe_all)


@app.command("tag-entities")
def tag_entities(
    reference: str | None = typer.Option(
        None, "--reference", "-r",
        help="Tag only this initiative's speeches (always re-tagged); "
             "omit to tag the whole corpus."),
    tag_all: bool = typer.Option(
        False, "--all",
        help="Re-tag every speech, not just those without entities. Needed after a "
             "normalization or stoplist change. By default only untagged speeches run."),
):
    """Extract the non-person named entities from stored speech text and persist
    them to ``Speech.entities``. Incremental by default; ``--reference`` targets
    one initiative. Mentions/interruptions are left untouched."""
    from qhld_engine.application.speeches.backfill_entities import BackfillEntities

    BackfillEntities().execute(
        [reference] if reference else None, incremental=not tag_all)
