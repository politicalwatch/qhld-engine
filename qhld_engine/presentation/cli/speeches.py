"""`qhld speeches` — enrichment over already-extracted speeches.

Currently: ``tag-mentions``, the one-shot backfill that populates ``Speech.mentions``
for the existing corpus (new speeches are tagged at extract time). ``BackfillMentions``
is imported lazily so ``--help`` never connects to Mongo or loads the spaCy model.
"""

import typer

app = typer.Typer(help="Enrich already-extracted speeches (mentions, …).")


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
