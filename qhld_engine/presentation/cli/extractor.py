"""`qhld extractor` — extract members, groups, initiatives, votes and interventions.

Thin presentation layer: every command instantiates the existing ``ExtractorTask``
and calls the matching method. ``ExtractorTask`` is imported lazily so ``--help``
never connects to Mongo or loads a country extractor module.
"""

import typer

app = typer.Typer(help="Extract data from the parliament source.")


def _task():
    from qhld_engine.extractors.extractor import ExtractorTask

    return ExtractorTask()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Run the full extraction (members + groups + initiatives) when called with no subcommand."""
    if ctx.invoked_subcommand is None:
        _task().run()


@app.command("mark-complete")
def mark_complete():
    """Record that a whole extraction run finished.

    Meant as the last step of the daily pipeline: whatever stops the pipeline early
    leaves this timestamp where it was, which is what makes it mean "the data is
    current" rather than "extraction started".
    """
    from qhld_engine.application.freshness import EXTRACTION, mark_refreshed

    mark_refreshed(EXTRACTION)


@app.command("members")
def members():
    _task().members()


@app.command("load-groups")
def load_groups(groups_file: str):
    _task().load_groups(groups_file)


@app.command("calculate-composition-groups")
def calculate_composition_groups():
    _task().calculate_composition_groups()


@app.command("initiatives")
def initiatives():
    _task().initiatives()


@app.command("totals")
def totals():
    _task().totals()


@app.command("references")
def references():
    _task().references()


@app.command("votes")
def votes():
    _task().votes()


@app.command("interventions")
def interventions():
    _task().interventions()


@app.command("speeches")
def speeches(
    since: str | None = typer.Option(
        None, "--since", help="First day to enumerate, YYYY-MM-DD. Defaults to "
                              "SPEECH_EXTRACTION_SINCE."),
    until: str | None = typer.Option(
        None, "--until", help="Last day to enumerate, YYYY-MM-DD. Defaults to today."),
):
    """Daily speech extraction: enumerate a date range and extract what is missing.

    The whole range is walked on every run, deliberately. The Diario that carries a
    speech's text is published days or weeks after the sitting, so a run that
    advanced a watermark would strand every transcript that appeared behind it;
    re-reading the range costs little enough to make the sweep self-repairing
    instead. Narrow it with --since/--until only for backfills and debugging.
    """
    _task().speeches(since, until)


@app.command("all-initiatives")
def all_initiatives():
    _task().all_initiatives()


@app.command("all-references")
def all_references():
    _task().all_references()


@app.command("all-votes")
def all_votes():
    _task().all_votes()


@app.command("all-interventions")
def all_interventions():
    _task().all_interventions()


@app.command("all-speeches")
def all_speeches():
    _task().all_speeches()


@app.command("single-initiative")
def single_initiative(reference: str):
    _task().single_initiatives(reference)


@app.command("single-intervention")
def single_intervention(reference: str):
    _task().single_interventions(reference)


@app.command("single-speech")
def single_speech(
    reference: str | None = typer.Argument(
        None, help="Initiative reference to extract. Omit it only when every "
                   "--video-id is already stored, so its reference can be looked up."),
    video_id: list[str] = typer.Option(
        [], "--video-id",
        help="Congress intervention id to save (repeatable). The reference is still "
             "downloaded and segmented in full — only the saving is narrowed — so "
             "targeting several at once is cheaper than one run each."),
):
    """Extract one initiative's speeches, or just some of them.

    Targeting leaves the untargeted speeches of that reference holding whatever the
    last full run gave them, which drifts out of step with the classifier as it
    changes. Meant for debugging and gold-set work; re-extract per reference for
    anything corpus-wide.
    """
    if reference is None and not video_id:
        raise typer.BadParameter("give a reference, a --video-id, or both")
    try:
        saved = _task().single_speeches(reference, video_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if video_id and not saved:
        typer.echo("No speech matched the given --video-id; nothing was saved.")
        raise typer.Exit(1)


@app.command("single-vote")
def single_vote(reference: str):
    _task().single_votes(reference)


@app.command("type-initiative")
def type_initiative(type_code: str):
    _task().type_initiatives(type_code)


@app.command("type-references")
def type_references(type_code: str):
    _task().type_references(type_code)


@app.command("type-interventions")
def type_interventions(type_code: str):
    _task().type_interventions(type_code)


@app.command("type-speeches")
def type_speeches(type_code: str):
    _task().type_speeches(type_code)


@app.command("type-votes")
def type_votes(type_code: str):
    _task().type_votes(type_code)


@app.command("type-all-initiative")
def type_all_initiative(type_code: str):
    _task().type_all_initiatives(type_code)


@app.command("type-all-references")
def type_all_references(type_code: str):
    _task().type_all_references(type_code)


@app.command("type-all-interventions")
def type_all_interventions(type_code: str):
    _task().type_all_interventions(type_code)


@app.command("type-all-speeches")
def type_all_speeches(type_code: str):
    _task().type_all_speeches(type_code)


@app.command("type-all-votes")
def type_all_votes(type_code: str):
    _task().type_all_votes(type_code)
