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


@app.command("single-initiative")
def single_initiative(reference: str):
    _task().single_initiatives(reference)


@app.command("single-intervention")
def single_intervention(reference: str):
    _task().single_interventions(reference)


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


@app.command("type-all-votes")
def type_all_votes(type_code: str):
    _task().type_all_votes(type_code)
