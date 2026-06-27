"""`qhld untagger` — remove tags from initiatives."""

import typer

app = typer.Typer(help="Remove tags from initiatives.")


def _task():
    from qhld_engine.untagger.untag_initiatives import UntagInitiatives

    return UntagInitiatives()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Untag every initiative when called with no subcommand."""
    if ctx.invoked_subcommand is None:
        _task().untag_all()


@app.command("all")
def all_():
    _task().untag_all()


@app.command("kb")
def kb(kb: str):
    _task().by_kb(kb)


@app.command("topic")
def topic(topic: str):
    _task().by_topic(topic)


@app.command("tag")
def tag(topic: str, tag: str):
    _task().by_tag(topic, tag)


@app.command("reference")
def reference(reference: str):
    _task().by_reference(reference)
