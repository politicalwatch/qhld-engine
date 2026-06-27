"""`qhld tagger` — tag initiatives against the knowledge bases."""

import typer

app = typer.Typer(help="Tag initiatives against the knowledge bases.")


def _task():
    from qhld_engine.tagger.tag_initiatives import TagInitiatives

    return TagInitiatives()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Tag all untagged initiatives and every knowledge base when called with no subcommand."""
    if ctx.invoked_subcommand is None:
        _task().run()


@app.command("all")
def all_():
    _task().run()


@app.command("all-long")
def all_long():
    _task().tag_long()


@app.command("amendments")
def amendments():
    _task().tag_amendments()


@app.command("kb")
def kb(kb: str):
    _task().tag_kb(kb)


@app.command("new-topic")
def new_topic(topic: str):
    _task().new_topic(topic)


@app.command("new-tag")
def new_tag(topic: str, tag: str):
    _task().new_tag(topic, tag)


@app.command("modify-regex")
def modify_regex(topic: str, tag: str):
    """Re-tag a single tag's regex: untag its current matches, then tag again."""
    from qhld_engine.untagger.untag_initiatives import UntagInitiatives

    UntagInitiatives().by_tag(topic, tag)
    _task().new_tag(topic, tag)


@app.command("rename-tag")
def rename_tag(topic: str, old_tag: str, new_tag: str):
    _task().rename(topic, old_tag, new_tag)


@app.command("reference")
def reference(reference: str):
    _task().by_reference(reference)
