"""`qhld embeddings` — build the speech vector index for semantic search.

Reads already-extracted speeches from Mongo and indexes their passages in Qdrant.
``IndexSpeeches`` is imported lazily so ``--help`` never connects to Mongo, Qdrant
or the embedder.
"""

import typer

app = typer.Typer(help="Generate and index speech embeddings for semantic search.")


def _service():
    from qhld_engine.application.speeches.index_speeches import IndexSpeeches

    return IndexSpeeches()


@app.command("index")
def index(
    reference: str | None = typer.Option(
        None, "--reference", "-r",
        help="Index only this initiative's speeches; omit to index all speeches."),
):
    """Index speech passages into the vector store (all speeches, or one reference)."""
    _service().execute([reference] if reference else None)
