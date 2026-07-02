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
        help="Index only this initiative's speeches (always re-indexed); "
             "omit to index the whole corpus."),
    index_all: bool = typer.Option(
        False, "--all",
        help="Re-index every speech, not just those missing from the collection. "
             "Needed after chunking or embedding-model config changes; slow "
             "(embeds the whole corpus). By default only new speeches are indexed."),
):
    """Index speech passages into the vector store.

    By default this is incremental: only speeches not already present in the
    target (per-model) collection are embedded. Use ``--all`` to force a full
    re-index, or ``--reference`` to (re)index a single initiative's speeches.
    """
    _service().execute([reference] if reference else None, incremental=not index_all)
