"""`qhld search` — natural-language semantic search over indexed speeches.

``SearchSpeeches`` is imported lazily so ``--help`` stays side-effect free.
"""

import typer

app = typer.Typer(help="Semantic search over parliamentary speeches.")


@app.command("speeches")
def speeches(
    query: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(10, "--k", help="Number of results."),
    group: str | None = typer.Option(None, "--group", help="Filter by parliamentary group."),
    legislature: str | None = typer.Option(None, "--legislature", help="Filter by legislature."),
    lang: str | None = typer.Option(None, "--lang", help="Filter by language (es/ca/eu/gl)."),
    speaker: str | None = typer.Option(None, "--speaker", help="Filter by speaker."),
):
    """Search speeches semantically and print the ranked hits."""
    from qhld_engine.application.search.search_speeches import SearchSpeeches

    filters = {
        "group": group,
        "legislature": legislature,
        "lang": lang,
        "speaker": speaker,
    }
    hits = SearchSpeeches().search(query, k=k, filters=filters)
    if not hits:
        typer.echo("No results.")
        return
    for hit in hits:
        payload = hit.payload
        speaker_line = payload.get("speaker") or "?"
        group_line = payload.get("group") or payload.get("role") or "-"
        snippet = (payload.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        typer.echo(
            f"[{hit.score:.3f}] {speaker_line} ({group_line}) "
            f"· {payload.get('reference')} · {payload.get('lang')}\n    {snippet}\n")
