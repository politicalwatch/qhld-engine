"""`qhld search` — natural-language semantic search over indexed speeches.

``SearchSpeeches`` is imported lazily so ``--help`` stays side-effect free.
"""

import typer

app = typer.Typer(help="Semantic search over parliamentary speeches.")


@app.command("speeches")
def speeches(
    query: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(
        10, "--k", help="Number of results (passages, or speeches with --grouped)."),
    grouped: bool = typer.Option(
        False, "--grouped",
        help="Return distinct speeches, each with its top matching passages as "
             "highlights, instead of individual passages."),
    highlights: int = typer.Option(
        3, "--highlights", help="Max highlight passages per speech (with --grouped)."),
    group: str | None = typer.Option(None, "--group", help="Filter by parliamentary group."),
    legislature: str | None = typer.Option(None, "--legislature", help="Filter by legislature."),
    lang: str | None = typer.Option(None, "--lang", help="Filter by language (es/ca/eu/gl)."),
    speaker: str | None = typer.Option(None, "--speaker", help="Filter by speaker."),
    reranker: str | None = typer.Option(
        None, "--reranker",
        help="Cross-encoder model to rerank results (e.g. BAAI/bge-reranker-v2-m3); "
             "omit for bi-encoder order only."),
):
    """Search speeches semantically and print the ranked hits."""
    from qhld_engine.application.search.search_speeches import SearchSpeeches
    from qhld_engine.infrastructure.config.settings import get_settings

    filters = {
        "group": group,
        "legislature": legislature,
        "lang": lang,
        "speaker": speaker,
    }
    settings = get_settings()
    if reranker:
        settings = settings.model_copy(
            update={"reranker_provider": "cross_encoder", "reranker_model": reranker})
    service = SearchSpeeches(settings=settings)
    if grouped:
        _print_grouped(
            service.search_grouped(query, page_size=k, highlights=highlights, filters=filters))
    else:
        _print_hits(service.search(query, k=k, filters=filters))


def _snippet(payload, length=240):
    text = (payload.get("text") or "").strip().replace("\n", " ")
    return text[:length] + "…" if len(text) > length else text


def _print_hits(hits):
    if not hits:
        typer.echo("No results.")
        return
    for hit in hits:
        payload = hit.payload
        speaker = payload.get("speaker") or "?"
        group_line = payload.get("group") or payload.get("role") or "-"
        typer.echo(
            f"[{hit.score:.3f}] {speaker} ({group_line}) "
            f"· {payload.get('reference')} · {payload.get('lang')}\n    {_snippet(payload)}\n")


def _print_grouped(groups):
    if not groups:
        typer.echo("No results.")
        return
    for group in groups:
        head = group.highlights[0].payload if group.highlights else {}
        speaker = head.get("speaker") or "?"
        group_line = head.get("group") or head.get("role") or "-"
        typer.echo(
            f"[{group.score:.3f}] {speaker} ({group_line}) · {head.get('reference')} "
            f"· {len(group.highlights)} passage(s)")
        for hit in group.highlights:
            typer.echo(f"    · [{hit.score:.3f}] {hit.payload.get('lang')}  {_snippet(hit.payload, 200)}")
        typer.echo("")
