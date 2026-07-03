"""`qhld eval` — semantic-search A/B benchmark CLI.

Driving adapter over the ``RunBenchmark`` application service. Runs the frozen
query set across an embedding-model x reranker grid and prints rank / MRR /
hit@k / recall@k / MAP per dimension. Run in-container (repo volume-mounted),
where ``Settings`` already reaches Qdrant + ollama:

    docker exec qhld-engine qhld eval ab \\
        --models granite-embedding:278m,bge-m3:567m --rerankers none
"""

import typer

app = typer.Typer(
    name="eval",
    help="A/B benchmark semantic search across embedding models and rerankers.",
    no_args_is_help=True,
)


def _split(value):
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command("ab")
def ab(
    models: str = typer.Option(..., "--models", help="Comma-separated ollama embedding tags."),
    rerankers: str = typer.Option("none", "--rerankers", help="Comma-separated rerankers ('none' = bi-encoder only)."),
    k: int = typer.Option(10, "--k", help="Retrieval depth per query."),
    hit_at: int = typer.Option(5, "--hit-at", help="hit@k / recall@k metric threshold."),
    queryset: str = typer.Option(None, "--queryset", help="Path to a query-set JSON (defaults to the frozen set)."),
    verbose: bool = typer.Option(False, "--verbose", help="Dump top-k per query."),
):
    """Benchmark each (model x reranker) cell over the frozen query set."""
    from qhld_engine.application.evaluation.benchmark import RunBenchmark

    runner = RunBenchmark(queryset) if queryset else RunBenchmark()
    model_list, reranker_list = _split(models), _split(rerankers)
    typer.echo(
        f"Query set: {len(runner.queryset)} queries · retrieval k={k} · "
        f"metric @{hit_at} · models={model_list} · rerankers={reranker_list}"
    )
    for model in model_list:
        for reranker in reranker_list:
            rows = runner.run(model, reranker=reranker, k=k)
            _print_report(model, reranker, rows, hit_at, verbose)


def _print_report(model, reranker, rows, hit_at, verbose):
    from qhld_engine.domain.evaluation import scoring

    label = model if reranker in ("none", "noop") else f"{model} + {reranker}"
    typer.echo(f"\n=== {label} ===")
    typer.echo(
        f"{'id':<4}{'dimension':<13}{'rank':>5}{'hit@'+str(hit_at):>7}"
        f"{'score':>8}  penalty"
    )
    for row in rows:
        if verbose:
            _dump(row)
        rank = row["rank"] if row["rank"] is not None else "-"
        hitk = "Y" if scoring.hit_at_k(row["rank"], hit_at) else "n"
        score = row["score"] if row["score"] is not None else "-"
        penalty = ""
        if row.get("lang"):
            penalty = f"nolang_rank={row.get('nolang_rank')} nolang_score={row.get('nolang_score')}"
        typer.echo(
            f"{row['id']:<4}{row['dimension']:<13}{str(rank):>5}{hitk:>7}"
            f"{str(score):>8}  {penalty}"
        )

    typer.echo(f"  aggregates (MRR / hit@{hit_at} / recall@{hit_at} / MAP):")
    for dimension, metrics in scoring.aggregate(rows, k=hit_at).items():
        typer.echo(
            f"    {dimension:<13} n={metrics['n']:<3} MRR={metrics['mrr']:<8} "
            f"hit@{hit_at}={metrics[f'hit_at_{hit_at}']:<8} "
            f"recall@{hit_at}={metrics[f'recall_at_{hit_at}']:<8} MAP={metrics['map']}"
        )


def _dump(row):
    typer.echo(f"  · {row['id']} {row['query']!r}")
    for position, hit in enumerate(row.get("hits", []), start=1):
        payload = hit.payload
        typer.echo(
            f"      {position:>2}. [{hit.score:.3f}] {payload.get('reference')} "
            f"· {payload.get('lang')} · {payload.get('speaker')}"
        )
