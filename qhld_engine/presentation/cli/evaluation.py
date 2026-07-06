"""`qhld eval` — A/B benchmark CLI.

Two sibling benchmarks, one per task:
- ``retrieval`` — semantic search across an embedding-model x reranker grid
  (rank / MRR / hit@k / recall@k / MAP), over ``RunBenchmark``.
- ``parse`` — the NL query parser across LLMs / rule-based (per-slot P/R/F1,
  cost + latency via LangSmith), over ``RunParseBenchmark``.

Run in-container (repo volume-mounted), where ``Settings`` already reaches
Qdrant + ollama:

    docker exec qhld-engine qhld eval retrieval \\
        --models granite-embedding:278m,bge-m3:567m --rerankers none
"""

import typer

app = typer.Typer(
    name="eval",
    help="A/B benchmarks: retrieval (embedding x reranker) and parse (query parser).",
    no_args_is_help=True,
)


def _split(value):
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command("retrieval")
def retrieval(
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


@app.command("parse")
def parse(
    parsers: str = typer.Option(
        "llm,rule_based", "--parsers",
        help="Comma-separated query parsers to compare ('llm', 'rule_based'). "
             "Ignored when --models is given."),
    models: str = typer.Option(
        None, "--models",
        help="Comma-separated 'provider:model' LLM specs to A/B the 'llm' parser over "
             "(e.g. 'ollama:gpt-oss:20b,openai:gpt-5.4-nano-2026-03-17'). Split on the "
             "FIRST colon, so ollama tags keep theirs. LangSmith traces every run when "
             "LANGSMITH_TRACING is set, grouped by model."),
    repeats: int = typer.Option(
        1, "--repeats", min=1,
        help="Runs per model; the summary reports the median of each metric to smooth "
             "latency noise (results are otherwise ~stable at temperature 0)."),
    baseline: bool = typer.Option(
        True, "--baseline/--no-baseline",
        help="In --models mode, also run the rule_based parser as a $0/fast reference."),
    queryset: str = typer.Option(None, "--queryset", help="Path to a parse query-set JSON."),
    verbose: bool = typer.Option(False, "--verbose", help="Dump predicted vs gold per query."),
):
    """Compare query parsers on the frozen parse set: per-slot P/R/F1, exact-match,
    topic overlap and mean latency (LLM structured-output vs spaCy+dateparser).

    With --models, sweeps the 'llm' parser across several LLMs and prints a median
    comparison summary; token counts + $ cost land in LangSmith (project 'qhld')."""
    from qhld_engine.application.evaluation.parse_benchmark import RunParseBenchmark

    runner = RunParseBenchmark(queryset) if queryset else RunParseBenchmark()

    if not models:
        typer.echo(
            f"Parse query set: {len(runner.queries)} queries · today={runner.today.isoformat()} "
            f"· parsers={_split(parsers)}")
        for name in _split(parsers):
            rows = runner.run(name)
            _print_parse_report(name, rows, verbose)
        return

    specs = _parse_models(models)
    typer.echo(
        f"Parse A/B · {len(runner.queries)} queries · today={runner.today.isoformat()} "
        f"· repeats={repeats} · models={[label for _, _, label in specs]}"
        + (" · +rule_based" if baseline else ""))
    summary = []
    for provider, model, label in specs:
        try:
            first_rows, median = _run_scored(runner, "llm", provider, model, repeats)
        except Exception as exc:  # noqa: BLE001 - one bad model must not sink the sweep
            typer.echo(f"\n=== {label}: FAILED ({type(exc).__name__}: {exc}) — skipped ===")
            continue
        _print_parse_report(label, first_rows, verbose)
        summary.append((label, median))
    if baseline:
        try:
            first_rows, median = _run_scored(runner, "rule_based", None, None, repeats)
            _print_parse_report("rule_based", first_rows, verbose)
            summary.append(("rule_based", median))
        except ImportError:
            # dateparser not installed — don't discard the (already-run) LLM results.
            typer.echo(
                "\n[skipped rule_based baseline: dateparser not installed "
                "— run `uv sync`, or pass --no-baseline]")
    _print_parse_summary(summary, repeats)


@app.command("mentions")
def mentions(
    goldset: str = typer.Option(
        None, "--goldset", help="Path to a mentions gold-set JSON (defaults to the frozen set)."),
    verbose: bool = typer.Option(False, "--verbose", help="Show predicted vs gold per speech."),
):
    """Score index-time mention extraction (NER → resolved deputies) against the
    frozen gold set: micro P/R/F1 over mentions, per-speech exact-match, latency."""
    from qhld_engine.application.evaluation.mentions_benchmark import RunMentionsBenchmark
    from qhld_engine.domain.evaluation import mentions_scoring

    runner = RunMentionsBenchmark(goldset) if goldset else RunMentionsBenchmark()
    typer.echo(f"Mentions gold set: {len(runner.entries)} speeches")
    rows = runner.run()
    report = mentions_scoring.score(rows)
    if verbose:
        for row in rows:
            ok = "✓" if set(row["pred_deputies"]) == set(row["gold_deputies"]) else "✗"
            typer.echo(f"  {ok} {row.get('id', row['speech_id'])} {row.get('reference', '')}")
            missed = sorted(set(row["gold_deputies"]) - set(row["pred_deputies"]))
            spurious = sorted(set(row["pred_deputies"]) - set(row["gold_deputies"]))
            if missed:
                typer.echo(f"      missed (FN): {missed}")
            if spurious:
                typer.echo(f"      spurious (FP): {spurious}")
    m = report["micro"]
    typer.echo("\n=== mentions ===")
    typer.echo(f"{'':<7}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    typer.echo(
        f"{'MICRO':<7}{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}   "
        f"{m['tp']}/{m['fp']}/{m['fn']}")
    typer.echo(
        f"  exact-match={report['exact_match']}  mean-latency={report['mean_latency']}s")


def _parse_models(value):
    """Parse 'provider:model' specs into (provider, model, label) triples, splitting on
    the FIRST colon so ollama tags like 'gpt-oss:20b' keep their colon."""
    specs = []
    for item in _split(value):
        if ":" not in item:
            raise typer.BadParameter(
                f"model spec {item!r} must be 'provider:model' (e.g. 'ollama:gpt-oss:20b')")
        provider, model = item.split(":", 1)
        specs.append((provider.strip(), model.strip(), item))
    return specs


def _run_scored(runner, parser_name, provider, model, repeats):
    """Run a parser ``repeats`` times; return (first-pass rows, median-metrics dict)."""
    from qhld_engine.domain.evaluation import parse_scoring

    passes = []
    first_rows = None
    for i in range(repeats):
        rows = runner.run(parser_name, llm_provider=provider, llm_model=model)
        if i == 0:
            first_rows = rows
        report = parse_scoring.score(rows)
        passes.append({
            "micro_f1": report["micro"]["f1"],
            "exact_match": report["exact_match"],
            "topic_f1": report["topic_f1"],
            "mean_latency": report["mean_latency"],
            "parse_fail": sum(1 for r in rows if r.get("parse_error")),
        })
    median = {key: _median([p[key] for p in passes]) for key in passes[0]}
    return first_rows, median


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2, 4)


def _print_parse_summary(summary, repeats):
    plural = "s" if repeats > 1 else ""
    typer.echo(f"\n=== summary (median of {repeats} run{plural}) ===")
    typer.echo(
        f"{'model':<44}{'micro-F1':>9}{'exact':>8}{'topic-F1':>9}"
        f"{'latency(s)':>12}{'fail':>6}")
    for label, m in summary:
        typer.echo(
            f"{label:<44}{m['micro_f1']:>9}{m['exact_match']:>8}"
            f"{m['topic_f1']:>9}{m['mean_latency']:>12}{m['parse_fail']:>6}")


def _print_parse_report(name, rows, verbose):
    from qhld_engine.domain.evaluation import parse_scoring

    report = parse_scoring.score(rows)
    typer.echo(f"\n=== parser: {name} ===")
    if verbose:
        for row in rows:
            ok = "✓" if not any(
                parse_scoring.slot_counts(row["pred_filters"], row["gold"], s)[1:] != (0, 0)
                for s in parse_scoring.SLOTS) else "✗"
            typer.echo(f"  {ok} {row['id']} {row['query']!r}")
            typer.echo(f"      gold: {row['gold']}  topic={row.get('gold_topic')!r}")
            typer.echo(f"      pred: {row['pred_filters']}  topic={row['pred_topic']!r}")
    typer.echo(f"{'slot':<13}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    for slot, m in report["slots"].items():
        if m["tp"] + m["fp"] + m["fn"] == 0:
            typer.echo(f"{slot:<13}{'n/a':>7}{'n/a':>7}{'n/a':>7}   (no cases)")
            continue
        typer.echo(
            f"{slot:<13}{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}   "
            f"{m['tp']}/{m['fp']}/{m['fn']}")
    micro = report["micro"]
    typer.echo(
        f"{'MICRO':<13}{micro['precision']:>7}{micro['recall']:>7}{micro['f1']:>7}   "
        f"{micro['tp']}/{micro['fp']}/{micro['fn']}")
    typer.echo(
        f"  exact-match={report['exact_match']}  topic-F1={report['topic_f1']}  "
        f"mean-latency={report['mean_latency']}s")
    fails = sum(1 for row in rows if row.get("parse_error"))
    if fails:
        kinds = sorted({row["parse_error"] for row in rows if row.get("parse_error")})
        typer.echo(
            f"  parse-failures={fails}/{len(rows)} ({', '.join(kinds)}) "
            f"— counted as empty predictions")


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
