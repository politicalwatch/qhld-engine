"""`qhld eval` — A/B benchmark CLI.

One benchmark per task:
- ``retrieval`` — semantic search across an embedding-model x reranker grid
  (rank / MRR / hit@k / recall@k / MAP), over ``RunBenchmark``.
- ``pool`` — relevance-judging aid for the retrieval query set: pools each
  query's retrieved-but-unjudged references across cells for review.
- ``parse`` — the NL query parser across LLMs / rule-based (per-slot P/R/F1,
  cost + latency via LangSmith), over ``RunParseBenchmark``.
- ``mentions`` — index-time mention extraction against its gold set.
- ``fidelity`` — the retrieval layer against brute force (recall vs exact
  search), over ``RunFidelityBenchmark``. Needs no labels, so unlike
  ``retrieval`` it can be run at whatever n a question needs — which is what
  lets it arbitrate a retrieval-layer knob that ``retrieval`` cannot.
- ``gate`` — the intent gate, over ``RunGateBenchmark``: how much junk it
  refuses AND how many legitimate searches it refuses with it. The second half
  is the number the gate has never had.

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


def _cell_label(model, reranker, sparse="none"):
    label = model
    if reranker not in ("none", "noop"):
        label += f" + {reranker}"
    if sparse and sparse != "none":
        label += f" + {sparse}"
    return label


@app.command("retrieval")
def retrieval(
    models: str = typer.Option(..., "--models", help="Comma-separated ollama embedding tags."),
    rerankers: str = typer.Option("none", "--rerankers", help="Comma-separated rerankers ('none' = bi-encoder only)."),
    sparse: str = typer.Option(
        "none", "--sparse",
        help="Comma-separated sparse providers ('none' = dense only; 'bm25' = "
             "hybrid fusion — needs that model's hybrid collection indexed)."),
    k: int = typer.Option(10, "--k", help="Retrieval depth per query."),
    hit_at: int = typer.Option(5, "--hit-at", help="hit@k / recall@k metric threshold."),
    floors: str = typer.Option(
        "0", "--floors",
        help="Comma-separated relevance-floor cutoffs (e.g. '0,0.05,0.15'). Applied "
             "post-hoc to the reranked scores, so one retrieval run per cell scores "
             "every floor. Ignored on reranker-less cells (cosine/RRF scores are "
             "not thresholdable)."),
    queryset: str = typer.Option(None, "--queryset", help="Path to a query-set JSON (defaults to the frozen set)."),
    verbose: bool = typer.Option(False, "--verbose", help="Dump top-k per query."),
):
    """Benchmark each (model x reranker x sparse) cell over the frozen query set."""
    from qhld_engine.application.evaluation.benchmark import RunBenchmark

    runner = RunBenchmark(queryset) if queryset else RunBenchmark()
    model_list, reranker_list, sparse_list = _split(models), _split(rerankers), _split(sparse)
    floor_list = [float(value) for value in _split(floors)]
    typer.echo(
        f"Query set: {len(runner.queryset)} queries · retrieval k={k} · "
        f"metric @{hit_at} · models={model_list} · rerankers={reranker_list} · "
        f"sparse={sparse_list}"
    )
    for model in model_list:
        for reranker in reranker_list:
            for sp in sparse_list:
                rows = runner.run(model, reranker=reranker, sparse=sp, k=k)
                _print_report(_cell_label(model, reranker, sp), rows, hit_at, verbose)
                if any(floor_list):
                    if reranker in ("none", "noop"):
                        typer.echo(
                            "  [floor sweep skipped: no reranker in this cell — "
                            "cosine/RRF scores are not thresholdable]")
                    else:
                        _print_floor_sweep(rows, floor_list, hit_at)


@app.command("pool")
def pool(
    models: str = typer.Option(..., "--models", help="Comma-separated ollama embedding tags."),
    rerankers: str = typer.Option("none", "--rerankers", help="Comma-separated rerankers ('none' = bi-encoder only)."),
    sparse: str = typer.Option(
        "none", "--sparse",
        help="Comma-separated sparse providers ('none' = dense only; 'bm25' = "
             "hybrid fusion — needs that model's hybrid collection indexed)."),
    k: int = typer.Option(10, "--k", help="Retrieval depth per query."),
    queryset: str = typer.Option(None, "--queryset", help="Path to a query-set JSON (defaults to the frozen set)."),
    json_out: str = typer.Option(None, "--json", help="Also write the candidate pool to this JSON file."),
):
    """List candidate references for relevance judging: run the query set over
    every (model x reranker x sparse) cell and pool each query's retrieved
    references that are not yet judged (in neither expected_refs nor
    rejected_refs), with the best-ranked snippet as evidence. After judging,
    move each candidate into the query's expected_refs or rejected_refs — a
    re-run then reports no new candidates."""
    import json

    from qhld_engine.application.evaluation.benchmark import RunBenchmark
    from qhld_engine.domain.evaluation import scoring

    runner = RunBenchmark(queryset) if queryset else RunBenchmark()
    model_list, reranker_list, sparse_list = _split(models), _split(rerankers), _split(sparse)
    typer.echo(
        f"Query set: {len(runner.queryset)} queries · retrieval k={k} · "
        f"models={model_list} · rerankers={reranker_list} · sparse={sparse_list}"
    )
    rows_by_cell = {}
    for model in model_list:
        for reranker in reranker_list:
            for sp in sparse_list:
                rows_by_cell[_cell_label(model, reranker, sp)] = runner.run(
                    model, reranker=reranker, sparse=sp, k=k)
    candidates = scoring.pool_candidates(rows_by_cell)

    queries = {entry["id"]: entry["query"] for entry in runner.queryset}
    total = 0
    for query_id, query_candidates in candidates.items():
        typer.echo(f"\n=== {query_id} {queries[query_id]!r} ===")
        if not query_candidates:
            typer.echo("  (no new candidates)")
            continue
        total += len(query_candidates)
        for candidate in query_candidates:
            typer.echo(
                f"  {candidate['ref']:<12} rank={candidate['rank']:<3} "
                f"score={candidate['score']:<7} [{candidate['cell']}] "
                f"{candidate.get('lang')} · {candidate.get('speaker')}"
            )
            snippet = " ".join((candidate.get("text") or "").split())
            typer.echo(f"      {snippet[:300]}")
    typer.echo(f"\n{total} new candidates across {len(candidates)} queries")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(candidates, handle, ensure_ascii=False, indent=2)
        typer.echo(f"Pool written to {json_out}")


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
    reasoning: str = typer.Option(
        None, "--reasoning",
        help="Comma-separated reasoning-effort levels to sweep each model over "
             "(e.g. 'none,low,medium'); every model runs at every level. Values are "
             "model-specific and passed straight through — gpt-5.4-nano and "
             "gpt-5.6-luna take none/low/medium/high/xhigh. Omit for the provider "
             "default. NB: 'none' is also the only level where temperature reaches "
             "the API on gpt-5 models, so that cell differs in two ways."),
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
    # One cell per (model x effort). No --reasoning means a single cell per model at
    # whatever the provider defaults to, so the output is unchanged for callers that
    # don't ask for the sweep.
    efforts = _split(reasoning) if reasoning else [None]
    typer.echo(
        f"Parse A/B · {len(runner.queries)} queries · today={runner.today.isoformat()} "
        f"· repeats={repeats} · models={[label for _, _, label in specs]}"
        + (f" · reasoning={efforts}" if reasoning else "")
        + (" · +rule_based" if baseline else ""))
    summary = []
    for provider, model, label in specs:
        for effort in efforts:
            cell = f"{label} · effort={effort}" if effort else label
            try:
                first_rows, median = _run_scored(
                    runner, "llm", provider, model, repeats, effort)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not sink the sweep
                typer.echo(f"\n=== {cell}: FAILED ({type(exc).__name__}: {exc}) — skipped ===")
                continue
            _print_parse_report(cell, first_rows, verbose)
            summary.append((cell, median))
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
    """Score index-time mention extraction (NER → resolved people) against the frozen
    gold set: micro P/R/F1 over mentions, per-speech exact-match, latency — reported
    separately for deputies (unchanged basis) and the new non-deputy figures. Speeches
    whose gold carries occurrence counts also get a COUNTS line, which is the only
    dimension that can see a change attaching more occurrences to a person the speech
    already names."""
    from qhld_engine.application.evaluation.mentions_benchmark import RunMentionsBenchmark
    from qhld_engine.domain.evaluation import mentions_scoring

    runner = RunMentionsBenchmark(goldset) if goldset else RunMentionsBenchmark()
    typer.echo(f"Mentions gold set: {len(runner.entries)} speeches")
    rows = runner.run()
    deputy = mentions_scoring.score(rows, "pred_deputies", "gold_deputies")
    non_deputy = mentions_scoring.score(rows, "pred_non_deputies", "gold_non_deputies")
    counts = mentions_scoring.score_counts(rows)

    if verbose:
        for row in rows:
            for label, pk, gk in (("dep", "pred_deputies", "gold_deputies"),
                                  ("non", "pred_non_deputies", "gold_non_deputies")):
                missed = sorted(set(row.get(gk, [])) - set(row.get(pk, [])))
                spurious = sorted(set(row.get(pk, [])) - set(row.get(gk, [])))
                if missed or spurious:
                    typer.echo(f"  {row.get('id', row['speech_id'])} "
                               f"{row.get('reference', '')} [{label}]")
                    if missed:
                        typer.echo(f"      missed (FN): {missed}")
                    if spurious:
                        typer.echo(f"      spurious (FP): {spurious}")
        for miss in mentions_scoring.count_mismatches(rows):
            typer.echo(f"  {miss['id']} {miss['reference']} [count] "
                       f"{miss['name']}: gold {miss['gold']}, got {miss['pred']}")

    typer.echo("\n=== mentions ===")
    typer.echo(f"{'':<12}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    for label, rep in (("DEPUTIES", deputy), ("NON-DEPUTY", non_deputy)):
        m = rep["micro"]
        typer.echo(
            f"{label:<12}{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}   "
            f"{m['tp']}/{m['fp']}/{m['fn']}")
    typer.echo(
        f"  deputy exact-match={deputy['exact_match']}  "
        f"mean-latency={deputy['mean_latency']}s")
    if counts["pairs"]:
        typer.echo(
            f"  COUNTS      exact={counts['exact_rate']} "
            f"({counts['exact']}/{counts['pairs']} people over "
            f"{counts['speeches']} speeches)  "
            f"occurrences {counts['pred_occurrences']}/{counts['gold_occurrences']}  "
            f"under={counts['under']} over={counts['over']} "
            f"missing={counts['missing']}")


@app.command("bitext")
def bitext(
    instrument: str = typer.Option(
        "embeddings", "--instrument",
        help="classifier (end to end, the number that matters) | tokens | embeddings | llm."),
    goldset: str = typer.Option(
        None, "--goldset", help="Path to a bitext gold set (defaults to the frozen one)."),
    prompt_file: str = typer.Option(
        None, "--prompt-file",
        help="For --instrument llm: a prompt template with {a}, {b} and {lang}. The "
             "prompt is the operating point, so this is the knob worth sweeping."),
    verbose: bool = typer.Option(False, "--verbose", help="Show every disagreement."),
):
    """Score an instrument for "is this Spanish paragraph a rendering of that
    co-official one" against the hand-labelled gold set.

    Reports RANK and GATE separately because they are different questions. RANK — is the
    true partner the top candidate in this speech — is what the alignment needs. GATE — is
    there a threshold admitting every pair and rejecting every non-pair — is what deciding
    whether a rendering exists needs, and it is much harder: within one speech every
    paragraph shares the subject, the names and the figures. An instrument can rank
    perfectly and gate nothing, which is exactly what happens for Basque.
    """
    from qhld_engine.application.evaluation import bitext_benchmark as bench
    from qhld_engine.domain.evaluation import bitext_scoring

    runner = bench.RunBitextBenchmark(goldset)
    pairs = sum(len(s["pairs"]) for s in runner.entries)
    typer.echo(f"Bitext gold set: {len(runner.entries)} speeches, {pairs} pairs")

    if instrument in ("classifier", "classifier-tokens"):
        from qhld_ai.infrastructure.language import detect

        similarity = None
        if instrument == "classifier":
            from qhld_ai.application.speeches.paragraph_similarity import (
                create_paragraph_similarity)
            similarity = create_paragraph_similarity()
        rows = runner.run_classifier(detect, similarity=similarity)
        report = bitext_scoring.score_split(rows)
        typer.echo("")
        typer.echo(bitext_scoring.format_split(report, f"=== bitext ({instrument}) ==="))
        for miss in report["misses"]:
            typer.echo(
                f"    {miss['video_id']} [{miss['lang']}]"
                f"{' REFUSED' if miss['undecided'] else ''}"
                f"  left in as-delivered: {miss['missed'] or '-'}"
                f"  wrongly removed: {miss['spurious'] or '-'}")
        # The other direction: an original whose rendering the alignment missed stands in
        # the Spanish block as well as the as-delivered one, and a paragraph in both blocks
        # is invisible to the figures above.
        typer.echo("")
        typer.echo("  --- the co-official side ---")
        typer.echo(bitext_scoring.format_pairs(bitext_scoring.score_pairs(rows)))
        # And the third failure: a paragraph too short to have a language of its own is
        # absorbed into a neighbouring run, and leaves whichever block that run leaves.
        # Neither figure above separates that from an alignment that over-claimed a
        # paragraph it had actually read.
        typer.echo("")
        typer.echo("  --- block coverage ---")
        typer.echo(bitext_scoring.format_coverage(bitext_scoring.score_coverage(rows)))
        return

    if instrument == "llm":
        template = open(prompt_file).read() if prompt_file else None
        rows = runner.run_decided(bench.llm_judgement(template))
        report = bitext_scoring.score_decisions(rows)
        typer.echo("\n=== bitext (llm) ===")
        for lang, cell in sorted(report.items()):
            typer.echo(
                f"  {lang}  recall {cell['tp']}/{cell['tp'] + cell['fn']}="
                f"{cell['recall']:.2f}   specificity {cell['tn']}/"
                f"{cell['tn'] + cell['fp']}={cell['specificity']:.2f}   "
                f"precision={cell['precision']:.2f}" if cell["precision"] is not None
                else f"  {lang}  recall {cell['recall']}")
        existence = bitext_scoring.speech_level(rows)
        typer.echo(f"  speech-level: a rendering found in "
                   f"{existence['detected']}/{existence['speeches']} speeches that have "
                   f"one" + (f"; missed {existence['missed']}" if existence["missed"]
                             else ""))
    else:
        scorer = (bench.token_overlap() if instrument == "tokens"
                  else bench.embedding_cosine())
        rows = runner.run_scored(scorer)
        report = bitext_scoring.score(rows)
        typer.echo("")
        typer.echo(bitext_scoring.format_report(report, f"=== bitext ({instrument}) ==="))

    if verbose:
        for row in rows:
            said = row.get("decision", row.get("score"))
            if "decision" in row and bool(row["decision"]) != bool(row["is_pair"]):
                typer.echo(f"  {row['video_id']} [{row['lang']}] "
                           f"{row['source']}->{row['candidate']}: "
                           f"{'FALSE ALARM' if row['decision'] else 'MISSED'}")


@app.command("fidelity")
def fidelity(
    k: str = typer.Option(
        "10,50", "--k",
        help="Comma-separated retrieval depths. 50 is the product-relevant one — "
             "it is the reranker's candidate pool (hybrid_prefetch_limit)."),
    arm: str = typer.Option(
        "corpus", "--arm",
        help="corpus (vectors sampled from the collection; deterministic, needs no "
             "embedder) | real (the 45 labelled + 49 parser queries, embedded) | both."),
    collection: str = typer.Option(
        None, "--collection",
        help="Override the derived collection name. Skips embedder-based name "
             "resolution, so the corpus arm then needs nothing but Qdrant."),
    graph_reference: bool = typer.Option(
        False, "--graph-reference",
        help="Also run the uncompressed reference collection, which isolates HNSW's "
             "own error from quantization's."),
    filters: bool = typer.Option(
        False, "--filters",
        help="Sweep the payload-filter cells, stratified by cardinality."),
    sparse: bool = typer.Option(
        False, "--sparse",
        help="Measure the fused dense+sparse shape the product serves, instead of the "
             "dense branch alone. Fusion dilutes whatever the dense branch loses, so "
             "this is the product-facing figure, not the sensitive one."),
    size: int = typer.Option(None, "--size", help="Override the frozen sample size."),
    cache: str = typer.Option(
        None, "--cache", help="Path to cache the real arm's query embeddings."),
    json_out: str = typer.Option(None, "--json", help="Write per-cell results here."),
):
    """Measure how faithfully the retrieval layer finds what brute force would.

    Ground truth is ``exact: true`` on the same collection, so no labels are
    needed and the query set is unbounded — which is the point. ``qhld eval
    retrieval`` was measured unable to arbitrate a retrieval-layer change
    (saturated at MRR 1.0, n=4-6 per dimension, opposite signs across cells);
    this can, because it measures what ENTERS the pipeline rather than what
    leaves it.

    Read the result as fidelity, never as relevance: it says whether a cell finds
    what an exhaustive search would, not whether that is the right answer.
    """
    from qhld_engine.application.evaluation.fidelity_benchmark import (
        RunFidelityBenchmark,
    )

    runner = RunFidelityBenchmark(collection=collection)
    depths = [int(value) for value in _split(k)]
    cells = runner.asset["cells"]["default"]
    arms = ["corpus", "real"] if arm == "both" else [arm]

    targets = [(runner.collection, "serving")]
    if graph_reference:
        targets.append((runner.asset["reference_collections"]["float32"], "graph reference"))

    results = {}
    for name, role in targets:
        typer.echo(f"\n=== {name} ({role}) ===")
        for arm_name in arms:
            vectors, sparse_vectors, provenance = _fidelity_arm(
                runner, arm_name, name, size, cache, sparse)
            typer.echo(f"  arm={arm_name} n={len(vectors)} · {provenance}")
            for filter_cell in (runner.filter_cells() if filters else [{"label": None, "filter": None}]):
                if filter_cell["label"]:
                    count = runner.count(filter_cell["filter"], name)
                    typer.echo(
                        f"\n  filter {filter_cell['label']} · {count} points "
                        f"({count / max(runner.count(None, name), 1):.1%})")
                rows = _run_fidelity_cells(
                    runner, vectors, sparse_vectors, cells, depths, name,
                    filter_cell["filter"])
                _print_fidelity(rows, depths)
                results.setdefault(name, {}).setdefault(arm_name, {})[
                    filter_cell["label"] or "unfiltered"] = rows

    if json_out:
        import json

        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        typer.echo(f"\nwrote {json_out}")


@app.command("gate")
def gate(
    repeats: int = typer.Option(
        5, "--repeats", min=1,
        help="Passes over each arm. The gate runs on an LLM call, so its verdict is "
             "not stable per query; the report is a BAND across passes plus a "
             "per-query refusal frequency, never a point estimate."),
    models: str = typer.Option(
        None, "--models",
        help="Comma-separated 'provider:model' specs to sweep the parser over "
             "(same form as `eval parse`). Omit to measure the production cell."),
    reasoning: str = typer.Option(
        None, "--reasoning",
        help="Comma-separated reasoning-effort levels to sweep each model over."),
    arms: str = typer.Option(
        "legitimate,non-search,unsupported-language,junk", "--arms",
        help="Which arms to run. 'legitimate' measures false positives; the other "
             "three measure misses, but only 'junk' has the relevance floor behind "
             "it, so they are reported separately and never merged."),
    queryset: str = typer.Option(None, "--queryset", help="Path to a gate query-set JSON."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Dump every query with its outcome and parse."),
):
    """Measure the intent gate in both directions: the junk it refuses, and the
    legitimate searches it refuses with it.

    The second half is the gap this command exists to close. The gate's only
    number on record — 23/24 junk probes suppressed — was measured on the
    RELEVANCE FLOOR, by a harness that runs no parse and no gate at all, over a
    set containing nothing but junk. So it says nothing about how often a real
    search is turned away, and a refusal fails closed: the user is told this is
    not a speech search, which they cannot tell from the product being broken.
    """
    from qhld_engine.application.evaluation.gate_benchmark import (
        RunGateBenchmark, VocabularyError)
    from qhld_engine.domain.evaluation import gate_scoring

    runner = RunGateBenchmark(queryset) if queryset else RunGateBenchmark()
    try:
        collection, vocabulary = runner.check_vocabulary()
    except VocabularyError as exc:
        typer.echo(f"ABORTED: {exc}")
        raise typer.Exit(code=1)

    wanted = _split(arms)
    specs = _parse_models(models) if models else [(None, None, runner.model_label())]
    efforts = _split(reasoning) if reasoning else [None]
    typer.echo(
        f"Intent gate · legitimate={len(runner.legitimate)} "
        f"non-search={len(runner.non_search)} "
        f"unsupported-language={len(runner.unsupported_language)} "
        f"junk={len(runner.junk)} "
        f"· repeats={repeats} · today={runner.today.isoformat()}\n"
        f"  collection={collection} ({vocabulary} distinct speakers) · arms={wanted}")

    for provider, model, label in specs:
        for effort in efforts:
            cell = f"{label} · effort={effort}" if effort else label
            scored = {}
            for arm in wanted:
                runs = [
                    runner.run(arm, llm_provider=provider, llm_model=model,
                               reasoning_effort=effort)
                    for _ in range(repeats)
                ]
                scored[arm] = gate_scoring.score_arm(runs)
                _print_gate_arm(cell, arm, scored[arm], verbose)
            # One matrix per refuse-arm. Merging them would average an
            # unbackstopped failure with a backstopped one.
            for arm in ("non-search", "unsupported-language", "junk"):
                if "legitimate" in scored and arm in scored:
                    _print_gate_confusion(cell, scored["legitimate"], scored[arm], arm,
                                          repeats)


def _print_gate_arm(cell, arm, report, verbose):
    from qhld_engine.domain.evaluation.gate_scoring import (
        REFUSED_EMPTY, REFUSED_FLAG, REFUSED_LANGUAGE)

    reading = {
        "legitimate": "refusing these is a FALSE POSITIVE",
        "non-search": "passing these is a MISS — nothing downstream catches them",
        "unsupported-language": "passing these serves a language we do not support",
        "junk": "passing these is a MISS, but the relevance floor is a backstop",
    }.get(arm, arm)
    band = report["band"]
    typer.echo(f"\n=== {cell} · arm={arm} · n={report['n']} ===")
    typer.echo(f"  {reading}")
    typer.echo(
        f"  refusal rate  band {band['min']}–{band['max']}  median {band['median']}")
    typer.echo(
        f"  ever refused  {report['ever']}/{report['n']} ({report['ever_rate']})   "
        f"always {report['always']}/{report['n']} ({report['always_rate']})")
    if report["ceiling_95"] is not None:
        typer.echo(
            f"  NOT a measured zero — no refusal in {report['n']} probes bounds the "
            f"true rate at {report['ceiling_95']} (95%, rule of three). Widen the set "
            "to tighten it.")
    first = report["per_run"][0]
    sites = first["by_site"]
    typer.echo(
        f"  by site       flag={sites[REFUSED_FLAG]}  empty-parse={sites[REFUSED_EMPTY]}"
        f"  language={sites[REFUSED_LANGUAGE]}   (first pass)")
    if first["wrong_reason"]:
        typer.echo(
            f"  WRONG REASON  {first['wrong_reason']} refused, but not the way the probe "
            f"expects — right verdict, wrong message: "
            f"{', '.join(first['wrong_reason_ids'])}")
    typer.echo(f"  {'class':<16}{'n':>4}{'refused':>9}{'rate':>8}")
    for name, bucket in sorted(report["per_run"][0]["by_class"].items()):
        typer.echo(
            f"  {name:<16}{bucket['n']:>4}{bucket['refused']:>9}{bucket['rate']:>8}")
    if report["unstable"]:
        typer.echo(
            f"  UNSTABLE — refused on some passes and not others "
            f"({len(report['unstable'])}):")
        for entry in report["unstable"]:
            typer.echo(
                f"    {entry['id']:<5}{entry['refused']}/{entry['repeats']} "
                f"{','.join(entry['sites']):<14} {entry['query']!r}")
    refused = [e for e in report["queries"] if e["refused"]]
    if refused and not verbose:
        typer.echo("  refused at least once:")
        for entry in refused:
            typer.echo(
                f"    {entry['id']:<5}{entry['refused']}/{entry['repeats']} "
                f"[{entry['class']}] {entry['query']!r}")
    if verbose:
        for entry in report["queries"]:
            mark = "✓" if not entry["refused"] else "✗"
            routes = "/".join(entry["routes"]) or "-"
            typer.echo(
                f"  {mark} {entry['id']:<5}{entry['refused']}/{entry['repeats']} "
                f"[{entry['class']}] route={routes} {entry['query']!r}")


def _print_gate_confusion(cell, legitimate, positives, arm, repeats):
    from qhld_engine.domain.evaluation import gate_scoring

    matrix = gate_scoring.confusion(legitimate, positives)
    typer.echo(
        f"\n=== {cell} · gate vs {arm} (median pass of {repeats}) ===")
    typer.echo("  positive = 'refuse the query'")
    typer.echo(
        f"  {arm} refused (TP) {matrix['tp']:>3}   {arm} passed (FN)  {matrix['fn']:>3}")
    typer.echo(
        f"  legit refused (FP){matrix['fp']:>4}   legit passed (TN) {matrix['tn']:>4}")
    typer.echo(
        f"  FALSE-POSITIVE RATE {matrix['false_positive_rate']}   "
        f"suppression {matrix['suppression']}")
    typer.echo(
        f"  precision {matrix['precision']}   recall {matrix['recall']}   "
        f"F1 {matrix['f1']}")
    typer.echo(
        "  Cite as a band with n, repeats, date and model — the gate is an LLM call, "
        "so a point estimate would claim a stability it does not have.")


def _fidelity_arm(runner, arm_name, collection, size, cache, sparse):
    """Build one arm's query vectors, plus the sparse companions the fused shape
    needs. Returns the provenance string too: a fidelity number is meaningless
    without knowing which corpus state produced the queries."""
    if arm_name == "corpus":
        vectors, fingerprint = runner.sample_corpus_vectors(collection, size=size)
        recorded = runner.asset["fingerprints"].get(collection)
        if recorded and recorded != fingerprint:
            drift = "CORPUS MOVED — not comparable to the recorded baseline"
        elif recorded:
            drift = "matches the recorded baseline"
        else:
            drift = "no baseline recorded yet"
        provenance = f"seed={runner.asset['sample']['seed']} sha={fingerprint[:12]} ({drift})"
        texts = None
    else:
        vectors = runner.real_query_vectors(cache_path=cache)
        texts = runner.real_query_texts()
        provenance = f"real queries, embedded with {runner.settings.embedding_model}"

    sparse_vectors = None
    if sparse:
        if texts is None:
            raise typer.BadParameter(
                "--sparse needs query TEXT to build a lexical vector, which the "
                "corpus arm does not have (it samples vectors, not documents). "
                "Use --arm real with --sparse.")
        from qhld_ai.infrastructure.sparse.factory import create_sparse_embedder_from_env

        embedder = create_sparse_embedder_from_env(runner.settings)
        sparse_vectors = [embedder.embed_query(text) for text in texts]
    return vectors, sparse_vectors, provenance


def _run_fidelity_cells(runner, vectors, sparse_vectors, cells, depths, collection,
                        query_filter):
    rows = []
    for depth in depths:
        for cell in cells:
            row = runner.run_cell(
                vectors, cell, depth, collection, query_filter, sparse_vectors)
            row["k"] = depth
            rows.append(row)
    return rows


def _print_fidelity(rows, depths):
    """One block per depth, with a paired bootstrap against the first cell — the
    arm that currently serves — so a difference is reported as real or as noise
    rather than left for the reader to eyeball."""
    from qhld_engine.domain.evaluation import fidelity_scoring

    for depth in depths:
        block = [row for row in rows if row["k"] == depth]
        if not block:
            continue
        typer.echo(
            f"\n    k={depth}  {'cell':<44}{'recall':>8}{'perfect':>10}"
            f"{'worst':>8}{'1st miss':>10}   vs baseline")
        baseline = block[0]
        for row in block:
            delta = ""
            if row is not baseline:
                test = fidelity_scoring.paired_bootstrap(
                    baseline["per_query"], row["per_query"])
                if test:
                    delta = (
                        f"{test['delta'] * 100:+.2f} pp "
                        f"[{test['lo'] * 100:+.2f}, {test['hi'] * 100:+.2f}] "
                        f"{'SIGNIFICANT' if test['significant'] else 'n.s.'}")
            shallowest = row["shallowest"] if row["shallowest"] is not None else "-"
            typer.echo(
                f"    {'':6}{row['label']:<44}"
                f"{fidelity_scoring.format_recall(row['mean']):>8}"
                f"{str(row['perfect']) + '/' + str(row['n']):>10}"
                f"{fidelity_scoring.format_recall(row['worst']):>8}"
                f"{str(shallowest):>10}   {delta}")


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


def _run_scored(runner, parser_name, provider, model, repeats, effort=None):
    """Run a parser ``repeats`` times; return (first-pass rows, median-metrics dict)."""
    from qhld_engine.domain.evaluation import parse_scoring

    passes = []
    first_rows = None
    for i in range(repeats):
        rows = runner.run(parser_name, llm_provider=provider, llm_model=model,
                          reasoning_effort=effort)
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


def _print_report(label, rows, hit_at, verbose):
    from qhld_engine.domain.evaluation import scoring

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
    junk = scoring.suppression(rows)
    if junk:
        typer.echo(
            f"    {'offdomain':<13} n={junk['n']:<3} suppressed={junk['suppressed']}/{junk['n']} "
            f"max-leak={junk['max_leak'] if junk['max_leak'] is not None else '-'}"
        )


def _print_floor_sweep(rows, floors, hit_at):
    """One line per floor value, re-scored post-hoc from the cell's single run:
    retention (recall@k / MAP) per dimension plus junk suppression."""
    from qhld_engine.domain.evaluation import scoring

    dimensions = list(scoring.aggregate(rows, k=hit_at))
    typer.echo("\n  floor sweep (post-hoc cutoff on the reranked scores):")
    typer.echo(
        f"    {'floor':<7}"
        + "".join(f"{dim[:16]:>18}" for dim in dimensions)
        + f"{'offdomain':>18}"
    )
    typer.echo(
        f"    {'':<7}"
        + "".join(f"{f'R@{hit_at}/MAP':>18}" for _ in dimensions)
        + f"{'leaks max-score':>18}"
    )
    for floor in floors:
        floored = scoring.apply_floor(rows, floor)
        agg = scoring.aggregate(floored, k=hit_at)
        junk = scoring.suppression(floored)
        cells = "".join(
            f"{agg[dim][f'recall_at_{hit_at}']:.3f}/{agg[dim]['map']:.3f}".rjust(18)
            for dim in dimensions
        )
        if junk:
            leaks = junk["n"] - junk["suppressed"]
            max_leak = junk["max_leak"] if junk["max_leak"] is not None else "-"
            junk_cell = f"{leaks}/{junk['n']} {max_leak}".rjust(18)
        else:
            junk_cell = f"{'-':>18}"
        typer.echo(f"    {floor:<7}" + cells + junk_cell)


def _dump(row):
    typer.echo(f"  · {row['id']} {row['query']!r}")
    for position, hit in enumerate(row.get("hits", []), start=1):
        payload = hit.payload
        typer.echo(
            f"      {position:>2}. [{hit.score:.3f}] "
            f"{'+'.join(payload.get('references') or [])} "
            f"· {payload.get('lang')} · {payload.get('speaker')}"
        )
