"""`qhld diagnostics` — read what the search gates turned away and could not identify.

Every natural-language search records what it could not do to ``db.search_diagnostics``:
values the resolver failed to identify, ties it broke arbitrarily, and whole queries it
refused. This reads that collection back. Nothing here writes.

The refusals view exists to be run regularly rather than once. A query refused as a
prompt injection also bans the address that sent it, so the rows are the record of what
those bans were based on — and the only way to notice a query that was classified hostile
and should not have been. Such a query belongs in the gate probe set as a new legitimate
row, which is why ``--examples`` prints the text verbatim: that view exists to copy from.

Reading the collection means reading text that callers wrote, some of it deliberately
hostile. It is stored sanitised (control characters stripped, truncated), so printing it
is safe; do not pipe it into anything that treats input as instructions.

Runs against Mongo, so on the host it needs the usual overrides:

    MONGO_HOST=localhost MONGO_PORT=27018 qhld diagnostics refusals

Inside the container the defaults already point at the right host.
"""

import typer

app = typer.Typer(help="Read what search could not identify, and what it refused.")

# How much of a query the table view shows. The full text is one --examples away, and a
# terminal-width row is what makes a long list scannable, which is the whole point of it.
_WIDTH = 78

BY_COUNT = "count"
BY_RECENT = "recent"


def _truncate(text, width=_WIDTH):
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _sorted(rows, order):
    if order == BY_RECENT:
        # Missing last_seen sorts oldest rather than crashing: these documents predate
        # nothing, but a hand-inserted or partially-migrated row should not take the
        # command down.
        return sorted(rows, key=lambda r: (r.last_seen is not None, r.last_seen),
                      reverse=True)
    return sorted(rows, key=lambda r: (r.count, r.last_seen is not None), reverse=True)


def _when(moment):
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


@app.command("refusals")
def refusals(
    reason: str = typer.Option(
        None, "--reason",
        help="Only this refusal reason: 'prompt_injection' or 'not_a_speech_search'. "
             "The stored 'refused_' prefix is optional."),
    sort: str = typer.Option(
        BY_COUNT, "--sort",
        help="'count' for what we turn away most, 'recent' for what has changed since "
             "the last review — the order to hunt false positives in."),
    limit: int = typer.Option(50, "--limit", min=1, help="Rows to show."),
    examples: bool = typer.Option(
        False, "--examples",
        help="Print the stored queries verbatim, with when they were seen and which "
             "parser read them. This is the view to copy a probe out of."),
):
    """The queries the search gates refused.

    A `prompt_injection` row IS a ban: that class bans on the first occurrence, so every
    row here is an address that was turned away. Read them for anything that should not
    have been refused at all, or should have been refused without a ban.
    """
    from tipi_data.repositories.search_diagnostics import SearchDiagnostics

    if sort not in (BY_COUNT, BY_RECENT):
        raise typer.BadParameter(f"--sort must be {BY_COUNT!r} or {BY_RECENT!r}")

    rows = SearchDiagnostics.refusals()
    if reason:
        wanted = reason if reason.startswith("refused_") else f"refused_{reason}"
        rows = [row for row in rows if row.outcome == wanted]

    if not rows:
        # An empty result is a real answer here — no refusals recorded — and saying so
        # beats printing a header over nothing and leaving the reader wondering whether
        # the filter or the collection was empty.
        print("No refusals recorded" + (f" for {reason!r}" if reason else "") + ".")
        return

    shown = _sorted(rows, sort)[:limit]
    total = sum(row.count for row in rows)
    banned = sum(row.count for row in rows if row.outcome == "refused_prompt_injection")
    print(f"{len(rows)} distinct refused queries · {total} refusals · "
          f"{banned} of them bans (prompt_injection)")
    print(f"showing {len(shown)} by {sort}\n")

    for row in shown:
        print(f"  {row.count:>5}×  {_when(row.last_seen)}  {row.outcome}")
        print(f"         {_truncate(row.key)}")
        if row.languages:
            print(f"         read as: {', '.join(sorted(row.languages))}")
        if examples:
            for example in row.examples:
                stamp = _when(example.get("at"))
                model = example.get("parser_model") or "provider default"
                print(f"         · [{stamp}] ({model}) {example.get('query', '')}")
        print()


@app.command("gaps")
def gaps(
    field: str = typer.Option(
        None, "--field",
        help="Only this resolver field: 'mentions', 'speaker' or 'entities'. "
             "Omit for every gap, worst first."),
    limit: int = typer.Option(50, "--limit", min=1, help="Rows to show."),
    examples: bool = typer.Option(
        False, "--examples", help="Print the queries each gap was seen in."),
):
    """The people and themes the resolver could not identify, worst first.

    "Worst" is how many real people got nothing back, not raw popularity — a value that
    made a query unsatisfiable outranks one that merely resolved oddly. `unresolved` rows
    are missing catalog entries; `ambiguous` rows resolved successfully but by breaking a
    tie, so more than one name under `chosen` means the answer moves between restarts.
    """
    from tipi_data.repositories.search_diagnostics import SearchDiagnostics

    rows = (SearchDiagnostics.by_field(field) if field
            else [row for row in SearchDiagnostics.get_all()
                  if not row.outcome.startswith("refused_")])
    if not rows:
        print("No gaps recorded" + (f" for field {field!r}" if field else "") + ".")
        return

    print(f"{len(rows)} gaps · {sum(row.count for row in rows)} sightings · "
          f"{sum(row.blocking_count for row in rows)} left a query unsatisfiable\n")
    for row in rows[:limit]:
        blocking = f", {row.blocking_count} blocking" if row.blocking_count else ""
        print(f"  {row.count:>5}×{blocking}  {row.field}:{row.key}  ({row.outcome})")
        if row.surface_forms:
            print(f"         seen as: {', '.join(sorted(row.surface_forms))}")
        # Kept even when every entry is null: "we had no candidate at all" is the finding
        # that means a person is missing from the catalog entirely.
        if row.suggestions:
            hints = ", ".join(str(hint) for hint in row.suggestions)
            print(f"         hints:   {hints}")
        if row.chosen:
            print(f"         chose:   {', '.join(sorted(row.chosen))}"
                  f"{'  (unstable)' if len(row.chosen) > 1 else ''}")
        if examples:
            for example in row.examples:
                print(f"         · [{_when(example.get('at'))}] {example.get('query', '')}")
        print()
