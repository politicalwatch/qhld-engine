"""`qhld subtitles` — timed subtitle cues for an intervention.

``align`` runs forced alignment of one speech's stored transcript against its own
video, so the subtitles carry the stenographers' words and only their timing comes
from a model. A co-official-language intervention yields two tracks, one per language
block. The service is imported lazily so ``--help`` never connects to Mongo or
downloads the acoustic model.
"""

import typer

app = typer.Typer(help="Generate timed subtitle cues for interventions.")


@app.command("align")
def align(
    speech_id: str = typer.Argument(
        ..., help="Speech id, or the Congress intervention id the public URLs use."),
    vtt: str | None = typer.Option(
        None, "--vtt",
        help="Also write the WebVTT track here; '-' prints it. For inspection only "
             "— the track served to players is rendered on demand from the stored "
             "cues, never from a file."),
    lang: str | None = typer.Option(
        None, "--lang",
        help="Which track --vtt writes. Defaults to the as-delivered language; a "
             "co-official speech also has one in Spanish."),
    translation: bool = typer.Option(
        False, "--translation",
        help="Write the Diario's rendering rather than what was said — the only way "
             "to name it on a speech whose two blocks are both Spanish."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-align languages that already have cues. Off by default because it "
             "costs a fresh download of the whole video."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Align and report, but store nothing."),
):
    """Align a speech's transcript to its video and store the cues, one track per
    language block."""
    from qhld_ai.domain.subtitles import subtitle_track
    from tipi_data import DoesNotExist
    from tipi_data.repositories.speeches import Speeches

    from qhld_engine.application.speeches.align_speech import AlignSpeech

    try:
        speech = Speeches.get(speech_id)
    except DoesNotExist:
        # The public URLs are keyed on the Congress intervention id, so that is what
        # a person reading the site has to hand.
        speech = Speeches.get_by_video_id(speech_id)

    alignments = AlignSpeech().execute(speech.id, force=force, persist=not dry_run)
    for alignment in alignments:
        kind = "as delivered" if alignment.original else "translation"
        typer.echo(
            f"{speech.id} [{alignment.lang}, {kind}]: {len(alignment.cues)} cues, "
            f"score {alignment.score} ({alignment.verdict})"
            + (" — not stored" if dry_run else ""))

    if not vtt:
        return
    chosen = _track(alignments, lang, False if translation else None)
    # The same renderer the API serves from, guard included — so what is written
    # here is exactly what a player would receive.
    track = subtitle_track(chosen, speech.speech)
    if track is None:
        # Only reachable when an existing alignment was reused: its cues index a
        # transcript that has since changed, and captioning one sentence with
        # another is worse than no subtitles. Re-run with --force.
        raise typer.BadParameter(
            f"{speech.id} was aligned against a different version of its "
            "transcript; re-align it with --force")
    if vtt == "-":
        typer.echo(track)
    else:
        with open(vtt, "w") as handle:
            handle.write(track)
        typer.echo(f"wrote {vtt} ({chosen.lang}, "
                   f"{'as delivered' if chosen.original else 'translation'})")


def _track(alignments, lang, original=None):
    """The alignment ``--vtt`` should render.

    Without ``--lang`` this is the as-delivered one, which is the track a monolingual
    speech has and the one a reader checking the timings against the video wants. A
    speech whose co-official passage the Diario also printed in Spanish has two ``es``
    tracks, and ``--translation`` is what tells them apart; asked by language alone, the
    as-delivered one answers, as it does everywhere else.
    """
    if lang is None and original is None:
        return next((a for a in alignments if a.original), alignments[0])
    wanted = [a for a in alignments
              if (lang is None or a.lang == lang)
              and (original is None or bool(a.original) is original)]
    if wanted:
        return next((a for a in wanted if a.original), wanted[0])
    raise typer.BadParameter(
        f"no {lang or 'matching'} track; this speech has "
        + ", ".join(f"{a.lang} ({'as delivered' if a.original else 'translation'})"
                    for a in alignments))
