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
    chosen = _track(alignments, lang)
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
        typer.echo(f"wrote {vtt} ({chosen.lang})")


def _track(alignments, lang):
    """The alignment ``--vtt`` should render.

    Without ``--lang`` this is the as-delivered one, which is the track a monolingual
    speech has and the one a reader checking the timings against the video wants.
    """
    if lang is None:
        return next((a for a in alignments if a.original), alignments[0])
    for alignment in alignments:
        if alignment.lang == lang:
            return alignment
    raise typer.BadParameter(
        f"no {lang} track; this speech has "
        + ", ".join(a.lang for a in alignments))
