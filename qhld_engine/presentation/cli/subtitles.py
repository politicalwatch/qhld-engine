"""`qhld subtitles` — timed subtitle cues for an intervention.

``align`` runs forced alignment of one speech's stored transcript against its own
video, so the subtitles carry the stenographers' words and only their timing comes
from a model. The service is imported lazily so ``--help`` never connects to Mongo
or downloads the acoustic model.
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
    force: bool = typer.Option(
        False, "--force",
        help="Re-align a speech that already has cues. Off by default because it "
             "costs a fresh download of the whole video."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Align and report, but store nothing."),
):
    """Align one speech's transcript to its video and store the cues."""
    from qhld_ai.domain.subtitles import Cue, render_vtt
    from tipi_data import DoesNotExist
    from tipi_data.repositories.speeches import Speeches

    from qhld_engine.application.speeches.align_speech import AlignSpeech

    try:
        speech = Speeches.get(speech_id)
    except DoesNotExist:
        # The public URLs are keyed on the Congress intervention id, so that is what
        # a person reading the site has to hand.
        speech = Speeches.get_by_video_id(speech_id)

    alignment = AlignSpeech().execute(speech.id, force=force, persist=not dry_run)
    typer.echo(
        f"{speech.id}: {len(alignment.cues)} cues, {alignment.lang}, "
        f"score {alignment.score} ({alignment.verdict})"
        + (" — not stored" if dry_run else ""))

    if not vtt:
        return
    text = speech.speech[alignment.block_index].text
    track = render_vtt(
        [Cue(char_start=cue.char_start, char_end=cue.char_end,
             start=cue.start_ms / 1000, end=cue.end_ms / 1000)
         for cue in alignment.cues],
        text)
    if vtt == "-":
        typer.echo(track)
    else:
        with open(vtt, "w") as handle:
            handle.write(track)
        typer.echo(f"wrote {vtt}")
