# Fixture: speeches for initiative 172/000001

Real captured data for an *interpelación urgente* (XV legislature) — chosen because it
exercises the tricky cases end-to-end:

- a **parliamentarian** speaking a co-official language (Néstor Rego, GMx, Galician), and
- a **government member** with no parliamentary group whose Diario heading is role-based
  (`La señora MINISTRA DE INCLUSIÓN, SEGURIDAD SOCIAL Y MIGRACIONES (Saiz Delgado):`).

## Files
- `interventions_page1.json` — the intervention-search API response (`CongressApi.get_video`),
  page 1 (4 interventions, fits one page).
- `session_raw.txt` — the raw Diario de Sesiones text (`PDFExtractor(..., format_output=False)`),
  **windowed**: trimmed to ~1.5k chars before the `(Número de expediente 172/000001)` anchor
  through the boundary after the last intervention. `normalize_session_text` discards everything
  before the anchor anyway, and the segmenter never reads past the last speaker, so this window
  reproduces byte-identical extraction to the full 463k-char session PDF (verified at capture time).
- `session_link.txt` — the session PDF path the service builds for this initiative.

## Regenerating / adding more
Capture with the live stack up (or host network): fetch `get_video(ref, 1).json()` and
`PDFExtractor(link, format_output=False).retrieve()`, then keep `raw[anchor.start()-1500 :
anchor.end()+~55k]`, asserting the windowed text yields the same speeches as the full text.
Consumed by `tests/unit/application/test_speech_extraction_characterization.py`.
