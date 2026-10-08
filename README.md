# Extract Stills V2

Offline, scene-aware still selection for Windows x64 and Apple Silicon Macs. Smart extraction is the default; the original every-24-frames extractor remains available as Legacy mode. V2 is a beta on the codex/v2.0 branch. Main contains the unchanged V1 baseline.

The current Mac development bundle is tested on macOS 15 and contains dependencies requiring macOS 15. Support for macOS 13 remains a stable-release target; use the current Mac bundle on macOS 15 or newer.

## Desktop app

Windows: open **Extract Stills.exe** in the complete GUI distribution folder. macOS: open **Extract Stills.app**. Keep the entire distribution together; models, Python, Qt and FFmpeg are bundled. Extraction needs no internet connection.

Drop video files or folders, choose Smart or Legacy, and run. Smart defaults to original-resolution 8-bit PNG in sRGB. Folder scans are nonrecursive unless enabled. Each source gets a separate run folder beside it under stills, or under the chosen output root.

Enable **Review before export** to choose scored candidates yourself. Thumbnails are grouped by scene; click a preview for a larger view. Low/Mid/High scores describe local sharpness, text stability, exposure and facial-expression signals. Recognized wording and Words ready/incomplete cues help review animated or cropped text. Reports are optional.

Smart favors open eyes with a small smile bonus, settled titles and scene coverage. It targets 6/10/15/20 stills for 6/15/30/over-30-second videos, interpolates between anchors, and increases the automatic count for distinct scenes and complete readable wording occurrences. Automatic counts remain the default. Quality shortfalls produce fewer stills and an explanation. Count override is a maximum; reports explain any wording occurrences excluded by that cap.

Focus ranking combines multiple native-pixel scales, noise correction and edge acutance. It checks original-resolution face/text regions and scene tiles, then searches wider native-frame neighborhoods around promising moments. Continuous rankings distinguish sharp candidates that previously received the same score; distant weaker frames cannot pad a run solely for temporal variety.

Local Latin-script OCR identifies changed words even inside a continuous shot. Native regional detection repairs fragmented word crops and long fine print; temporal consensus waits for complete, unclipped, readable wording instead of brief typing/scroll fragments. Per-character confidence distinguishes uncertain marks from strong words. Native color-pixel checks help resolve OCR fluctuations while preserving real letter, digit and colored-text changes. The same wording can recur later, while a repeated identical image is exported once and linked to both occurrences. OCR confidence and completeness are evidence-based estimates, not guarantees of spelling or original opacity.

Smart analysis takes longer than the earlier beta because it inspects more original-resolution moments and reads their text locally. Temporary pixel evidence is released after selection; saved analyses retain the decisions and diagnostics needed for review and export.

The ending selection is a stable frame near the end of the last usable shot. Text stability is an estimate from video pixels, not a measurement of original text opacity. Detection is heuristic; serious expressions and unreliable or tiny faces are treated appropriately rather than requiring a smile.

## Source installation

Use Python 3.12 and FFmpeg with zscale, tonemap and iccgen filters. The pinned OpenCV package requires macOS 13+. Use one OpenCV distribution per environment.

    python -m venv .venv
    # Activate .venv in your shell, then:
    python -m pip install -r requirements-lock.txt
    python -m pip install --no-deps -e .
    python scripts/setup_models.py
    extract-stills gui

Model setup is an explicit, one-time download from the sources in models/manifest.json. Runtime never downloads models; missing or modified assets fail with an actionable error. Verify offline with:

    python scripts/setup_models.py --verify

The source package uses editable installation so it can locate the checkout's model assets. For an installation without Python, use the complete native app bundle.

## Commands

The console distribution has its own extract-stills executable and models. Console commands work without opening Qt:

    extract-stills extract "clip.mov"
    extract-stills extract "deliverables" --recursive --report --json
    extract-stills extract "clip.mov" --count 12 --format tiff --bit-depth 16
    extract-stills extract "clip.mov" --format png --bit-depth 16 --color source
    extract-stills analyze "clip.mov" --output "review" --report --json
    extract-stills export "review/clip/run_YYYYMMDD_HHMMSS/analysis.json" --frames 24,96,180
    extract-stills extract "clip.mov" --mode legacy

Alternatively use **python -m stills_tool** with the same arguments. Passing a video directly is shorthand for extract.

Frame IDs are zero-based source frame indexes, including in reports and explicit selections. Original timestamps are retained for variable frame rates. Analyze writes analysis.json and candidate previews; export validates the original video fingerprint before using that analysis. Re-exporting creates another run folder.

JSON stdout contains version, status, results and errors; diagnostics/progress go to stderr. Batches continue after individual errors. Exit status is 0 on success, 1 for processing failures, and 2 for argument syntax errors.

## Output and color

- PNG and TIFF: 8-bit or 16-bit; JPEG: 8-bit only.
- sRGB is the default, including HDR-to-SDR tone mapping and embedded ICC profiles.
- Source color requires 16-bit PNG/TIFF, preserves source transfer/gamut and writes a color metadata sidecar. Use a color-aware workflow for HDR source-color images.
- Output pixels come directly from the source, never from JPEG analysis previews. Converting an 8-bit source into a 16-bit file does not restore missing source precision.
- Missing color tags are assumed Rec.709 SDR and reported.
- Manifest includes frame IDs, timestamps, scores, source/output color and actual selection count.
- Optional scores.html, scores.json and scores.csv include all scored candidates, native focus evidence, recognized wording, text readiness, occurrence coverage, reasons and duplicate decisions. Keep previews with the HTML report.

## Legacy compatibility

Legacy always saves every 24th decoded frame plus the literal final frame as PNG in the adjacent stills folder, with the original filenames. Smart export settings are unavailable in this mode. The original script is also preserved unchanged:

    python -m pip install -r requirements-legacy.txt
    python extract_stills.py "clip.mov"

Use a separate environment for the standalone V1 requirements to avoid installing two OpenCV distributions into the V2 environment.

## Development and validation

    python -m pip install -r requirements-dev.txt
    python scripts/setup_models.py
    python -m pytest
    python tests/gui_smoke.py
    python build_app.py
    python scripts/verify_packaged.py --help

See [packaging instructions](docs/PACKAGING.md), [third-party notices](THIRD_PARTY_NOTICES.md), [validation evidence](docs/VALIDATION.md) and [approved plan](V2_PLAN.md). Native Mac builds must be validated on a Mac; Windows results do not establish Mac compatibility. V2 stays separate from main until validation and visual acceptance are complete.
