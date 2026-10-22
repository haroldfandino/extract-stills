# V2 beta 4 validation

Validation date: October 22, 2026. Application source: adfb343 on codex/v2.0, version 2.0.0b4. This is a development beta. Main remains the V1 baseline. [Beta 3 evidence](VALIDATION-BETA3.md) records the earlier duration/wording coverage behavior; Beta 4 defaults to one still per detected shot.

## Source and GUI checks

- 240 engine, shot detection, OCR, sharpness and integration tests passed on Windows x64 with Python 3.12 and locked dependencies. One optional face acceptance test was skipped because its older private video fixture is unavailable.
- 13 offscreen GUI checks passed, covering automatic defaults, shot grouping, opening on the chosen candidate, manual selection/coverage, saved scene-schema compatibility, themes, cancellation, shutdown and Legacy mode.
- 18 detector tests cover hard cuts, one/two-frame inserts, static branded canvases with portrait-inset cuts, same-palette edits, pans, zooms, subtitles, flashes, dissolves, fades, blank entrances/tails and bounded image history.
- Integration tests verify repeated A/B/A shots retain separate stills, short inserts export, count caps report missing coverage, and low-quality shots receive explicit best-available fallbacks.
- Ranking regressions cover blink/transition penalties surviving readability score caps, visible faces, settled UI frames beating scroll blur, native face/text focus and differing wording density.
- Existing original-resolution, source-frame identity, VFR/nonzero PTS, rotation, PNG/TIFF/JPEG, bit-depth/ICC, source-color/HDR, source integrity and unchanged Legacy parity checks pass.

## Six supplied advertisements

All six supplied videos were analyzed locally with the frozen shot detector: two HD and four 4K clips at 23.976 fps, five approximately 30 seconds and one approximately 15 seconds. Each decoded source frame was screened; shortlisted frames were inspected at native resolution. Final ranking was replayed on those saved native measurements after visual review found score-cap and scroll-blur selection issues, then the chosen original source frames were exported again. This replay is not a fresh extraction benchmark. A fresh final frozen-application run is recorded separately below.

| Local fixture | Duration / resolution | Detected shots | Exported stills | Best-available shots flagged |
| --- | --- | ---: | ---: | ---: |
| A | 30 seconds / 4K | 15 | 15 | 0 |
| B | 30 seconds / HD | 17 | 17 | 5 |
| C | 30 seconds / HD | 18 | 18 | 2 |
| D | 15 seconds / 4K | 6 | 6 | 0 |
| E | 30 seconds / 4K | 11 | 11 | 2 |
| F | 30 seconds / 4K | 10 | 10 | 0 |
| Total | | 77 | 77 | 9 |

Every detected shot has exactly one selected frame. Source SHA-256, byte size and modification time remain unchanged. Source media, frame measurements, annotation files and exports stay local and excluded from Git/CI.

Review labels were recorded from fixed timeline samples and dense frame sheets before comparison with detector output. All 44 initially marked definitive hard cuts match within one native source frame. Two additional detections were subsequently confirmed as real jump cuts by inspecting original-resolution frames; the annotations explicitly distinguish these post-detection adjudications from the initial blind reference set. A matching count alone was not used as proof of detection accuracy.

Strict annotated transition-window matches are 23 of 27. Two Birthday UI changes are found with boundaries two/three frames after the labeled transition interval. Hotel and Jam each keep a continuous story-to-memory UI scroll in the same detected shot. These are authored composition changes, with less objective shot semantics than physical camera edits. They remain useful targets for user feedback; this result does not claim frame-accurate parity with an editing application's detector.

Visual review inspected the 77 initial selections and all 17 changed final recommendations. The final selections no longer show the identified blink, whole-frame scroll blur or incoming-overlay defects. Nine shots retain conservative quality/readability warnings; tiny text, intentional fades and partial faces still need judgment. Selection scores and OCR confidence do not certify every visible word or original text opacity.

Final local review: `validation/beta4_review/review.html`. It shows every chosen still, clean transition endpoints, boundary evidence and links to the original-resolution exported PNGs; all 770 image/export links were checked. Reference comparisons are in `validation/beta4_review/summary.json`. Final analyses and exports are under `validation/beta4_final_ready`; initial fresh analyses are kept separately under `validation/shot_samples_beta4_final`.

Reproduce with source installation and local assets:

    python -m pytest -q
    python tests/gui_smoke.py
    python scripts/validate_samples.py "C:/Users/harol/Desktop/Extract-stills-test" --output validation/beta4_fresh
    python scripts/validate_shots.py validation/beta4_fresh --compare --annotations validation/shot_review/manual_annotations.json --output validation/beta4_review_fresh

The private annotation JSON is local validation evidence, not a repository test fixture. See [shot detection and human review](SHOT-DETECTION.md) to create independent annotations for other footage.

## Native applications

Windows GUI: `dist/beta4/gui/Extract Stills/Extract Stills.exe`. Console: `dist/beta4/cli/extract-stills/extract-stills.exe`. Preserve the complete app folders. Earlier beta distributions remain separate.

The final frozen Windows GUI/CLI passed synthetic checks from a temporary directory with restricted PATH and blocked proxy settings: bundled model hashes/notices, original-resolution PNG8/ICC and TIFF16/ICC, native text focus, three A/B/A shots with repeated pixels retained, no output-count increase for an in-shot caption change, GUI startup and the original Legacy worker. Application and bundled package metadata both report 2.0.0b4. Evidence: `validation/beta4_packaged-verification.json`.

A fresh full extraction through that final frozen console passed on the 30-second HD advertisement in 152.31 seconds. It found 17 shots and exported 17 original-resolution stills, matching every selected source-frame ID in the reviewed replay. The corrected fallback frames retain open eyes/settled imagery, the saved analysis includes uncapped quality evidence, and the source fingerprint is unchanged. Evidence: `validation/beta4_frozen_Kids/verification.json`. This checks the complete final pipeline independently of the cached ranking replay.

Distribution: `release/Extract-Stills-Windows-x64-V2-beta4.zip`, with an adjacent SHA-256 file. The archive contains complete GUI/CLI app folders, documentation and synthetic verification results; supplied acceptance media is excluded.

The native CI workflow also validates Windows x64 and Apple Silicon macOS 15. Local Windows checks do not establish new Mac compatibility. The current dependency set requires macOS 15; macOS 13 compatibility remains a stable-release target. Keep V2 on its branch until visual acceptance and the existing native/offline/distribution release gates are complete.
