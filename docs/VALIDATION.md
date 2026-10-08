# V2 beta 3 validation

Validation date: October 8, 2026. Application source: 0e33ee5 on codex/v2.0. This is a development beta, not the stable V2 release.

## Completed checks

- 213 engine, OCR, sharpness and integration tests passed on Windows x64 with Python 3.12 and locked dependencies.
- 11 offscreen GUI tests passed, including automatic Smart defaults, review selection, light/dark themes, cancellation, safe shutdown and Legacy mode.
- Native focus tests cover defocus, directional motion blur, noise, fine text, subject detail against busy backgrounds and rounded-score ties.
- Text tests cover real one-word/digit changes, changing captions inside a continuous shot, typing/occlusion/scroll fragments, field-order jitter, repeated text and native color-pixel verification. Mixed-color and isoluminant text changes remain distinct; uncertain graphic tokens cannot silently erase supported numeric changes.
- Real FFmpeg fixtures cover variable frame rates/nonzero PTS, rotation, source-frame identity, short shots, static holds, source integrity and legacy parity.
- Export fixtures verify PNG/TIFF/JPEG bit depth, sRGB ICC profiles, source-color precision and HDR/PQ tone mapping.
- Four local vision assets, including the Latin OCR model and its exact dictionary, are checksum-verified. Runtime never downloads models.
- Replacement frozen Windows GUI/CLI checks pass from a temporary working directory with restricted PATH/proxy settings: PNG8/ICC, TIFF16/ICC, original resolution, offline asset lookup, GUI startup and the original Legacy worker. The frozen OCR fixture covers ALPHA/BRAVO/CLEAR with three readable states, automatic count growth from two to three and native text focus.
- Native Windows x64 and Apple Silicon macOS 15 CI passed source, GUI startup, builds and frozen exports. Final application-source run: 37840694071. The current Mac bundle requires macOS 15 or newer.

## Supplied-video review

Eight supplied Rec.709 SDR clips were analyzed at native resolution: HD and 4K, 15/30 seconds, 23.976 fps. Final consensus was replayed against lossless native color regions, enriched with character confidence, then scored, deduplicated and exported again. This reuses video analysis; the fresh frozen-app run below validates the final pipeline independently.

The final replay produced 186 original-resolution PNG stills and covered all 165 eligible recognized wording occurrences. Every source SHA-256, size and modification time remained unchanged. Media, patches and outputs remain local and excluded from Git/CI. The initial complete eight-video analysis took 2,126.47 seconds; final replay/export took 609.41 CPU-process seconds across parallel jobs. These replay timings are not fresh extraction benchmarks.

| Fixture | Duration / resolution | Selected / automatic target | Eligible wording occurrences covered |
| --- | --- | --- | --- |
| A | 15 seconds / HD | 10 / 10 | 6 / 6 |
| B | 30 seconds / HD | 23 / 23 | 22 / 22 |
| C | 15 seconds / 4K | 17 / 17 | 14 / 14 |
| D | 30 seconds / 4K | 32 / 32 | 28 / 28 |
| E | 15 seconds / 4K | 15 / 15 | 13 / 13 |
| F | 30 seconds / 4K | 44 / 44 | 41 / 41 |
| G | 15 seconds / 4K | 19 / 19 | 17 / 17 |
| H | 30 seconds / 4K | 26 / 26 | 24 / 24 |

Automatic count remains the default. Distinct readable wording can increase it beyond the duration base; an explicit count remains a cap. No export warnings or eligible-occurrence omissions were reported. Recognized eligibility is a heuristic estimate, not a human annotation of every visible word.

Review rejected the occluded Trusted caption, a five-frame growing/exiting phrase and blouse lace misread as zeros. The final ending captions remain selected. Some small logos, punctuation/cursors, product microtext and clipped UI preview labels still produce conservative extra occurrences or uncertain readings. Strict editorial acceptance of all visible lettering needs human review; use Review before export and the reports for these cases. Motion/readability estimates and OCR confidence do not prove original opacity, spelling or semantic completeness.

Local review gallery: validation/beta3_final/review.html. Machine-readable results: validation/beta3_final/summary.json. The replacement frozen Fatty15 30-second run is retained separately in validation/beta3_frozen_Fatty30_final.

A fresh complete extraction through the final frozen Windows CLI passed in 343.14 seconds: 23 PNG images, automatic base 15/effective target 23, all 22 eligible recognized occurrences covered, final frame 704, no warnings and unchanged source SHA-256. It matches the replay selection count. The preceding build took 516.25 seconds on that clip; removing redundant unchanged-word comparisons improved this measured case.

Reproduce fresh source extraction:

    python scripts/validate_samples.py "D:/Filmkraft/Tools/Extract_stills/input" --output validation/beta3_fresh

## Native applications and stable-release gates

Windows GUI: dist/beta3/gui/Extract Stills/Extract Stills.exe. Console: dist/beta3/cli/extract-stills/extract-stills.exe. Distribution: release/Extract-Stills-Windows-x64-V2-beta3.zip, with an adjacent SHA-256 file. Preserve the whole app folder. Prior beta artifacts are retained separately.

Beta 2 produced 103 stills across the same eight videos and passed native Windows/macOS 15 checks. Beta 3 adds more thorough native focus and wording analysis, so fresh processing takes longer. Temporary color-pixel evidence is released after selection to keep production analyses compact.

A restricted PATH/proxy smoke is not equivalent to disabling networking on a clean machine. Full offline clean-machine testing, macOS 13-compatible dependencies/native validation, signing/redistribution materials and user visual acceptance remain stable-release gates. Keep V2 separate from main; do not tag v2.0.0 or merge this draft yet.
