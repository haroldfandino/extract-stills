# V2 beta validation

Validation date: October 8, 2026. This is a development build, not the stable V2 release.

## Completed checks

- 50 source integration and quality tests passed on Windows x64 with Python 3.12 and the locked dependencies.
- 10 offscreen GUI tests passed in light and dark themes, including review selection, batch failure continuation, cancellation and safe window shutdown.
- Real FFmpeg fixtures cover variable frame rates/nonzero start PTS, rotation, source-frame identity, scene coverage, short shots, static holds and legacy parity.
- Export fixtures verify PNG/TIFF/JPEG bit depth, embedded sRGB ICC profiles, high-precision source-color pixels and HDR/PQ tone mapping.
- Integrity and failure cases cover missing/changed inputs, unavailable models, explicit frame selections, output collisions and cancelled subprocess cleanup.
- The original legacy implementation is unchanged apart from Git's line-ending normalization. Legacy output matches the original decoder, filenames, stride and final-frame behavior.
- Local models are checksum-verified. Bounded Windows vision-runtime monitoring found no outbound TCP connections; missing models fail locally instead of downloading at runtime.
- Final Windows beta 2 GUI/CLI bundles passed bundled FFmpeg/model lookup from a temporary directory with a restricted PATH, original-resolution PNG8/ICC and TIFF16/ICC output, GUI startup and the frozen legacy worker. A supplied HD clip also exported through the frozen TIFF16 path with actual uint16 RGB samples and unchanged source.
- Native CI run 37800366278 passed Windows x64 and Apple Silicon macOS 15.7.9 source tests, GUI startup, native builds and frozen PNG/legacy checks. Its Mac binary audit reports a highest dependency minimum of macOS 15.0; this is not a macOS 13-compatible build.

## Real-video acceptance

All eight user-supplied Rec.709 SDR clips passed the final beta 2 run: four 15-second and four 30-second videos, with HD and 4K sources at 23.976 fps. The run produced 103 original-resolution PNG stills in 690.81 seconds on the local Windows workstation. Every source's SHA-256, size and modification time remained unchanged. Sources and stills are excluded from Git and CI.

| Fixture | Duration / resolution | Selected / effective target | Detected scenes | Seconds |
| --- | --- | --- | --- | --- |
| A | 15 seconds / HD | 10 / 10 | 5 | 32.72 |
| B | 30 seconds / HD | 15 / 15 | 11 | 70.23 |
| C | 15 seconds / 4K | 10 / 10 | 13 | 75.67 |
| D | 30 seconds / 4K | 15 / 15 | 17 | 129.89 |
| E | 15 seconds / 4K | 10 / 10 | 12 | 69.74 |
| F | 30 seconds / 4K | 18 / 18 | 20 | 116.80 |
| G | 15 seconds / 4K | 10 / 10 | 12 | 72.62 |
| H | 30 seconds / 4K | 15 / 15 | 18 | 123.14 |

Detected scenes include transitions and duplicate content; only worthwhile unique scenes affect the target. Visual review led to regression-tested rejection of empty gradients and short unsettled text transitions. Source-relative sharpness alone must not promote a uniformly blurred scene.

Reproduce on the local workstation:

    python scripts/validate_samples.py "D:/Filmkraft/Tools/Extract_stills/input" --output validation/samples_final

The script saves selected count, effective target, detected scenes, ending time, source integrity, timings and output paths in validation/samples_final/summary.json. HTML score reports and extracted stills are retained locally for visual review. The local gallery is validation/samples_final/review.html.

## Native applications and release gates

Native CI runs on Windows x64 and Apple Silicon macOS, using synthetic fixtures. V2 branch source pushes build and verify both app entry points and produce temporary development archives.

A restricted PATH/proxy smoke is not equivalent to disabling networking on a clean machine. Full offline clean-machine testing, macOS 13-compatible dependencies and native validation, application signing/redistribution materials and user visual acceptance remain stable-release gates. The current Mac beta requires macOS 15 or newer.

Keep V2 on codex/v2.0 until these release gates are satisfied; do not merge it into main or tag v2.0.0 yet.
