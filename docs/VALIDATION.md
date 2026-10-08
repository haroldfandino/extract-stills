# V2 beta validation

Validation date: October 8, 2026. This is a development build, not the stable V2 release.

## Completed checks

- 46 source integration and quality tests passed on Windows x64 with Python 3.12 and the locked dependencies.
- 10 offscreen GUI tests passed in light and dark themes, including review selection, batch failure continuation, cancellation and safe window shutdown.
- Real FFmpeg fixtures cover variable frame rates/nonzero start PTS, rotation, source-frame identity, scene coverage, short shots, static holds and legacy parity.
- Export fixtures verify PNG/TIFF/JPEG bit depth, embedded sRGB ICC profiles, high-precision source-color pixels and HDR/PQ tone mapping.
- Integrity and failure cases cover missing/changed inputs, unavailable models, explicit frame selections, output collisions and cancelled subprocess cleanup.
- The original legacy implementation is unchanged apart from Git's line-ending normalization. Legacy output matches the original decoder, filenames, stride and final-frame behavior.
- Local models are checksum-verified. Bounded Windows vision-runtime monitoring found no outbound TCP connections; missing models fail locally instead of downloading at runtime.
- An initial native Windows GUI/CLI build passed bundled FFmpeg/model lookup from a temporary directory with a restricted PATH, original-resolution PNG8/ICC output, GUI startup and the frozen legacy worker.

## Real-video acceptance

Eight user-supplied Rec.709 SDR clips are being validated locally: four 15-second and four 30-second videos, with HD and 4K sources at 23.976 fps. Sources and stills are excluded from Git and CI.

Reproduce on the local workstation:

    python scripts/validate_samples.py "D:/Filmkraft/Tools/Extract_stills/input" --output validation/samples

The script saves selected count, effective target, detected scenes, ending time, source integrity, timings and output paths in validation/samples/summary.json. HTML score reports and extracted stills are retained locally for visual review.

## Native applications and release gates

Native CI runs on Windows x64 and Apple Silicon macOS, using synthetic fixtures. V2 branch source pushes build and verify both app entry points and produce temporary development archives.

The final Windows rebuild, eight-video batch and native CI results are recorded after completion. A restricted PATH/proxy smoke is not equivalent to disabling networking on a clean machine. Full offline clean-machine testing, actual macOS 13 compatibility, application signing/redistribution materials and user visual acceptance remain stable-release gates.

Keep V2 on codex/v2.0 until these release gates are satisfied; do not merge it into main or tag v2.0.0 yet.
