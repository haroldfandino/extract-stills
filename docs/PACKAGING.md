# Native application packaging

Build on the target platform with Python 3.12: Windows x64 or Apple Silicon macOS. PyInstaller bundles Python, the Python dependencies, local models, FFmpeg, FFprobe, and notices. Both applications are one-folder builds; keep the complete folders together when moving them.

## Build and verify

Create a virtual environment, install `requirements-dev.txt`, and fetch the models using the documented project setup. Put native `ffmpeg` and `ffprobe` in `ffmpeg/` or `ffmpeg/bin/`; the helper falls back to PATH at build time. All four assets must match `models/manifest.json`: face landmarker, text detector, Latin text recognizer, and its exact `.yml` dictionary/configuration. The entire model directory is bundled, including that dictionary; runtime recognition has no download step or YAML package requirement. FFmpeg must supply `zscale`, `tonemap`, and `iccgen`; nonfree builds are rejected.

```powershell
.venv\Scripts\python.exe build_app.py --dry-run
.venv\Scripts\python.exe build_app.py
.venv\Scripts\python.exe scripts\verify_packaged.py --cli "dist\cli\extract-stills\extract-stills.exe" --gui "dist\gui\Extract Stills\Extract Stills.exe"
```

```sh
.venv/bin/python build_app.py --dry-run
.venv/bin/python build_app.py
.venv/bin/python scripts/verify_packaged.py --cli dist/cli/extract-stills/extract-stills --gui "dist/gui/Extract Stills.app/Contents/MacOS/Extract Stills"
```

Windows outputs `dist/gui/Extract Stills/Extract Stills.exe` and `dist/cli/extract-stills/extract-stills.exe`. Mac outputs `dist/gui/Extract Stills.app` and `dist/cli/extract-stills/extract-stills`. Distinct directories prevent the console and GUI builds from overwriting one another. `--target cli` or `--target gui` builds one entry point. Icons are taken from `assets/icon.ico` or `assets/icon.icns` when present.

Use `--dist-dir dist/beta3` to preserve an earlier beta for comparison. Its entry points live under `dist/beta3/gui` and `dist/beta3/cli`. Keep the earlier Beta 2 ZIP and save Beta 3 separately as `release/Extract-Stills-Windows-x64-V2-beta3.zip`. Package collection discovers all `stills_tool` modules, including sharpness, OCR recognition, and temporal text-state tracking, automatically.

The verifier generates its own small FFV1 test clips, runs the frozen console from a temporary working directory, restricts PATH, checks model hashes and bundled notices, and verifies original resolution, 8-bit PNG, 16-bit RGB TIFF, embedded ICC profiles, and unchanged input. A second clip holds `ALPHA`, `BRAVO`, and `CLEAR` for ten frames each against the same background; its frozen analysis must read all three complete captions, record their coverage, grow the duration target from two to three images, and produce native text-region focus evidence. The verifier's host uses the pinned Pillow embedded font to generate pixels, while the frozen application supplies OCR and selection. It needs neither system fonts nor FFmpeg's optional `drawtext` filter. An optional GUI smoke test runs Qt offscreen and checks the frozen desktop's legacy worker. Proxy settings discourage network access; this does not constitute network isolation. Repeat extraction with networking disabled on a clean machine without Python or system FFmpeg before release.

## macOS support and signing

Build with an arm64 Python interpreter, native arm64 dependencies, and native FFmpeg. The `.app` declares a minimum macOS version of 13.0 and the helper sets the deployment target. Those settings do not lower the minimum version of an already compiled dependency. On Mac the verifier audits bundled Mach-O headers, records their declared minimum OS versions, and lists dependencies requiring a version above 13.0. Verify every native dependency and test on actual macOS 13 before claiming macOS 13 compatibility. A newer GitHub runner tests that runner's OS only; a passing beta artifact may still require newer macOS dependencies until that release gate is resolved.

An unsigned developer build receives PyInstaller's ad-hoc signature, which is insufficient for a normal trusted download experience. Gatekeeper may require the user to explicitly approve opening it. For public distribution use `--codesign-identity "Developer ID Application: ..."`, notarize the finished archive, and staple the notarization ticket. Signing credentials and notarization belong to a separate release process. Windows signing is likewise a separate release step.

## Third-party materials and release artifacts

The source repository contains application source and model provenance. Development binaries and test outputs stay out of Git. The helper records the exact FFmpeg/FFprobe version, configuration, license output, binary hashes, model manifest, build platform, installed package versions, and available package license texts under each application's `licenses/` resources. It also collects supplied license/source notices from the project's `licenses/`, `models/licenses/`, and FFmpeg directory.

Inspect the recorded FFmpeg configuration for the actual license of the binary being shipped. GPL-enabled FFmpeg remains GPL; the application's MIT license does not relicense it. Publish the complete corresponding FFmpeg source, patches, dependency sources as required, and build instructions for those exact binaries alongside releases. A generic upstream source link or `ffmpeg -L` output is insufficient by itself. Keep all model licenses and dependency notices with the bundle. Use the [FFmpeg licensing page](https://www.ffmpeg.org/legal.html) and the included license texts when preparing redistribution materials.

The native CI workflow runs source tests on Windows x64 and Apple Silicon macOS. Code/configuration pushes to `codex/v2.0` also build both entry points and test the frozen applications with synthetic media; Markdown-only pushes skip the workflow. Pull requests run source checks only. A manual workflow run can opt into packaging with `package=true`. Successful package checks save development ZIPs as GitHub Actions artifacts for 14 days, including `packaged-verification.json`; these are beta test downloads, without a published release or signing/notarization promise. Mac ZIPs preserve application symlinks and executable permissions. User media and local validation output are never uploaded. Release only after native checks, original image resolution/bit-depth/color verification, offline checks, GUI review, and source redistribution materials are complete.

References: [PyInstaller native builds](https://pyinstaller.org/en/stable/usage.html), [PyInstaller macOS signing](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing), [GitHub runner architectures](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
