# Extract Stills V2.0

Approved on October 7, 2026. Implementation is deferred until October 8, 2026.

## Summary and repository

Build an offline, scene-aware extractor that becomes the default in the GUI and CLI. Preserve the current extractor unchanged as **Legacy mode**.

- Create a standalone checkout at `D:\Filmkraft\extract-stills`, connected to [the requested repository](https://github.com/haroldfandino/extract-stills). Leave the existing `postproduction-tools` checkout untouched.
- Publish the unchanged extractor and its requirements on `main` as the V1 baseline, with usage documentation and a `v1.0.0` tag.
- Develop V2 on `codex/v2.0`, including the new description as project documentation. Keep V2 separate until validation and visual acceptance are complete.

## Selection engine

- Use FFmpeg/FFprobe for probing, decoding, and final extraction. Retain original frame indexes and presentation timestamps so variable frame rates remain accurate.
- Scan the whole video at reduced resolution. Detect shots with [PySceneDetect's adaptive detector](https://www.scenedetect.com/docs/0.6.7/api/detectors.html), score candidates at four frames per second, and refine promising moments at native frame rate. Include candidates from shots shorter than the sampling interval.
- Use duration targets of **6 stills for 6 seconds, 10 for 15 seconds, 15 for 30 seconds, and 20 above 30 seconds**. Interpolate between anchors; below six seconds, target one per second. Allow one-frame duration tolerance around the anchors.
- Increase the automatic target when needed to cover every worthwhile, distinct scene. Select one strong representative per eligible scene, then fill remaining slots across longer scenes and the timeline. Export fewer when good, unique frames are unavailable. An explicit count override limits the output.
- Score overall and regional sharpness, exposure/clipping, transition artifacts, text stability, and facial expressions. Use [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) for open-eye preferences and a small smile bonus. Missing or unreliable face detections contribute neutrally.
- Use a local PP-OCR mobile text detector to track text position, edges, and contrast across adjacent frames. Prefer settled text; describe this as a stability estimate, since original opacity cannot be recovered reliably from flattened video.
- Remove exact duplicates using native decoded-pixel comparisons. Find possible near-duplicates with perceptual hashes, then confirm with regional image differences so changed titles, expressions, and poses remain distinct.
- Reserve the clearest stable frame from the final nonblank shot as the end card, counted within the selection and exported last. If no suitable ending exists, report that limitation.

## GUI, CLI, and reports

Create a Qt-independent core with separate analysis, selection, and export operations shared by both interfaces.

- Build a PySide6 GUI using Technical QC's visual style: light/dark themes, drag-and-drop, file/folder selection, batch progress, cancellation, and per-video results.
- Default to **Smart extraction** with automatic export. Offer **Review before export** with scored thumbnails, scene grouping, larger previews, and manual selection.
- Expose **Legacy extraction** in the GUI and CLI. Keep its original behavior: every 24th frame, the literal last frame, PNG output, original filenames, and the adjacent `stills` folder. Disable incompatible smart-mode settings.
- Provide these command interfaces:
  - `extract-stills extract INPUT...` - automatic extraction; smart mode by default.
  - `extract-stills analyze INPUT...` - save candidate analysis and previews for review.
  - `extract-stills export ANALYSIS --frames ...` - export explicit candidate selections after checking source identity.
  - `extract-stills gui` - launch the desktop interface.
- Support mode, count, output location, format, bit depth, color mode, optional recursive folder scanning, reports, and machine-readable JSON results. Continue batches after individual failures and return a failing exit status when any input fails.
- Generate optional HTML, JSON, and CSV reports with candidate timestamps, scene IDs, Low/Mid/High ratings, component scores, selection reasons, duplicate decisions, and count shortfalls.

## Export and packaging

- Default to **8-bit PNG**, at original resolution. Support PNG/TIFF at 8 or 16 bits and JPEG at 8 bits.
- Export directly from the original video; analysis previews must never supply final image pixels.
- Default to full-range, color-managed sRGB, with CPU HDR tone mapping and embedded ICC profiles using [FFmpeg's color filters](https://ffmpeg.org/ffmpeg-filters.html#iccgen). Offer source-color preservation for 16-bit PNG/TIFF, accompanied by source/output color metadata.
- Smart-mode output goes into a separate, collision-safe run folder per video under the adjacent `stills` directory. Save chronologically numbered images and a compact extraction manifest. Detailed score reports remain optional.
- Target **Windows x64 and Apple Silicon macOS 13+**, using Python 3.12 internally. Lock compatible dependencies, starting with PySide6 6.8.3, MediaPipe 0.10.21, OpenCV contrib 4.11, PySceneDetect 0.6.7.1, and ONNX Runtime 1.23.2.
- Build native desktop and console entry points with bundled Python, FFmpeg, models, and license notices. Packaged extraction must work offline from first launch.

## Validation and release

- Use the user-provided videos in `D:\Filmkraft\Tools\Extract_stills\input` for real-video validation, alongside synthetic fixtures. Write test outputs into separate folders and preserve the source videos.
- Verify legacy parity: frame indexes, final-frame handling, filenames, paths, and image output.
- Test scene coverage, short shots, camera movement, blurred frames, blinks, natural expressions, entering/fading text, changing titles, duplicates, static holds, and stable end cards followed by black frames.
- Test variable frame rates, rotation, 23.976 fps durations, corrupt inputs, cancellation, batch failures, output collisions, and GUI/CLI selection parity.
- Verify actual image bit depth, ICC profiles, HDR-to-SDR appearance, and source-color exports. Reject JPEG/16-bit combinations.
- Test native Windows and Mac bundles without installed Python, system FFmpeg, or network access.
- Review representative 6-, 15-, 30-second and longer clips before declaring V2 stable. Merge `codex/v2.0` through a reviewed PR only after these gates pass, then tag `v2.0.0`.

Defaults: CPU processing, automatic selection, nonrecursive folder scanning, natural expressions, and scene coverage that may exceed the duration target. Intel Macs and cloud processing are outside V2 scope.

## Approved refinement: October 8, 2026

- Improve focus accuracy with noise-resistant multiscale measurements on original face/text pixels, continuous candidate rankings and wider native-frame searches.
- Cover complete, readable wording changes within shots using offline recognition and temporal/layout consensus. Avoid typewriter fragments, clipped words, incidental OCR flicker and fragmented detector boxes.
- Keep automatic counts as the default, with the existing duration anchors and explicit count override. Automatic targets may grow to cover additional readable wording; exact duplicate images can cover repeated occurrences without exporting redundant files.
