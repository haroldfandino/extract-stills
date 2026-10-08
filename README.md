# Extract Stills

Extract PNG stills from a video every 24 frames, plus the final decoded frame.

## Version 1

Requires Python and OpenCV:

```sh
python -m pip install -r requirements.txt
python extract_stills.py "path/to/video.mp4"
```

Images are saved in a `stills` folder beside the video as `<video>_frame_<index>.png`. Frame indexes are zero-based. The original extractor is preserved unchanged.

## Version 2

The offline, scene-aware GUI and CLI are being developed on `codex/v2.0`. That branch contains the approved `V2_PLAN.md`. Version 2 will be merged here after validation and visual review.
