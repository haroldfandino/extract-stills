# Shot detection and review

Smart extraction defaults to **one best still per detected shot**. A physical camera shot is one continuous view; a cut to a different view starts another shot, including a return to earlier footage. Designed graphic views are assessed in the same way. A caption change, blink, pan or zoom within a view does not by itself request another still. Duration and OCR wording counts do not set the output count.

## Detection and selection

The scanner inspects every decoded source frame. A streaming detector compares small color images, the spatial extent of changes and registered image continuity. It proposes hard cuts, consolidates gradual transitions, and suppresses brief exposure flashes that return to the same view. Short inserts have no fixed minimum shot duration. A short authored scroll or wipe that settles into a different designed view can count as a composition transition; continuous camera movement remains one shot. This distinction is partly semantic and needs review for animated advertisements.

Each shot gets its own shortlist of sharp moments from across its duration. Short shots retain every frame for native analysis. Original-resolution face, text and focus evidence then ranks candidates, with a finer source-frame search around promising moments. Settled imagery outside identified transition intervals is preferred. If no candidate meets the quality gates, the best available frame is retained and flagged for review. Repeated imagery in separate shots receives separate stills.

Exports decode the chosen frames from the original video at source resolution, using the selected output format and color conversion. JPEG analysis previews are never the export source. Analysis records source frame indexes, timestamps, shot ranges, accepted boundary evidence, selected frames and coverage. Frame indexes are zero-based; shot end-frame indexes are inclusive. New analyses use `shots` and `shot_id`; saved `scenes` / `scene_id` analyses remain compatible.

## Human validation

Record reference cuts independently of detector output. Start with a timeline overview:

    python scripts/validate_shots.py "C:/Users/harol/Desktop/Extract-stills-test" --output validation/shot_timeline

Open `overview.html`, then scrub the original video frame by frame around each edit in an editor. Quarter-second overview samples can miss short inserts. Mark the first frame of each new hard-cut shot; record the whole visible interval for a dissolve, fade or authored transition. Distinguish camera edits from graphic composition changes. An annotation file has this form:

```json
{
  "videos": [{
    "video": "clip.mp4",
    "hard_cuts": [{"frame": 24}],
    "authored_transitions": [{"start_frame": 50, "end_frame": 61}]
  }]
}
```

Run extraction, then compare source frames on either side of every detected boundary with the chosen still:

    python scripts/validate_samples.py "C:/Users/harol/Desktop/Extract-stills-test" --output validation/beta4_samples
    python scripts/validate_shots.py validation/beta4_samples --compare --annotations validation/shot_annotations.json --output validation/shot_comparison

The comparison's `review.html` and `summary.json` show per-shot selection and reference matches. Hard cuts match within one source frame; gradual transitions match within the annotated interval. Inspect additional detections as well as missing edits: matching all reference cuts alone does not establish that there are no false cuts. Check that each real shot has one representative, and assess sharpness, eyes, readable complete text and transition contamination at full output resolution. Record disagreements by video name and source frame index for reproducible feedback.

## Practical limits

Boundary confidence and quality scores are heuristics. Dissolves, fades and wipes receive a representative midpoint boundary; this does not guarantee the frame accuracy or editing semantics of DaVinci Resolve or Premiere. Rapid motion, similar compositions, flashes, animated layouts and very short shots remain difficult cases. OCR estimates do not guarantee spelling or complete wording. Manual review can replace a recommendation or select extra frames. An explicit count is a cap and can leave shots uncovered; Automatic preserves the one-per-shot default. Sample-specific results belong in [validation evidence](VALIDATION.md).
