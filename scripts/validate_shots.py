"""Make a local, source-timed shot-review gallery without uploading acceptance media.

The overview samples the timeline independently of the shot detector. Human
annotations can therefore be compared against a later detector result without
treating that result as ground truth.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stills_tool.core import CREATE_FLAGS, discover_inputs, find_tool, probe_video


def _labelled(image, text, width=256):
    image = image.convert("RGB")
    image.thumbnail((width, 144))
    tile = Image.new("RGB", (width, 168), "#161a20")
    tile.paste(image, ((width - image.width) // 2, 0))
    ImageDraw.Draw(tile).text((6, 147), text, fill="white", font=ImageFont.load_default())
    return tile


def overview(source, destination, interval=0.25):
    destination.mkdir(parents=True, exist_ok=True)
    videos = []
    for path in discover_inputs([source]):
        meta = probe_video(path)
        stride = max(1, round(meta["fps"] * interval))
        folder = destination / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        frames = folder / "timeline"
        frames.mkdir(exist_ok=True)
        command = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(path),
                   "-vf", f"select='not(mod(n,{stride}))',scale=320:-2", "-fps_mode", "vfr",
                   "-q:v", "3", str(frames / "%05d.jpg")]
        subprocess.run(command, check=True, creationflags=CREATE_FLAGS)
        images = sorted(frames.glob("*.jpg"))
        pages = []
        for first in range(0, len(images), 40):
            page = Image.new("RGB", (256 * 5, 168 * 8), "#10141a")
            for j, image in enumerate(images[first:first + 40]):
                index = (first + j) * stride
                stamp = meta["frames"][index]["timestamp"]
                with Image.open(image) as opened:
                    tile = _labelled(opened, f"{stamp:.3f}s | frame {index}")
                page.paste(tile, ((j % 5) * 256, (j // 5) * 168))
            name = f"timeline_{first // 40 + 1:02d}.jpg"
            page.save(folder / name, quality=88)
            pages.append(name)
        videos.append({"name": path.name, "source": str(path.resolve()),
                       "width": meta["width"], "height": meta["height"], "fps": meta["fps"],
                       "duration": meta["duration"], "frame_count": meta["frame_count"],
                       "sample_stride_frames": stride, "sheets": pages})
        print(json.dumps(videos[-1]), flush=True)
        (destination / "overview.json").write_text(json.dumps(videos, indent=2), encoding="utf-8")
    body = []
    for video in videos:
        name = html.escape(video["name"])
        body.append(f"<h2>{name}</h2><p>{video['width']}×{video['height']} · "
                    f"{video['duration']:.3f}s · {video['frame_count']} frames · "
                    "timeline samples independent of detector</p>")
        for sheet in video["sheets"]:
            ref = html.escape(f"{Path(video['name']).stem}/{sheet}")
            body.append(f'<a href="{ref}"><img loading="lazy" src="{ref}" alt="{name} timeline"></a>')
    (destination / "overview.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Independent shot timeline review</title>'
        '<style>body{background:#11151b;color:#eee;font:16px sans-serif;margin:28px}'
        'img{max-width:100%;margin:12px 0}h2{margin-top:48px}a{color:#80bdff}</style>'
        '<h1>Independent shot timeline review</h1>'
        '<p>Frames sampled at fixed timeline intervals. These sheets do not use detector boundaries. '
        'Record physical shot changes separately before comparing detector results.</p>' + ''.join(body),
        encoding="utf-8")


def comparison(analyses_root, destination, annotations=None):
    """Show each selected still with independently decoded before/after boundaries."""
    destination.mkdir(parents=True, exist_ok=True)
    references = {}
    if annotations:
        data = json.loads(annotations.read_text(encoding="utf-8"))
        references = {v["video"]: v for v in data["videos"]}
    paths = ([analyses_root] if analyses_root.is_file()
             else sorted(analyses_root.rglob("analysis.json")))
    summaries, sections = [], []
    for analysis_path in paths:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        source = Path(analysis["source"])
        shots = analysis.get("shots", analysis.get("scenes", []))
        selected = set(analysis["selected_ids"])
        candidates = {c["id"]: c for c in analysis["candidates"]}
        last_frame = analysis["video"]["frame_count"] - 1
        def clean_edge_frames(position):
            shot = shots[position]
            entry = shot.get("boundary", {})
            exit_info = shots[position + 1].get("boundary", {}) if position + 1 < len(shots) else {}
            start, end = shot["start_frame"], shot["end_frame"]
            clamp = lambda n: min(last_frame, max(0, int(n)))
            return [("Before entry transition", clamp(entry.get("start", start) - 1)),
                    ("After entry transition", clamp(entry.get("end", start - 1) + 1)),
                    ("Before exit transition", clamp(exit_info.get("start", end + 1) - 1)),
                    ("After exit transition", clamp(exit_info.get("end", end) + 1))]
        folder = destination / source.stem
        folder.mkdir(exist_ok=True)
        indexes = set()
        for position, shot in enumerate(shots):
            indexes.update(n for _, n in clean_edge_frames(position))
        selection = "+".join(f"eq(n,{n})" for n in sorted(indexes))
        command = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(source),
                   "-vf", f"select='{selection}',scale=480:-2", "-fps_mode", "vfr",
                   "-q:v", "3", str(folder / "boundary_%05d.jpg")]
        subprocess.run(command, check=True, creationflags=CREATE_FLAGS)
        boundary = dict(zip(sorted(indexes), sorted(folder.glob("boundary_*.jpg"))))
        exported = {}
        manifest = analysis_path.parent / "manifest.json"
        if manifest.is_file():
            exported = {f["frame_index"]: f["path"] for f in
                        json.loads(manifest.read_text(encoding="utf-8")).get("files", [])}
        cards, counts = [], []
        for position, shot in enumerate(shots):
            start, end = shot["start_frame"], shot["end_frame"]
            picks = sorted(n for n in selected if start <= candidates[n]["frame_index"] <= end)
            counts.append(len(picks))
            frames = clean_edge_frames(position)
            images = []
            for label, frame in frames:
                ref = html.escape(os.path.relpath(boundary[frame], destination).replace("\\", "/"))
                stamp = analysis["video"]["frames"][frame]["timestamp"]
                images.append(f'<figure><a href="{ref}"><img loading="lazy" src="{ref}"></a>'
                              f'<figcaption>{label} · frame {frame} · {stamp:.3f}s</figcaption></figure>')
            chosen = []
            for index in picks:
                candidate = candidates[index]
                ref_path = Path(analysis["analysis_dir"]) / candidate["preview"]
                ref = html.escape(os.path.relpath(ref_path, destination).replace("\\", "/"))
                original = html.escape(os.path.relpath(exported.get(index, ref_path), destination).replace("\\", "/"))
                chosen.append(f'<a href="{original}"><img loading="lazy" src="{ref}"></a>'
                              f'<p>Selected frame {candidate["frame_index"]} · '
                              f'{candidate["timestamp"]:.3f}s · score {candidate["score"]:.2f}</p>')
            detail = html.escape(json.dumps(shot.get("boundary", {})))
            edge = shot.get("boundary", {})
            boundary_time = analysis["video"]["frames"][start]["timestamp"]
            edge_text = (f'Entry: {html.escape(edge.get("kind", "cut"))} midpoint/frame {start} '
                         f'at {boundary_time:.3f}s'
                         + (f' · transition frames {edge["start"]}–{edge["end"]}'
                            if "start" in edge and "end" in edge else '')
                         if position else 'Start of video')
            cards.append(f'<article><h3>Shot {shot["id"] + 1} · frames {start}–{end} · '
                         f'{len(picks)} selected</h3><p>{edge_text}</p><div class="chosen">{"".join(chosen)}</div>'
                         f'<div class="boundaries">{"".join(images)}</div><details>'
                         f'<summary>Boundary evidence</summary><pre>{detail}</pre></details></article>')
        reference = references.get(source.name, {})
        starts = [s["start_frame"] for s in shots[1:]]
        hits = [{**cut, "matched": any(abs(cut["frame"] - n) <= 1 for n in starts)}
                for cut in reference.get("hard_cuts", [])]
        transitions = [{**cut, "matched": any(cut["start_frame"] <= n <= cut["end_frame"] for n in starts)}
                       for cut in reference.get("authored_transitions", [])]
        summary = {"video": source.name, "analysis": str(analysis_path.resolve()),
                   "shots": len(shots), "selected": len(selected),
                   "exactly_one_per_shot": all(n == 1 for n in counts),
                   "selected_per_shot": counts, "human_hard_cut_matches": hits,
                   "human_transition_matches": transitions}
        blind = [x for x in hits if x.get("review_phase") != "post_detection_adjudication"]
        adjudicated = [x for x in hits if x.get("review_phase") == "post_detection_adjudication"]
        summary["blind_hard_cut_matches"] = {"matched": sum(x["matched"] for x in blind), "total": len(blind)}
        summary["post_detection_adjudications"] = adjudicated
        summaries.append(summary)
        reference_text = (f'<p>Blind human hard-cut matches: {sum(c["matched"] for c in blind)}/{len(blind)}; '
                          f'authored transition matches: {sum(c["matched"] for c in transitions)}/{len(transitions)}. '
                          'UI scroll labels are semantically less certain than hard cuts.</p>'
                          + (f'<p>Extra detector proposals later verified at native resolution: '
                             f'{sum(c["matched"] for c in adjudicated)}/{len(adjudicated)}. '
                             'These later adjudications are separate from the blind reference.</p>' if adjudicated else '')
                          if reference else '')
        sections.append(f'<h2>{html.escape(source.name)}</h2><p>{len(shots)} shots · '
                        f'{len(selected)} selected stills · exactly one per shot: '
                        f'{summary["exactly_one_per_shot"]}</p>{reference_text}{"".join(cards)}')
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (destination / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Shot-by-shot selected still review</title>'
        '<style>body{background:#11151b;color:#eee;font:16px sans-serif;margin:28px}'
        'article{background:#202630;padding:18px;margin:22px 0}a{color:#80bdff}'
        '.chosen img{max-width:760px;width:100%;max-height:440px;object-fit:contain}'
        '.boundaries{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}'
        'figure{margin:0}figure img{width:100%}figcaption{font-size:13px}'
        'pre{white-space:pre-wrap}h2{margin-top:60px}</style>'
        '<h1>Shot-by-shot selected still review</h1><p>Each selected still is followed by clean source '
        'frames before and after the entry and exit transitions. Hard cuts use adjacent frames. '
        'Click a selected still to open its original-resolution export when available.</p>'
        + ''.join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation/shot_review"))
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--compare", action="store_true", help="Input is an analysis.json or folder of analyses")
    parser.add_argument("--annotations", type=Path, help="Independent human annotation JSON for cut matching")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.compare:
        comparison(args.input, args.output, args.annotations)
    else:
        overview(args.input, args.output, args.interval)


if __name__ == "__main__":
    main()
