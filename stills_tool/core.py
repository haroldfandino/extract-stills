"""Shared engine. Source frame identity is independent of analysis resolution."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from .types import CancelledError, Options
from .sharpness import measure_focus

EXTENSIONS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".webm"}
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _check(cancel):
    if cancel and cancel():
        raise CancelledError("Operation cancelled.")


def _progress(callback, stage, done, total):
    if callback:
        callback(stage, done, total)


def find_tool(name):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    suffix = ".exe" if os.name == "nt" else ""
    for directory in [root / "binaries", root / "ffmpeg"]:
        path = directory / (name + suffix)
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} is unavailable. Install FFmpeg or use the bundled desktop app.")


def _run(args, cancel=None):
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=CREATE_FLAGS)
    try:
        while True:
            _check(cancel)
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(stderr.decode("utf-8", "replace")[-3500:] or
                               f"Command failed with exit status {process.returncode}.")
        return stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def discover_inputs(paths, recursive=False):
    files = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            files.extend(p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    return sorted(set(files), key=lambda p: str(p).casefold())


def fingerprint(path, cancel=None):
    stat = Path(path).stat()
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            _check(cancel)
            digest.update(chunk)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def probe_video(path, cancel=None):
    raw = _run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_frames", "-show_streams", "-show_format", "-show_entries",
                "frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration_time,width,height:"
                "stream=width,height,pix_fmt,avg_frame_rate,time_base,color_space,color_transfer,"
                "color_primaries,color_range,nb_frames,sample_aspect_ratio:stream_side_data=rotation:"
                "format=duration", "-of", "json", str(path)], cancel)
    data = json.loads(raw)
    if not data.get("streams") or not data.get("frames"):
        raise ValueError("The file has no decodable video frames.")
    stream = data["streams"][0]
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0
    if fps <= 0:
        times = [float(f["best_effort_timestamp_time"]) for f in data["frames"]
                 if "best_effort_timestamp_time" in f]
        deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
        fps = 1 / float(np.median(deltas)) if deltas else 24.0
    frames = []
    for i, frame in enumerate(data["frames"]):
        if (frame.get("width"), frame.get("height")) != (stream["width"], stream["height"]):
            raise ValueError("Videos whose frame dimensions change are unsupported.")
        if "best_effort_timestamp" not in frame or "best_effort_timestamp_time" not in frame:
            raise ValueError("A decoded video frame is missing its source timestamp.")
        frames.append({"pts": int(frame["best_effort_timestamp"]),
                       "source_time": float(frame["best_effort_timestamp_time"]),
                       "duration": float(frame.get("pkt_duration_time", 0)) or 1 / fps})
    origin = frames[0]["source_time"]
    for frame in frames:
        frame["timestamp"] = frame["source_time"] - origin
    if any(b["source_time"] <= a["source_time"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Video frame timestamps are not strictly increasing.")
    declared = stream.get("nb_frames")
    if declared and declared != "N/A" and int(declared) != len(frames):
        raise ValueError("Decoded frame count differs from declared video frame count.")
    rotation = int(next((d["rotation"] for d in stream.get("side_data_list", [])
                         if "rotation" in d), 0)) % 360
    width, height = stream["width"], stream["height"]
    if rotation in {90, 270}:
        width, height = height, width
    return {**stream, "width": width, "height": height, "rotation": rotation, "fps": fps,
            "frame_count": len(frames), "duration": frames[-1]["timestamp"] + frames[-1]["duration"],
            "frames": frames, "first_source_time": origin}


def duration_target(duration, fps=24):
    for anchor in (6, 15, 30):
        if abs(duration - anchor) <= 1 / fps + 1e-6:
            duration = anchor
            break
    if duration <= 6:
        return max(1, math.ceil(duration))
    if duration <= 15:
        return math.floor(6 + (duration - 6) * 4 / 9 + 0.5)
    if duration <= 30:
        return math.floor(10 + (duration - 15) / 3 + 0.5)
    return 20


def color_filter(video, mode="srgb"):
    primaries = video.get("color_primaries", "bt709")
    transfer = video.get("color_transfer", "bt709")
    matrix = video.get("color_space", "bt709")
    primaries = "bt709" if primaries in {"unknown", "unspecified", ""} else primaries
    transfer = "bt709" if transfer in {"unknown", "unspecified", ""} else transfer
    matrix = "gbr" if video.get("pix_fmt", "").startswith(("rgb", "gbr", "bgr")) else matrix
    matrix = "bt709" if matrix in {"unknown", "unspecified", ""} else matrix
    input_range = "full" if video.get("color_range") == "pc" or matrix == "gbr" else "limited"
    first = f"zscale=pin={primaries}:tin={transfer}:min={matrix}:rin={input_range}"
    if mode == "source":
        return first + f":p={primaries}:t={transfer}:m=gbr:r=full,format=gbrp16le"
    if transfer in {"smpte2084", "arib-std-b67"}:
        return (first + ":t=linear:npl=100,format=gbrpf32le,"
                "tonemap=tonemap=mobius:param=0.3:desat=0:peak=10,"
                "zscale=p=bt709:t=iec61966-2-1:m=gbr:r=full")
    return first + ":p=bt709:t=iec61966-2-1:m=gbr:r=full,format=gbrpf32le"


def _filter_file(directory, name, value):
    path = Path(directory) / (name + ".filter")
    path.write_text(value, encoding="utf-8")
    return ["-filter_script:v", str(path)]


def _make_run_dir(source, output):
    base = Path(output) if output else Path(source).parent / "stills"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(source).stem)
    base = base.expanduser().resolve() / stem
    base.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    for number in range(10000):
        path = base / (name if not number else f"{name}_{number:02d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate a unique output directory.")


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.6)
    return float(cv2.Laplacian(smooth, cv2.CV_32F).var())


def _cheap_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Screening operates at one consistent scale; final ranking uses original
    # face/text pixels and scene tiles, never lossy JPEG previews.
    focus_frame = cv2.resize(frame, (max(8, frame.shape[1] // 2), max(8, frame.shape[0] // 2)),
                             interpolation=cv2.INTER_AREA)
    focus = measure_focus(focus_frame)
    return {"sharpness": sharpness(frame), "mean_luma": float(gray.mean()),
            "dark_fraction": float(np.mean(gray < 5)), "clipped_fraction": float(np.mean(gray > 250)),
            "contrast": float(gray.std()), "edge_fraction": float(np.mean(cv2.Canny(gray, 60, 120) > 0)),
            "screen_focus": focus["focus"], "screen_noise": focus["noise"]}


def _write_preview(path, frame):
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise IOError(f"Cannot write preview: {path}")


def _decode_scan(source, video, directory, progress, cancel):
    scale = min(1.0, 960 / max(video["width"], video["height"]))
    width = max(2, int(video["width"] * scale) // 2 * 2)
    height = max(2, int(video["height"] * scale) // 2 * 2)
    detector = AdaptiveDetector(adaptive_threshold=2.0, min_content_val=3.0,
                                min_scene_len=max(2, round(video["fps"] * 0.15)))
    # Graphic ads often change only a portion of the image. Supplement adaptive
    # cuts with conservative absolute content cuts, separated by 0.4 seconds.
    content_detector = ContentDetector(threshold=8.0, min_scene_len=max(2, round(video["fps"] * 0.4)))
    filter_value = color_filter(video) + f",scale={width}:{height},format=bgr24"
    cuts, records, buckets = [], [], {}
    preview_dir = directory / "previews"
    preview_dir.mkdir()
    error_file = tempfile.TemporaryFile()
    process = subprocess.Popen([find_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(source),
                                "-map", "0:v:0", *_filter_file(directory, "scan", filter_value),
                                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error_file,
                               creationflags=CREATE_FLAGS)
    retained = set()
    previous_gray = None
    keep_following = -1
    last_change_time = -10.0
    try:
        size = width * height * 3
        for index, timing in enumerate(video["frames"]):
            _check(cancel)
            chunks, remaining = [], size
            while remaining:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    raise RuntimeError("Video decoding stopped before all probed frames were read.")
                chunks.append(chunk)
                remaining -= len(chunk)
            image = np.frombuffer(b"".join(chunks), np.uint8).reshape(height, width, 3)
            metrics = _cheap_metrics(image)
            records.append(metrics)
            cuts.extend(detector.process_frame(index, image))
            cuts.extend(content_detector.process_frame(index, image))
            bucket = int(timing["timestamp"] * 4)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                block_h, block_w = height // 16 * 16, width // 16 * 16
                blocks = difference[:block_h, :block_w].reshape(block_h // 16, 16, block_w // 16, 16)
                localized_change = (bool(block_h and block_w) and difference.mean() < 1.8
                                    and blocks.mean(axis=(1, 3)).max() > 10)
                if localized_change and timing["timestamp"] - last_change_time >= 0.10:
                    last_change_time = timing["timestamp"]
                    keep_following = index + 2
            if index <= keep_following:
                retained.add(index)
                _write_preview(preview_dir / f"frame_{index:08d}.jpg", image)
            previous_gray = gray
            entry = buckets.get(bucket)
            if entry is None:
                buckets[bucket] = {"first": index, "best": index, "focus": metrics["screen_focus"],
                                   "events": set()}
                retained.add(index)
                _write_preview(preview_dir / f"frame_{index:08d}.jpg", image)
            elif metrics["screen_focus"] > entry["focus"]:
                old = entry["best"]
                if old != entry["first"] and old not in entry["events"]:
                    retained.discard(old)
                    (preview_dir / f"frame_{old:08d}.jpg").unlink(missing_ok=True)
                entry.update(best=index, focus=metrics["screen_focus"])
                retained.add(index)
                _write_preview(preview_dir / f"frame_{index:08d}.jpg", image)
            if index <= keep_following:
                buckets[bucket]["events"].add(index)
            if index % 24 == 0:
                _progress(progress, "Scanning scenes", index + 1, video["frame_count"])
        if process.stdout.read(1):
            raise RuntimeError("Decoder produced more frames than FFprobe reported.")
        process.wait()
        error_file.seek(0)
        errors = error_file.read().decode("utf-8", "replace").strip()
        if process.returncode or errors:
            raise RuntimeError(errors or "Video decode failed.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        error_file.close()
    boundaries = sorted({0, video["frame_count"], *(int(x) for x in cuts)})
    scenes = []
    for scene_id, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end <= start:
            continue
        scenes.append({"id": scene_id, "start_frame": start, "end_frame": end - 1,
                       "start": video["frames"][start]["timestamp"],
                       "end": video["frames"][end - 1]["timestamp"] + video["frames"][end - 1]["duration"]})
        best = max(range(start, end), key=lambda i: records[i]["screen_focus"])
        retained.add(best)
    _progress(progress, "Scanning scenes", video["frame_count"], video["frame_count"])
    return sorted(retained), records, scenes, (width, height)


def _source_frames(source, video, indexes, directory, cancel=None):
    """Stream explicit native frames once; release each image before the next."""
    indexes = sorted(set(indexes))
    if not indexes:
        return
    width, height = video["width"], video["height"]
    select = "+".join(f"eq(n\\,{i})" for i in indexes)
    filters = "select=" + select + "," + color_filter(video) + ",format=bgr24"
    errors_file = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v:0",
         *_filter_file(directory, "native_candidates", filters), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors_file, creationflags=CREATE_FLAGS)
    try:
        length = width * height * 3
        for index in indexes:
            _check(cancel)
            pieces, remaining = [], length
            while remaining:
                piece = process.stdout.read(remaining)
                if not piece:
                    raise RuntimeError(f"Native candidate decoding stopped at frame {index}.")
                pieces.append(piece)
                remaining -= len(piece)
            yield index, np.frombuffer(b"".join(pieces), np.uint8).reshape(height, width, 3)
        if process.stdout.read(1):
            raise RuntimeError("Native candidate decoder produced an unexpected frame count.")
        process.wait()
        errors_file.seek(0)
        error = errors_file.read().decode("utf-8", "replace").strip()
        if process.returncode or error:
            raise RuntimeError(error or "Native candidate decoding failed.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        errors_file.close()


def _scale_regions(regions, sx, sy):
    result = []
    for original in regions:
        region = dict(original)
        x, y, w, h = original["box"]
        region["box"] = [round(x * sx), round(y * sy), max(1, round(w * sx)), max(1, round(h * sy))]
        for key in ("polygon", "raw_polygon", "recognition_polygon"):
            if key in original:
                region[key] = [[float(x) * sx, float(y) * sy] for x, y in original[key]]
        result.append(region)
    return result


def _native_focus(frame, faces, text):
    height, width = frame.shape[:2]
    tiles = []
    tile = min(256, width, height)
    for cy in (0.2, 0.5, 0.8):
        for cx in (0.2, 0.5, 0.8):
            x = min(max(0, round(width * cx - tile / 2)), width - tile)
            y = min(max(0, round(height * cy - tile / 2)), height - tile)
            tiles.append(measure_focus(frame[y:y + tile, x:x + tile]))
    def regions_focus(regions, kind):
        values = []
        for region in regions:
            x, y, w, h = region["box"]
            crop = frame[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
            if crop.size and min(crop.shape[:2]) >= 8:
                measured = measure_focus(crop)
                region["focus"] = measured["focus"]
                region["focus_evidence"] = measured
                valid = (region.get("confidence", 0) >= 0.45 if kind == "face" else
                         region.get("recognition_confidence", 1.0) >= 0.55 and
                         not region.get("clipped", False) and region.get("complete_geometry", True))
                if measured["reliability"] > 0.1 and valid:
                    values.append(measured["focus"])
        return values
    face_values = regions_focus(faces, "face")
    text_values = regions_focus(text, "text")
    valid_tiles = [t["focus"] for t in tiles if t["reliability"] > 0.1]
    return {"scene": float(np.median(valid_tiles)) if valid_tiles else 0.0,
            "face": float(np.percentile(face_values, 25)) if face_values else None,
            "text": float(np.median(text_values)) if text_values else None,
            "noise": float(np.median([t["noise"] for t in tiles])),
            "measurement": "native source pixels", "tiles": tiles}


def _analyze_native_candidates(source, video, indexes, records, scenes, size, directory, vision, progress, cancel):
    starts = [s["start_frame"] for s in scenes]
    candidates = []
    stream = _source_frames(source, video, indexes, directory, cancel)
    try:
        for number, (index, frame) in enumerate(stream):
            _check(cancel)
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            detection = vision.analyze(image)
            sx, sy = video["width"] / size[0], video["height"] / size[1]
            faces = _scale_regions(detection.get("faces", []), sx, sy)
            regions = _scale_regions(detection.get("text", []), sx, sy)
            if hasattr(vision, "refine_text_regions"):
                regions = vision.refine_text_regions(frame, regions)
            if hasattr(vision, "recognize_text"):
                regions = vision.recognize_text(frame, regions)
            focus = _native_focus(frame, faces, regions)
            detection["faces"] = _scale_regions(faces, 1 / sx, 1 / sy)
            detection["text"] = _scale_regions(regions, 1 / sx, 1 / sy)
            preview = f"previews/frame_{index:08d}.jpg"
            _write_preview(directory / preview, image)
            candidates.append({"id": index, "frame_index": index,
                               "timestamp": video["frames"][index]["timestamp"],
                               "pts": video["frames"][index]["pts"],
                               "scene_id": bisect.bisect_right(starts, index) - 1,
                               "preview": preview, "metrics": {**records[index], "native_focus": focus},
                               "vision": detection})
            _progress(progress, "Checking native sharpness and words", number + 1, len(indexes))
    finally:
        stream.close()
    return candidates

def _source_frame(source, video, index, size=None, cancel=None):
    # Input seeking plus original integer PTS selection avoids rounded-time extraction.
    timestamp = video["frames"][index]["timestamp"]
    pts = video["frames"][index]["pts"]
    width, height = size or (video["width"], video["height"])
    filters = f"select=eq(pts\\,{pts})," + color_filter(video)
    if size:
        filters += f",scale={width}:{height}"
    filters += ",format=bgr24"
    prefix = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-copyts"]
    seek = ["-ss", str(max(0, timestamp - 1.0))] if timestamp > 1 else []
    args = prefix + seek + ["-i", str(source), "-map", "0:v:0", "-vf", filters,
                            "-frames:v", "1", "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = _run(args, cancel)
    if len(raw) != width * height * 3:
        # Some containers reset timestamps on seek. Decode by original index as fallback.
        filters = filters.replace(f"select=eq(pts\\,{pts})", f"select=eq(n\\,{index})")
        args = prefix + ["-i", str(source), "-map", "0:v:0", "-vf", filters,
                         "-frames:v", "1", "-fps_mode", "passthrough",
                         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = _run(args, cancel)
    if len(raw) != width * height * 3:
        raise RuntimeError(f"Could not decode source frame {index} exactly.")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def _native_hashes(source, video, indexes, directory, cancel):
    select = "+".join(f"eq(n\\,{i})" for i in indexes)
    raw = _run([find_tool("ffmpeg"), "-v", "error", "-nostdin", "-noautorotate",
                "-i", str(source), "-map", "0:v:0",
                *_filter_file(directory, "hashes", "select=" + select),
                "-fps_mode", "passthrough", "-pix_fmt", video["pix_fmt"],
                "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256", "-"], cancel)
    hashes = [line.split(",")[-1].strip() for line in raw.decode().splitlines()
              if line.strip() and not line.startswith("#")]
    if len(hashes) != len(indexes):
        raise RuntimeError("Native duplicate hash pass did not match the candidate frame count.")
    return dict(zip(indexes, hashes))


def _text_stability(current, neighbors, shape):
    texts = current.get("text", [])
    if not texts:
        return 1.0
    diagonal = math.hypot(*shape[:2])
    scores = []
    for text in texts:
        box = np.asarray(text["box"], dtype=float)
        comparisons = []
        for neighbor in neighbors:
            others = neighbor.get("text", [])
            if not others:
                comparisons.append(0.0)
                continue
            other = min(others, key=lambda t: np.linalg.norm(np.asarray(t["box"]) - box))
            distance = float(np.linalg.norm(np.asarray(other["box"]) - box)) / diagonal
            contrast = abs(other.get("contrast", 0) - text.get("contrast", 0))
            comparisons.append(max(0, 1 - distance * 35 - contrast * 2.5))
        scores.append(float(np.mean(comparisons)) if comparisons else 0.65)
    return float(np.mean(scores))


def _score_candidates(candidates, scenes, video):
    def relative_focus(value, reference):
        if value is None or value <= 0:
            return 0.0
        return 0.65 + 0.30 * math.tanh(math.log(value / max(reference, 0.001)))
    native_references = {}
    for candidate in candidates:
        focus = candidate["metrics"].get("native_focus")
        if focus:
            # Text density varies by wording; compare each readable occurrence
            # within its scene instead of treating letter count as optical focus.
            key = (candidate["scene_id"], candidate.get("text_state_id"))
            native_references.setdefault(key, []).append(focus)
    native_references = {
        key: {kind: float(np.median([m[kind] for m in values if m.get(kind) is not None and m[kind] > 0]))
              if any(m.get(kind) is not None and m[kind] > 0 for m in values) else 1.0
              for kind in ("scene", "face", "text")}
        for key, values in native_references.items()}
    for scene in scenes:
        group = [c for c in candidates if c["scene_id"] == scene["id"]]
        if not group:
            continue
        sharp_values = [c["metrics"]["sharpness"] for c in group]
        scale = max(4, float(np.percentile(sharp_values, 85)))
        for position, candidate in enumerate(group):
            metrics, vision = candidate["metrics"], candidate["vision"]
            sharp = min(1, metrics["sharpness"] / scale)
            neighbors = [c["vision"] for c in group[max(0, position - 1):position + 2]
                         if c is not candidate]
            stable = _text_stability(vision, neighbors, (video["height"], video["width"]))
            faces = [f for f in vision.get("faces", []) if f.get("confidence", 0) >= 0.45]
            eyes = min((f.get("open_eyes", 1) for f in faces), default=1.0)
            smile = float(np.mean([f.get("smile", 0) for f in faces])) if faces else 0
            region_values = [f["sharpness"] for f in faces if "sharpness" in f]
            region_values += [t["sharpness"] for t in vision.get("text", []) if "sharpness" in t]
            region = min(1, float(np.mean(region_values)) / scale) if region_values else sharp
            severe_softness = (metrics["sharpness"] < 1.0 and max(region_values, default=0) < 20)
            exposure_penalty = min(12, metrics["clipped_fraction"] * 20)
            uninformative = (metrics.get("edge_fraction", 1) < 0.0002
                             and not faces and not vision.get("text"))
            blank = metrics["contrast"] < 2 or uninformative
            transition = (min(candidate["frame_index"] - scene["start_frame"],
                              scene["end_frame"] - candidate["frame_index"]) < 2 and len(group) > 3)
            score = 40 + 28 * sharp + 12 * region + 12 * stable + 8 * eyes + 2 * smile
            native = metrics.get("native_focus")
            if native:
                reference = native_references[(candidate["scene_id"], candidate.get("text_state_id"))]
                overall = relative_focus(native["scene"], reference["scene"])
                subjects = []
                if native.get("face") is not None:
                    subjects.append(relative_focus(native["face"], reference["face"]))
                if native.get("text") is not None:
                    subjects.append(relative_focus(native["text"], reference["text"]))
                detail = float(np.mean(subjects)) if subjects else overall
                # Continuous focus contribution: no top-percentile saturation.
                score = 20 + 37 * overall + 25 * detail + 10 * stable + 8 * eyes + smile
                score += 1.5 * candidate.get("text_complete", 1.0)
                metrics["focus_quality"] = round(0.4 * overall + 0.6 * detail, 5)
            score -= exposure_penalty + (12 if transition else 0)
            if faces and eyes < 0.45:
                score -= 18
            if blank:
                score = 0
            if severe_softness:
                score = min(score, 49)
            if candidate.get("text_ready") is False:
                score = min(score, 49)
            candidate["score"] = round(max(0, min(100, score)), 2)
            candidate["tier"] = "High" if score >= 75 else "Mid" if score >= 50 else "Low"
            candidate["eligible"] = score >= 50 and not blank
            candidate["reasons"] = []
            if sharp < 0.45:
                candidate["reasons"].append("Soft compared with other frames in this scene")
            if faces and eyes < 0.65:
                candidate["reasons"].append("Possible blink or closed eyes")
            if stable < 0.7 and vision.get("text"):
                candidate["reasons"].append("Text may still be moving or fading")
            if transition:
                candidate["reasons"].append("Near a scene transition")
            if blank:
                candidate["reasons"].append("Blank or fading frame")
            if severe_softness and not blank:
                candidate["reasons"].append("Severe softness without a clear subject or text region")
            if candidate.get("text_ready") is False:
                candidate["reasons"].append(candidate.get("text_reason", "Words are not yet complete and readable"))
            if metrics["clipped_fraction"] > 0.25:
                candidate["reasons"].append("Large highlight clipping area")
            if not candidate["reasons"]:
                candidate["reasons"].append("Sharp, stable candidate")
            candidate["metrics"].update(text_stability=round(stable, 3),
                                        open_eyes=round(eyes, 3), smile=round(smile, 3))
    # Camera and UI animation can split a transition into its own short scene.
    # Such a scene must not force a blurred/fading still into the coverage quota.
    grouped = [[c for c in candidates if c["scene_id"] == scene["id"]] for scene in scenes]
    for position, (scene, group) in enumerate(zip(scenes, grouped)):
        if not group:
            continue
        duration = scene.get("end", (scene["end_frame"] + 1) / video.get("fps", 24))
        duration -= scene.get("start", scene["start_frame"] / video.get("fps", 24))
        text_group = [c for c in group if c["vision"].get("text")]
        neighbors = [grouped[i] for i in (position - 1, position + 1) if 0 <= i < len(grouped)]
        reference = max((float(np.percentile([c["metrics"]["sharpness"] for c in n], 85))
                         for n in neighbors if n), default=0)
        unsettled = (text_group and len(text_group) >= len(group) / 2 and
                     float(np.median([c["metrics"]["text_stability"] for c in text_group])) < 0.6)
        if (duration < 1 and unsettled and
                max(c["metrics"]["sharpness"] for c in group) < reference * 0.6):
            for candidate in group:
                candidate["eligible"] = False
                candidate["score"] = min(candidate["score"], 49)
                candidate["tier"] = "Low"
                candidate["reasons"].append("Unsettled transition between sharper scenes")


def _near_match(first, second):
    # Conservative local differences protect small title/pose/expression changes.
    difference = cv2.absdiff(first, second).max(axis=2)
    if float(difference.mean()) > 1.5 or np.mean(difference > 12) > 0.005:
        return False
    if float(np.percentile(difference, 99.95)) > 25:
        return False
    height, width = difference.shape
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            if difference[y:y + 32, x:x + 32].mean() > 5:
                return False
    return True


def _deduplicate(candidates, source, video, directory, end_id, progress, cancel):
    hashes = _native_hashes(source, video, [c["id"] for c in candidates], directory, cancel)
    exact, representatives, cache = {}, [], OrderedDict()
    phasher = cv2.img_hash.PHash_create()
    order = sorted(candidates, key=lambda c: (c["id"] == end_id, c["score"]), reverse=True)

    def full_frame(index):
        if index not in cache:
            cache[index] = _source_frame(source, video, index, cancel=cancel)
            if len(cache) > 4:
                cache.popitem(last=False)
        cache.move_to_end(index)
        return cache[index]

    for number, candidate in enumerate(order):
        _check(cancel)
        candidate["native_hash"] = hashes[candidate["id"]]
        candidate["duplicate_of"] = None
        if not candidate["eligible"]:
            continue
        if candidate["native_hash"] in exact:
            candidate["duplicate_of"] = exact[candidate["native_hash"]]
            candidate["reasons"].append("Exact duplicate of a stronger candidate")
        else:
            image = cv2.imread(str(directory / candidate["preview"]))
            phash = phasher.compute(image)
            for other, other_hash in representatives:
                if candidate.get("text_contents", []) != other.get("text_contents", []):
                    # A small but meaningful word or digit change can share a
                    # perceptual hash with an otherwise identical title card.
                    continue
                if phasher.compare(phash, other_hash) > 6:
                    continue
                other_preview = cv2.imread(str(directory / other["preview"]))
                # Most perceptual-hash matches are distinct textured or moving
                # frames. Reject those cheaply before source-resolution checks.
                # Full resolution still confirms every proposed duplicate merge.
                if not _near_match(image, other_preview):
                    continue
                # Expressions and text stability are meaningful even before pixel verification.
                if abs(candidate["metrics"]["open_eyes"] - other["metrics"]["open_eyes"]) > 0.15:
                    continue
                if _near_match(full_frame(candidate["id"]), full_frame(other["id"])):
                    candidate["duplicate_of"] = other["id"]
                    candidate["reasons"].append("Near duplicate confirmed at full resolution")
                    break
            if candidate["duplicate_of"] is None:
                representatives.append((candidate, phash))
        exact[candidate["native_hash"]] = (candidate["id"] if candidate["duplicate_of"] is None
                                          else candidate["duplicate_of"])
        _progress(progress, "Removing duplicates", number + 1, len(order))


def select_candidates(candidates, scenes, base_target, count=None, end_id=None):
    pool = [c for c in candidates if c["eligible"] and c.get("text_ready", True)
            and c["duplicate_of"] is None]
    if any(c.get("text_state_id") is not None for c in pool):
        return _select_word_coverage(pool, base_target, count, end_id, candidates)
    representatives = []
    for scene in scenes:
        group = [c for c in pool if c["scene_id"] == scene["id"]]
        if group:
            representatives.append(max(group, key=lambda c: c["score"]))
    target = count if count is not None else max(base_target, len(representatives))
    selected = []
    if end_id is not None and any(c["id"] == end_id for c in pool):
        selected.append(end_id)
    # With an override, choose scenes across the timeline rather than only the sharpest scenes.
    if len(representatives) > target:
        indexes = np.linspace(0, len(representatives) - 1, target).round().astype(int)
        representatives = [representatives[i] for i in sorted(set(indexes))]
    for candidate in representatives:
        if len(selected) < target and candidate["id"] not in selected:
            selected.append(candidate["id"])
    while len(selected) < target:
        remaining = [c for c in _quality_fill_pool(pool) if c["id"] not in selected]
        if not remaining:
            break
        selected_times = [c["timestamp"] for c in pool if c["id"] in selected]
        def utility(candidate):
            distance = min((abs(candidate["timestamp"] - t) for t in selected_times), default=1)
            return candidate["score"] + 35 * min(distance, 2)
        selected.append(max(remaining, key=utility)["id"])
    selected.sort()
    if end_id in selected:
        selected.remove(end_id)
        selected.append(end_id)
    return selected, target


def _select_word_coverage(pool, base_target, count, end_id, all_candidates=None):
    """Cover scenes and readable wording automatically; an override stays a cap."""
    by_id = {c["id"]: c for c in pool}
    covers = {c["id"]: {("scene", c["scene_id"])} |
              ({("text", c["text_state_id"])} if c.get("text_state_id") is not None else set())
              for c in pool}
    for candidate in all_candidates or []:
        duplicate = candidate.get("duplicate_of")
        if (duplicate in covers and candidate["eligible"] and candidate.get("text_ready", True) and
                candidate.get("text_contents", []) == by_id[duplicate].get("text_contents", [])):
            covers[duplicate].add(("scene", candidate["scene_id"]))
            if candidate.get("text_state_id") is not None:
                covers[duplicate].add(("text", candidate["text_state_id"]))
    required = set().union(*covers.values())
    selected = [end_id] if end_id in by_id else []
    covered = set().union(*(covers[i] for i in selected)) if selected else set()
    limit = count if count is not None else len(pool)
    while required - covered and len(selected) < limit:
        available = [c for c in pool if c["id"] not in selected and covers[c["id"]] - covered]
        if not available:
            break
        def utility(c):
            gained = covers[c["id"]] - covered
            # Preserve actual text changes even when one continuous shot has
            # several captions; a persistent logo cannot exhaust its scene.
            return (sum(2 if kind == "text" else 1 for kind, _ in gained), c["score"])
        best = max(available, key=utility)
        selected.append(best["id"])
        covered.update(covers[best["id"]])
    target = count if count is not None else max(base_target, len(selected))
    while len(selected) < target:
        available = [c for c in _quality_fill_pool(pool) if c["id"] not in selected]
        if not available:
            break
        selected_times = [by_id[i]["timestamp"] for i in selected]
        def utility(c):
            distance = min((abs(c["timestamp"] - t) for t in selected_times), default=1)
            return c["score"] + 35 * min(distance, 2)
        selected.append(max(available, key=utility)["id"])
    selected.sort()
    if end_id in selected:
        selected.remove(end_id)
        selected.append(end_id)
    return selected, target


def _quality_fill_pool(pool):
    """Temporal variety must not pad a run with clearly weaker native frames."""
    best = {}
    for candidate in pool:
        focus = candidate.get("metrics", {}).get("focus_quality")
        if focus is not None:
            key = (candidate["scene_id"], candidate.get("text_state_id"))
            best[key] = max(best.get(key, 0), focus)
    return [c for c in pool if c.get("metrics", {}).get("focus_quality") is None or
            c["metrics"]["focus_quality"] >= best[(c["scene_id"], c.get("text_state_id"))] - 0.10]


def _record_text_coverage(candidates, text_states, selected):
    by_id = {c["id"]: c for c in candidates}
    selected = set(selected)
    for state in text_states:
        state["selected_frame_id"] = None
        state["covered_by_duplicate"] = False
        for index in state["candidate_ids"]:
            candidate = by_id[index]
            if not candidate["eligible"] or not candidate.get("text_ready", True):
                continue
            duplicate = candidate.get("duplicate_of")
            target = index if duplicate is None else duplicate
            if target in selected and candidate.get("text_contents", []) == by_id[target].get("text_contents", []):
                state["selected_frame_id"] = target
                state["covered_by_duplicate"] = index != target
                break


def analyze_video(path, options=None, progress=None, cancel=None):
    options = options or Options()
    options.validate()
    source = Path(path).expanduser().resolve()
    _progress(progress, "Probing video", 0, 1)
    identity = fingerprint(source, cancel)
    video = probe_video(source, cancel)
    directory = _make_run_dir(source, options.output)
    warnings = []
    if any(video.get(k, "unknown") in {"unknown", "unspecified", ""}
           for k in ("color_space", "color_transfer", "color_primaries")):
        warnings.append("Missing color metadata: untagged components were assumed Rec.709 SDR.")
    try:
        from .vision import LocalVision
        vision = LocalVision()
        try:
            indexes, records, scenes, size = _decode_scan(source, video, directory, progress, cancel)
            from .text_states import assign_text_states
            analysis_video = {**video, "width": size[0], "height": size[1]}
            candidates = _analyze_native_candidates(
                source, video, indexes, records, scenes, size, directory, vision, progress, cancel)
            text_states = assign_text_states(candidates, scenes, analysis_video)
            _score_candidates(candidates, scenes, {**video, "width": size[0], "height": size[1]})
            # Revisit a wider native-rate neighborhood around two diverse
            # leading moments per scene and every readable wording occurrence.
            existing = {c["id"] for c in candidates}
            refine = set()
            anchors = []
            for scene in scenes:
                group = [c for c in candidates if c["scene_id"] == scene["id"] and c["eligible"]]
                if group:
                    best = max(group, key=lambda c: c["score"])
                    anchors.append(best)
                    distant = [c for c in group if abs(c["timestamp"] - best["timestamp"]) >= 0.6]
                    if distant:
                        anchors.append(max(distant, key=lambda c: c["score"]))
            for state in text_states:
                group = [c for c in candidates if c.get("text_state_id") == state["id"]]
                if group:
                    anchors.append(max(group, key=lambda c: (c["text_ready"], c["score"])))
            # Unsupported brief readings need a second native observation.
            anchors += [c for c in candidates if c.get("text_contents") and c.get("text_state_id") is None]
            for candidate in anchors:
                scene = scenes[candidate["scene_id"]]
                index = candidate["id"]
                start_time = candidate["timestamp"] - 0.25
                end_time = candidate["timestamp"] + 0.25
                neighborhood = [i for i in range(max(scene["start_frame"], index - math.ceil(video["fps"] * .4)),
                                                 min(scene["end_frame"], index + math.ceil(video["fps"] * .4)) + 1)
                                if start_time <= video["frames"][i]["timestamp"] <= end_time]
                refine.update(neighborhood)
            additions = sorted(refine - existing)
            candidates += _analyze_native_candidates(
                source, video, additions, records, scenes, size, directory, vision, progress, cancel)
            candidates.sort(key=lambda c: c["id"])
            text_states = assign_text_states(candidates, scenes, analysis_video)
            _score_candidates(candidates, scenes, analysis_video)
        finally:
            vision.close()
        end_id = None
        for scene in reversed(scenes):
            group = [c for c in candidates if c["scene_id"] == scene["id"] and c["eligible"]]
            if group:
                latest = max(c["timestamp"] for c in group)
                group = [c for c in group if c["timestamp"] >= latest - min(3, scene["end"] - scene["start"])]
                terminal = max(group, key=lambda c: c["timestamp"])
                terminal_state = terminal.get("text_state_id")
                if terminal_state is not None:
                    group = [c for c in group if c.get("text_state_id") == terminal_state]
                settled = [c for c in group if c["metrics"]["text_stability"] >= 0.8]
                end_id = max(settled or group, key=lambda c: c["score"])["id"]
                break
        if end_id is None:
            warnings.append("No suitable stable ending frame was found.")
        _deduplicate(candidates, source, video, directory, end_id, progress, cancel)
        base = duration_target(video["duration"], video["fps"])
        selected, target = select_candidates(candidates, scenes, base, options.count, end_id)
        _record_text_coverage(candidates, text_states, selected)
        missed_states = [s["id"] for s in text_states if s["selected_frame_id"] is None and
                         any(c.get("text_state_id") == s["id"] and c["eligible"] for c in candidates)]
        if missed_states:
            warnings.append(f"{len(missed_states)} readable text occurrences were not selected"
                            + (" because of the explicit count limit." if options.count is not None else
                               " after duplicate verification; see the candidate report."))
        if len(selected) < target:
            warnings.append(f"Selected {len(selected)} distinct eligible frames; target was {target}.")
        analysis = {"schema_version": 1, "scoring_version": "2.0-b3",
                    "source": str(source), "source_fingerprint": identity, "video": video,
                    "analysis_dir": str(directory), "scenes": scenes, "candidates": candidates,
                    "selected_ids": selected, "end_card_id": end_id, "text_states": text_states,
                    "base_target": base, "target": target, "warnings": warnings}
        (directory / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        if options.report:
            from .reports import write_reports
            write_reports(analysis)
        return analysis
    except Exception as error:
        (directory / "FAILED.txt").write_text(str(error), encoding="utf-8")
        raise


def _embed_tiff_srgb(path):
    # FFmpeg 7.1 TIFF encoding drops frame ICC side data. Append an IFD with an
    # ICC tag while preserving every existing compressed pixel byte and offset.
    from PIL import ImageCms
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    with Path(path).open("r+b") as stream:
        header = stream.read(8)
        endian = "<" if header[:2] == b"II" else ">" if header[:2] == b"MM" else None
        if endian is None or struct.unpack(endian + "H", header[2:4])[0] != 42:
            raise ValueError("Expected a classic TIFF from the FFmpeg encoder.")
        original_ifd = struct.unpack(endian + "I", header[4:8])[0]
        stream.seek(original_ifd)
        count = struct.unpack(endian + "H", stream.read(2))[0]
        entries = [stream.read(12) for _ in range(count)]
        next_ifd = stream.read(4)
        entries = [entry for entry in entries if struct.unpack(endian + "H", entry[:2])[0] != 34675]
        stream.seek(0, 2)
        if stream.tell() % 2:
            stream.write(b"\0")
        profile_offset = stream.tell()
        stream.write(profile)
        if stream.tell() % 2:
            stream.write(b"\0")
        new_ifd = stream.tell()
        entries.append(struct.pack(endian + "HHII", 34675, 7, len(profile), profile_offset))
        entries.sort(key=lambda entry: struct.unpack(endian + "H", entry[:2])[0])
        stream.write(struct.pack(endian + "H", len(entries)))
        stream.write(b"".join(entries))
        stream.write(next_ifd)
        stream.seek(4)
        stream.write(struct.pack(endian + "I", new_ifd))


def export_analysis(analysis, options=None, frame_ids=None, progress=None, cancel=None):
    options = options or Options()
    options.validate()
    if isinstance(analysis, (str, Path)):
        file = Path(analysis)
        if file.is_dir():
            file /= "analysis.json"
        analysis = json.loads(file.read_text(encoding="utf-8"))
        analysis["analysis_dir"] = str(file.resolve().parent)
    source = Path(analysis["source"])
    if fingerprint(source, cancel) != analysis["source_fingerprint"]:
        raise ValueError("The source video changed after analysis. Analyze it again before exporting.")
    candidates = {c["id"]: c for c in analysis["candidates"]}
    requested = analysis["selected_ids"] if frame_ids is None else list(dict.fromkeys(frame_ids))
    if not requested:
        raise ValueError("No frames selected for export.")
    if set(requested) - candidates.keys():
        raise ValueError("Selection contains frame IDs absent from this analysis.")
    order = sorted(requested)
    end_id = analysis.get("end_card_id")
    if end_id in order:
        order.remove(end_id)
        order.append(end_id)
    coverage = [dict(state) for state in analysis.get("text_states", [])]
    _record_text_coverage(analysis["candidates"], coverage, order)
    directory = Path(analysis["analysis_dir"])
    # Subsequent manual exports never overwrite an earlier extraction.
    changed_output = (options.output is not None and
                      Path(options.output).expanduser().resolve() != directory.parent.parent.resolve())
    if (directory / "manifest.json").exists() or changed_output:
        directory = _make_run_dir(source, options.output or directory.parent.parent)
    extension = {"png": "png", "tiff": "tif", "jpeg": "jpg"}[options.format]
    video = analysis["video"]
    select = "+".join(f"eq(n\\,{i})" for i in sorted(requested))
    pixel_format = "rgb24" if options.bit_depth == 8 else ("rgb48be" if extension == "png" else "rgb48le")
    filters = "select=" + select + "," + color_filter(video, options.color) + f",format={pixel_format}"
    if options.color == "srgb":
        filters += ",iccgen=color_primaries=bt709:color_trc=iec61966-2-1:force=1"
    codec_options = []
    if options.format == "jpeg":
        codec_options = ["-q:v", "2", "-pix_fmt", "yuvj444p"]
    elif options.format == "tiff":
        codec_options = ["-compression_algo", "deflate"]
    pattern = str(directory / f".export_%06d.{extension}")
    _progress(progress, "Exporting original-resolution stills", 0, len(requested))
    try:
        _run([find_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(source),
              "-map", "0:v:0", *_filter_file(directory, "export", filters),
              "-fps_mode", "passthrough", "-start_number", "1", *codec_options, pattern], cancel)
        temporaries = sorted(directory.glob(f".export_*.{extension}"))
        if len(temporaries) != len(requested):
            raise RuntimeError("Exported frame count does not match the explicit selection.")
        if options.format == "tiff" and options.color == "srgb":
            for temporary in temporaries:
                _check(cancel)
                _embed_tiff_srgb(temporary)
        temporary_map = dict(zip(sorted(requested), temporaries))
        files = []
        for number, index in enumerate(order, 1):
            destination = directory / f"{source.stem}_still_{number:03d}_frame_{index:08d}.{extension}"
            temporary_map[index].rename(destination)
            files.append({"frame_index": index, "timestamp": candidates[index]["timestamp"],
                          "path": str(destination), "score": candidates[index]["score"],
                          "recognized_text": candidates[index].get("text_contents", []),
                          "text_ready": candidates[index].get("text_ready", True),
                          "text_state_ids": [s["id"] for s in coverage
                                             if s.get("selected_frame_id") == index]})
        warnings = [w for w in analysis["warnings"] if not w.startswith("Selected ")]
        if len(files) < analysis["target"]:
            warnings.append(f"Selected {len(files)} distinct eligible frames; target was {analysis['target']}.")
        result = {"schema_version": 1, "status": "success", "source": str(source),
                  "output_dir": str(directory), "files": files, "selected_count": len(files),
                  "target": analysis["target"], "warnings": warnings,
                  "format": options.format, "bit_depth": options.bit_depth, "color": options.color,
                  "source_color": {key: video.get(key) for key in
                                   ("color_primaries", "color_transfer", "color_space", "color_range", "pix_fmt")},
                  "output_color": {"primaries": "bt709", "transfer": "iec61966-2-1", "range": "full"}
                                  if options.color == "srgb" else
                                  {"primaries": video.get("color_primaries"),
                                   "transfer": video.get("color_transfer"), "range": "full"},
                  "analysis": str(Path(analysis["analysis_dir"]) / "analysis.json")}
        result["text_states"] = coverage
        (directory / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if options.color == "source":
            (directory / "color_metadata.json").write_text(
                json.dumps({k: result[k] for k in ("source_color", "output_color")}, indent=2),
                encoding="utf-8")
        if options.report:
            from .reports import write_reports
            write_reports({**analysis, "selected_ids": order, "warnings": warnings,
                           "text_states": coverage}, directory)
        _progress(progress, "Exporting original-resolution stills", len(files), len(files))
        return result
    except Exception:
        for temporary in directory.glob(f".export_*.{extension}"):
            temporary.unlink(missing_ok=True)
        raise


def extract_legacy(path, progress=None, cancel=None):
    source = Path(path).resolve()
    # Validate before invoking the unmodified original, which prints errors instead of raising.
    video = probe_video(source, cancel)
    _progress(progress, "Legacy extraction", 0, video["frame_count"])
    args = [sys.executable]
    if not getattr(sys, "frozen", False):
        args += ["-m", "stills_tool"]
    args += ["--legacy-worker", str(source)]
    _run(args, cancel)
    indexes = list(range(0, video["frame_count"], 24))
    if indexes[-1] != video["frame_count"] - 1:
        indexes.append(video["frame_count"] - 1)
    directory = source.parent / "stills"
    files = []
    for index in indexes:
        file = directory / f"{source.stem}_frame_{index}.png"
        if not file.is_file():
            raise RuntimeError(f"The legacy extractor did not write frame {index}.")
        files.append({"frame_index": index, "timestamp": video["frames"][index]["timestamp"],
                      "path": str(file)})
    _progress(progress, "Legacy extraction", video["frame_count"], video["frame_count"])
    return {"status": "success", "source": str(source), "output_dir": str(directory),
            "files": files, "selected_count": len(files), "target": len(files), "warnings": []}
