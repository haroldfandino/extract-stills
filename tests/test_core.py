"""Engine integration tests: real video decoding and real image pixels.

Only the local face/text model is replaced by a neutral deterministic fixture.
FFmpeg, FFprobe, scene detection, selection, duplicate checks and export stay real.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib

import cv2
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stills_tool import cli, core, legacy_original, vision
from stills_tool.types import CancelledError, Options


@pytest.fixture
def ffmpeg():
    try:
        return core.find_tool("ffmpeg")
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.fixture
def neutral_vision(monkeypatch):
    instances = []

    class NeutralVision:
        def __init__(self):
            self.closed = False
            instances.append(self)

        def analyze(self, frame):
            assert frame.dtype == np.uint8 and frame.ndim == 3
            return {"faces": [], "text": [], "warnings": []}

        def close(self):
            self.closed = True

    monkeypatch.setattr(vision, "LocalVision", NeutralVision)
    return instances


def patterned_frames(count: int, height: int = 64, width: int = 96) -> list[np.ndarray]:
    """Known pixel content identifies every frame independently of timestamps."""
    y, x = np.indices((height, width))
    checker = ((x // 8 + y // 8) % 2).astype(np.uint8)
    base = np.stack((35 + checker * 80, 65 + checker * 90, 95 + checker * 70), axis=2)
    frames = []
    for index in range(count):
        frame = base.copy()
        frame[:8, :8] = [20 + index * 7 % 180, 60 + index * 11 % 160, 90 + index * 13 % 120]
        start = 10 + index % (width - 30)
        frame[24:40, start:start + 12] = [210, 185, 45]
        frames.append(frame)
    return frames


def high_precision_frame(height: int = 48, width: int = 80) -> np.ndarray:
    y, x = np.indices((height, width))
    # Low-order bits deliberately vary. An 8-bit round trip cannot preserve them.
    return np.stack((
        9000 + 117 * x + 61 * y,
        18000 + 89 * x + 43 * y,
        29000 + 71 * x + 29 * y,
    ), axis=2).astype("<u2")


def encode(ffmpeg: str, path: Path, frames: list[np.ndarray], *, vfr=False, codec=None,
           primaries="bt709", transfer="iec61966-2-1") -> Path:
    height, width = frames[0].shape[:2]
    sixteen = frames[0].dtype == np.uint16
    pixel_format = "bgr48le" if sixteen else "bgr24"
    command = [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", pixel_format,
        "-video_size", f"{width}x{height}", "-framerate", "24", "-i", "pipe:0",
    ]
    filters = [f"setparams=range=full:color_primaries={primaries}:color_trc={transfer}:colorspace=gbr"]
    if vfr:
        filters += ["settb=1/1000", "setpts=5000+N*N*40"]
    command += ["-vf", ",".join(filters)]
    if vfr:
        command += ["-fps_mode", "passthrough", "-enc_time_base", "filter"]
    command += codec or ["-c:v", "ffv1", "-pix_fmt", "gbrp16le" if sixteen else "bgr0"]
    command += [
        "-color_primaries", primaries, "-color_trc", transfer,
        "-colorspace", "rgb", "-color_range", "pc", str(path),
    ]
    result = subprocess.run(
        command, input=b"".join(frame.tobytes() for frame in frames),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return path


def analysis_for(source: Path, directory: Path, indexes: list[int], *, end_id=None) -> dict:
    """An explicit frame selection isolates decoding/export from scoring."""
    directory.mkdir(parents=True)
    video = core.probe_video(source)
    candidates = [{
        "id": index, "frame_index": index, "timestamp": video["frames"][index]["timestamp"],
        "pts": video["frames"][index]["pts"], "scene_id": 0, "score": 90.0,
        "tier": "High", "reasons": ["Integration fixture selection"],
        "metrics": {"sharpness": 100.0, "open_eyes": 1.0, "text_stability": 1.0},
        "preview": f"previews/frame_{index:08d}.jpg", "eligible": True, "duplicate_of": None,
    } for index in indexes]
    analysis = {
        "schema_version": 1, "scoring_version": "integration-fixture", "source": str(source.resolve()),
        "source_fingerprint": core.fingerprint(source), "analysis_dir": str(directory.resolve()),
        "video": video, "scenes": [{
            "id": 0, "start": 0.0, "end": video["duration"], "start_frame": 0,
            "end_frame": video["frame_count"] - 1,
        }], "candidates": candidates, "selected_ids": list(indexes), "end_card_id": end_id,
        "target": len(indexes), "base_target": len(indexes), "warnings": [],
    }
    (directory / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    return analysis


def read_pixels(path: str | Path) -> np.ndarray:
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert pixels is not None, f"Could not read exported image {path}"
    return pixels


def png_chunks(path: str | Path) -> dict[bytes, bytes]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = {}
    offset = 8
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        name = data[offset + 4:offset + 8]
        chunks[name] = data[offset + 8:offset + 8 + size]
        offset += size + 12
        if name == b"IEND":
            break
    return chunks


def assert_srgb_profile(path: str | Path, image_format: str) -> None:
    if image_format == "png":
        chunks = png_chunks(path)
        assert b"iCCP" in chunks, "sRGB PNG must contain an actual embedded ICC profile"
        name, compressed = chunks[b"iCCP"].split(b"\0", 1)
        assert name and compressed[0] == 0
        profile = zlib.decompress(compressed[1:])
    else:
        with Image.open(path) as image:
            profile = image.info.get("icc_profile")
            if image_format == "tiff" and not profile:
                profile = image.tag_v2.get(34675)
        assert profile, f"sRGB {image_format} must contain an actual embedded ICC profile"
    assert profile[36:40] == b"acsp", "ICC payload must contain the profile header signature"
    assert struct.unpack(">I", profile[:4])[0] == len(profile)


def test_vfr_nonzero_pts_and_export_preserve_original_frame_identity(tmp_path, ffmpeg):
    frames = patterned_frames(9)
    source = encode(ffmpeg, tmp_path / "variable.mkv", frames, vfr=True)
    video = core.probe_video(source)
    assert video["first_source_time"] == pytest.approx(5.0)
    assert [row["source_time"] for row in video["frames"]] == pytest.approx(
        [5 + index * index * 0.04 for index in range(9)], abs=0.001,
    )
    assert [row["timestamp"] for row in video["frames"]] == pytest.approx(
        [index * index * 0.04 for index in range(9)], abs=0.001,
    )
    for index in (0, 3, 8):
        decoded = core._source_frame(source, video, index)
        np.testing.assert_allclose(decoded, frames[index], atol=1)
    analysis = analysis_for(source, tmp_path / "analysis", [0, 3, 8], end_id=3)
    result = core.export_analysis(Path(analysis["analysis_dir"]) / "analysis.json", Options())
    assert [item["frame_index"] for item in result["files"]] == [0, 8, 3]
    assert [item["timestamp"] for item in result["files"]] == pytest.approx([0.0, 2.56, 0.36])
    for item in result["files"]:
        np.testing.assert_allclose(read_pixels(item["path"]), frames[item["frame_index"]], atol=1)
    assert_srgb_profile(result["files"][0]["path"], "png")


def test_rotation_uses_display_orientation_and_original_frame_ids(tmp_path, ffmpeg):
    frames = patterned_frames(4, height=48, width=80)
    original = encode(
        ffmpeg, tmp_path / "unrotated.mov", frames,
        codec=["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "bgr24"],
    )
    rotated = tmp_path / "rotated.mov"
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", str(original),
         "-c", "copy", str(rotated)], check=True, timeout=20,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    analysis = analysis_for(rotated, tmp_path / "analysis", [1, 3])
    assert analysis["video"]["rotation"] == 90
    assert (analysis["video"]["width"], analysis["video"]["height"]) == (48, 80)
    result = core.export_analysis(analysis, Options())
    for item in result["files"]:
        image = read_pixels(item["path"])
        assert image.shape == (80, 48, 3)
        np.testing.assert_allclose(image, np.rot90(frames[item["frame_index"]]), atol=1)


@pytest.mark.parametrize("count,expected", [(25, [0, 24]), (50, [0, 24, 48, 49])])
def test_legacy_matches_unmodified_extractor_pixels_names_and_final_frame(tmp_path, ffmpeg, count, expected):
    frames = patterned_frames(count)
    source = encode(ffmpeg, tmp_path / "fixture.mkv", frames)
    reference = tmp_path / "reference"
    reference.mkdir()
    reference_source = reference / source.name
    shutil.copy2(source, reference_source)
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_original.extract_stills(str(reference_source))
    result = core.extract_legacy(source)
    assert Path(result["output_dir"]) == tmp_path / "stills"
    assert [item["frame_index"] for item in result["files"]] == expected
    actual_names = sorted(Path(result["output_dir"]).glob("*.png"))
    reference_names = sorted((reference / "stills").glob("*.png"))
    assert [file.name for file in actual_names] == [file.name for file in reference_names]
    for item in result["files"]:
        name = f"fixture_frame_{item['frame_index']}.png"
        assert Path(item["path"]).name == name
        actual = read_pixels(item["path"])
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, read_pixels(reference / "stills" / name))
        np.testing.assert_array_equal(actual, frames[item["frame_index"]])


@pytest.mark.parametrize("image_format,depth", [
    ("png", 8), ("png", 16), ("tiff", 8), ("tiff", 16), ("jpeg", 8),
])
def test_real_export_format_bit_depth_color_and_icc(tmp_path, ffmpeg, image_format, depth):
    frame = high_precision_frame()
    source = encode(ffmpeg, tmp_path / "precision.mkv", [frame])
    analysis = analysis_for(source, tmp_path / "analysis", [0])
    result = core.export_analysis(analysis, Options(format=image_format, bit_depth=depth))
    file = Path(result["files"][0]["path"])
    assert file.suffix == {"png": ".png", "tiff": ".tif", "jpeg": ".jpg"}[image_format]
    image = read_pixels(file)
    assert image.shape == frame.shape
    assert image.dtype == (np.uint16 if depth == 16 else np.uint8)
    if image_format == "png":
        assert png_chunks(file)[b"IHDR"][8] == depth
    if image_format == "tiff":
        with Image.open(file) as tiff:
            assert tuple(tiff.tag_v2[258]) == (depth, depth, depth)
    assert_srgb_profile(file, image_format)
    if depth == 16:
        assert np.any(image % 257 != 0), "16-bit output must retain more than 8-bit precision"
        # Floating-point color conversion may differ by at most 1/4096 full scale.
        np.testing.assert_allclose(image, frame, atol=16)
    elif image_format != "jpeg":
        np.testing.assert_allclose(image.astype(float), frame.astype(float) / 257, atol=2)
    else:
        assert np.mean(np.abs(image.astype(float) - frame.astype(float) / 257)) < 3
    manifest = json.loads((Path(result["output_dir"]) / "manifest.json").read_text())
    assert manifest["bit_depth"] == depth
    assert manifest["output_color"] == {
        "primaries": "bt709", "transfer": "iec61966-2-1", "range": "full",
    }


@pytest.mark.parametrize("image_format", ["png", "tiff"])
def test_source_color_exports_preserve_high_precision_and_metadata(tmp_path, ffmpeg, image_format):
    frame = high_precision_frame()
    source = encode(ffmpeg, tmp_path / "source-color.mkv", [frame], transfer="bt709")
    analysis = analysis_for(source, tmp_path / "analysis", [0])
    result = core.export_analysis(analysis, Options(format=image_format, bit_depth=16, color="source"))
    image = read_pixels(result["files"][0]["path"])
    assert image.dtype == np.uint16 and np.any(image % 257 != 0)
    np.testing.assert_allclose(image, frame, atol=1)
    metadata = json.loads((Path(result["output_dir"]) / "color_metadata.json").read_text())
    assert metadata["source_color"]["color_transfer"] == "bt709"
    assert metadata["output_color"]["transfer"] == "bt709"
    assert metadata["output_color"]["range"] == "full"


def test_native_hash_does_not_round_away_one_code_value_change(tmp_path, ffmpeg):
    before = high_precision_frame()
    after = before.copy()
    after[13, 17, 1] += 1
    source = encode(ffmpeg, tmp_path / "native.mkv", [before, after, after])
    video = core.probe_video(source)
    hashes = core._native_hashes(source, video, [0, 1, 2], tmp_path, None)
    assert hashes[0] != hashes[1]
    assert hashes[1] == hashes[2]


def test_vfr_analysis_selection_and_refinement_keep_source_ids(tmp_path, ffmpeg, neutral_vision):
    frames = patterned_frames(9)
    source = encode(ffmpeg, tmp_path / "vfr-analysis.mkv", frames, vfr=True)
    analysis = core.analyze_video(source, Options(output=tmp_path / "output", count=3))
    assert analysis["video"]["first_source_time"] == pytest.approx(5.0)
    for candidate in analysis["candidates"]:
        index = candidate["id"]
        assert candidate["frame_index"] == index
        assert candidate["timestamp"] == pytest.approx(index * index * 0.04, abs=0.001)
        assert candidate["pts"] == analysis["video"]["frames"][index]["pts"]
        assert (Path(analysis["analysis_dir"]) / candidate["preview"]).is_file()
    assert len(analysis["selected_ids"]) == 3
    result = core.export_analysis(analysis, Options())
    for item in result["files"]:
        np.testing.assert_allclose(read_pixels(item["path"]), frames[item["frame_index"]], atol=1)


def test_hdr_to_srgb_is_nonblank_monotonic_and_profiled(tmp_path, ffmpeg):
    frame = np.zeros((48, 80, 3), dtype="<u2")
    for index, value in enumerate((5000, 16000, 32000, 48000)):
        frame[:, index * 20:(index + 1) * 20] = value
    source = encode(
        ffmpeg, tmp_path / "hdr.mkv", [frame], primaries="bt2020", transfer="smpte2084",
    )
    analysis = analysis_for(source, tmp_path / "analysis", [0])
    result = core.export_analysis(analysis, Options())
    image = read_pixels(result["files"][0]["path"])
    values = [float(image[:, index * 20:(index + 1) * 20].mean()) for index in range(4)]
    assert all(second > first for first, second in zip(values, values[1:]))
    assert values[-1] > 100 and values[0] < 30
    assert_srgb_profile(result["files"][0]["path"], "png")


def test_short_distinct_scene_is_covered_and_static_holds_are_collapsed(tmp_path, ffmpeg, neutral_vision):
    first, second, third = patterned_frames(3)
    second = second[:, :, ::-1].copy()
    third = np.roll(third, 4, axis=0)
    source = encode(ffmpeg, tmp_path / "short-shot.mkv", [first] * 19 + [second] * 4 + [third] * 19)
    analysis = core.analyze_video(source, Options(output=tmp_path / "output"))
    candidates = {candidate["id"]: candidate for candidate in analysis["candidates"]}
    assert any(19 <= index < 23 for index in analysis["selected_ids"]), "The short middle shot must be represented"
    assert len(analysis["selected_ids"]) == 3, "Identical holds should yield one representative per distinct shot"
    assert analysis["target"] >= 3, "Automatic target should grow to cover distinct scenes"
    assert len({candidates[index]["scene_id"] for index in analysis["selected_ids"]}) == 3
    assert all(instance.closed for instance in neutral_vision)


def test_black_tail_cannot_become_reserved_ending(tmp_path, ffmpeg, neutral_vision):
    frame = patterned_frames(1)[0]
    source = encode(ffmpeg, tmp_path / "ending.mkv", [frame] * 42 + [np.zeros_like(frame)] * 6)
    analysis = core.analyze_video(source, Options(output=tmp_path / "output"))
    assert analysis["end_card_id"] is not None and analysis["end_card_id"] < 42
    assert analysis["selected_ids"][-1] == analysis["end_card_id"]
    assert len(analysis["selected_ids"]) == 1
    assert any("target" in warning.lower() for warning in analysis["warnings"])


def test_corrupt_missing_and_invalid_options_fail_without_success_artifacts(tmp_path, ffmpeg, neutral_vision):
    corrupt = tmp_path / "broken.mp4"
    corrupt.write_bytes(b"This is not a video")
    output = tmp_path / "outputs"
    with pytest.raises((ValueError, RuntimeError)):
        core.analyze_video(corrupt, Options(output=output))
    with pytest.raises(FileNotFoundError):
        core.analyze_video(tmp_path / "missing.mp4", Options(output=output))
    with pytest.raises(ValueError, match="JPEG"):
        core.analyze_video(corrupt, Options(format="jpeg", bit_depth=16, output=output))
    assert not list(tmp_path.rglob("manifest.json"))


def test_cli_batch_continues_after_corrupt_and_missing_inputs(tmp_path, ffmpeg, neutral_vision, capsys):
    good = encode(ffmpeg, tmp_path / "good.mkv", patterned_frames(8))
    corrupt = tmp_path / "broken.mp4"
    corrupt.write_bytes(b"broken")
    missing = tmp_path / "missing.mp4"
    code = cli.main([
        "extract", str(corrupt), str(good), str(missing),
        "--output", str(tmp_path / "output"), "--count", "1", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["status"] == "failed"
    assert len(payload["results"]) == 1 and payload["results"][0]["source"] == str(good)
    assert len(payload["errors"]) == 2
    assert payload["results"][0]["selected_count"] == 1
    assert Path(payload["results"][0]["files"][0]["path"]).is_file()


def test_changed_source_and_unknown_or_empty_frame_selection_are_rejected(tmp_path, ffmpeg):
    source = encode(ffmpeg, tmp_path / "identity.mkv", patterned_frames(3))
    analysis = analysis_for(source, tmp_path / "analysis", [0, 2])
    for selection in ([], [999]):
        with pytest.raises(ValueError):
            core.export_analysis(analysis, Options(), frame_ids=selection)
    source.write_bytes(source.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="changed"):
        core.export_analysis(analysis, Options())
    assert not (Path(analysis["analysis_dir"]) / "manifest.json").exists()
    assert not list(Path(analysis["analysis_dir"]).glob("*_still_*"))


def test_cancellation_stops_analysis_and_closes_model(tmp_path, ffmpeg, neutral_vision):
    source = encode(ffmpeg, tmp_path / "cancel.mkv", patterned_frames(48))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    cancelled = threading.Event()

    def progress(stage, done, total):
        if stage == "Scanning scenes" and done > 0:
            cancelled.set()

    with pytest.raises(CancelledError):
        core.analyze_video(source, Options(output=tmp_path / "output"), progress, cancelled.is_set)
    assert neutral_vision and all(instance.closed for instance in neutral_vision)
    assert not list(tmp_path.rglob("manifest.json"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_cancellation_cleans_temporary_export_images(tmp_path, ffmpeg):
    source = encode(ffmpeg, tmp_path / "cancel-export.mkv", patterned_frames(6))
    analysis = analysis_for(source, tmp_path / "analysis", [0, 1, 2, 3, 4, 5])
    cancelled = threading.Event()

    def progress(stage, done, total):
        if stage.startswith("Exporting"):
            cancelled.set()

    with pytest.raises(CancelledError):
        core.export_analysis(analysis, Options(), progress=progress, cancel=cancelled.is_set)
    directory = Path(analysis["analysis_dir"])
    assert not (directory / "manifest.json").exists()
    assert not list(directory.glob(".export_*"))
    assert not list(directory.glob("*_still_*"))


def test_inflight_subprocess_cancellation_reaps_decoder(tmp_path, ffmpeg, monkeypatch):
    real_popen = subprocess.Popen
    processes = []

    def spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(core.subprocess, "Popen", spawn)
    began = time.monotonic()

    def cancel():
        return time.monotonic() - began > 0.1

    with pytest.raises(CancelledError):
        core._run([
            ffmpeg, "-v", "error", "-nostdin", "-re", "-f", "lavfi",
            "-i", "testsrc=size=64x48:rate=24", "-t", "5", "-f", "null", "-",
        ], cancel)
    assert processes and all(process.poll() is not None for process in processes)
    assert time.monotonic() - began < 3


def test_repeated_exports_allocate_collision_safe_runs(tmp_path, ffmpeg):
    source = encode(ffmpeg, tmp_path / "collision.mkv", patterned_frames(3))
    analysis = analysis_for(source, tmp_path / "analysis", [0, 2])
    first = core.export_analysis(analysis, Options())
    original = {item["path"]: hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() for item in first["files"]}
    second = core.export_analysis(analysis, Options(output=tmp_path / "runs"))
    third = core.export_analysis(analysis, Options(output=tmp_path / "runs"))
    assert len({first["output_dir"], second["output_dir"], third["output_dir"]}) == 3
    for path, digest in original.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert Path(second["output_dir"]).is_relative_to(tmp_path / "runs")
    assert Path(third["output_dir"]).is_relative_to(tmp_path / "runs")


def test_manual_export_honors_new_output_location_before_first_export(tmp_path, ffmpeg):
    source = encode(ffmpeg, tmp_path / "chosen-output.mkv", patterned_frames(2))
    analysis = analysis_for(source, tmp_path / "original-analysis", [0])
    chosen = tmp_path / "chosen-destination"
    result = core.export_analysis(analysis, Options(output=chosen))
    assert Path(result["output_dir"]).is_relative_to(chosen.resolve())
    assert not (Path(analysis["analysis_dir"]) / "manifest.json").exists()


def test_recursive_discovery_and_explicit_report_selection(tmp_path, ffmpeg):
    source = encode(ffmpeg, tmp_path / "first.mkv", patterned_frames(3))
    nested = tmp_path / "nested"
    nested.mkdir()
    second = encode(ffmpeg, nested / "second.mkv", patterned_frames(2))
    (nested / "readme.txt").write_text("ignore this non-video file", encoding="utf-8")
    assert core.discover_inputs([tmp_path, source]) == [source]
    assert set(core.discover_inputs([tmp_path], recursive=True)) == {source, second}
    analysis = analysis_for(source, tmp_path / "analysis", [0, 2])
    result = core.export_analysis(analysis, Options(report=True), frame_ids=[2])
    directory = Path(result["output_dir"])
    assert all((directory / name).is_file() for name in ("scores.html", "scores.csv", "scores.json"))
    summary = json.loads((directory / "scores.json").read_text())
    assert summary["selected_ids"] == [2]
    assert [row["selected"] for row in summary["candidates"]] == [False, True]
