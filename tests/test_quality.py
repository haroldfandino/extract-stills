"""Behavioral quality regressions and small offline model acceptance checks."""

from pathlib import Path
from types import SimpleNamespace
import shutil
import socket
import subprocess

import cv2
import numpy as np
import pytest

from stills_tool import core
from stills_tool.vision import LocalVision, MissingModelsError


def title_frame(title="BEST STILLS 2.0", shade=220):
    image = np.full((540, 960, 3), 80, dtype=np.uint8)
    cv2.putText(image, title, (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                (shade, shade, shade), 4, cv2.LINE_AA)
    return image


def text_detection(x=80, contrast=0.9):
    return {"faces": [], "text": [{"box": [x, 100, 240, 40], "contrast": contrast,
                                     "confidence": 0.95, "sharpness": 80}]}


def candidate(index, frame, scene_id=0, vision=None):
    return {"id": index, "frame_index": index, "scene_id": scene_id,
            "timestamp": index / 24, "metrics": core._cheap_metrics(frame),
            "vision": vision or {"faces": [], "text": []},
            "duplicate_of": None, "preview": f"frame_{index}.jpg"}


def score(candidates, scene_id=0):
    scene = {"id": scene_id, "start_frame": 0, "end_frame": 100}
    core._score_candidates(candidates, [scene], {"width": 960, "height": 540})
    return scene


@pytest.fixture(scope="module")
def vision():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    if not (model_dir / "face_landmarker.task").exists() or not (model_dir / "pp_ocrv5_mobile_det.onnx").exists():
        pytest.skip("Provision local models with python scripts/setup_models.py")
    with LocalVision(model_dir) as analyzer:
        yield analyzer


def test_global_sharpness_prefers_legible_title_over_motion_blur():
    clear = title_frame()
    smeared = cv2.GaussianBlur(clear, (0, 0), 8)
    assert core.sharpness(clear) > core.sharpness(smeared) * 5


def test_an_entirely_blurry_scene_cannot_be_promoted_by_relative_ranking():
    # Every frame is badly out of focus. The sharpest member is still unusable;
    # no clear reference in this scene should not turn it into a good still.
    images = [cv2.GaussianBlur(title_frame(), (0, 0), sigma) for sigma in (12, 15, 18)]
    candidates = [candidate(index + 10, image) for index, image in enumerate(images)]
    scene = score(candidates)
    selected, _ = core.select_candidates(candidates, [scene], 6)
    assert all(item["tier"] != "High" for item in candidates)
    assert selected == [], "Quality-first extraction must leave a shortfall instead of exporting severe blur"


def test_blank_frame_is_not_export_eligible():
    candidates = [candidate(10, np.zeros((540, 960, 3), dtype=np.uint8))]
    score(candidates)
    assert not candidates[0]["eligible"]


def test_open_eyes_outweigh_a_large_smile_during_a_blink():
    face = {"box": [300, 100, 160, 200], "confidence": 1, "sharpness": 80}
    opened = {"faces": [dict(face, open_eyes=0.97, smile=0.02)], "text": []}
    blinking = {"faces": [dict(face, open_eyes=0.05, smile=0.95)], "text": []}
    candidates = [candidate(10, title_frame(), vision=opened), candidate(11, title_frame(), vision=blinking)]
    score(candidates)
    assert candidates[0]["score"] > candidates[1]["score"]
    assert any("blink" in reason.lower() for reason in candidates[1]["reasons"])


def test_unreliable_face_does_not_penalize_a_good_frame():
    tiny = {"faces": [{"box": [10, 10, 12, 12], "confidence": 0,
                       "open_eyes": 0, "smile": 0, "sharpness": 0}], "text": []}
    candidates = [candidate(10, title_frame()), candidate(11, title_frame(), vision=tiny)]
    score(candidates)
    assert candidates[0]["score"] == candidates[1]["score"]
    assert candidates[1]["eligible"]


def test_settled_text_outscores_text_still_entering_the_frame():
    settled = core._text_stability(text_detection(), [text_detection(), text_detection()], (540, 960))
    moving = core._text_stability(text_detection(80), [text_detection(0), text_detection(170)], (540, 960))
    assert settled > 0.9
    assert moving < 0.7
    assert settled > moving


def test_fading_text_uses_the_detectors_normalized_contrast_scale():
    fading = core._text_stability(text_detection(contrast=0.15),
                                  [text_detection(contrast=0.1), text_detection(contrast=0.95)], (540, 960))
    settled = core._text_stability(text_detection(contrast=0.95),
                                   [text_detection(contrast=0.9), text_detection(contrast=0.95)], (540, 960))
    assert settled > 0.9
    assert fading < 0.7, "A major contrast change must not be classified as settled text"


def test_changed_small_title_is_not_removed_as_a_near_duplicate():
    first = np.full((1080, 1920, 3), 100, np.uint8)
    second = first.copy()
    cv2.putText(first, "$25", (1480, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    cv2.putText(second, "$75", (1480, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    assert not core._near_match(first, second)
    assert core._near_match(first, first.copy())


def test_duplicate_aliases_keep_zero_as_the_canonical_representative(tmp_path, monkeypatch):
    candidates = [candidate(index, title_frame()) for index in range(3)]
    for item in candidates:
        item.update(eligible=True, score=100 - item["id"], reasons=[])
        item["metrics"]["open_eyes"] = 1
    cv2.imwrite(str(tmp_path / candidates[0]["preview"]), title_frame())
    monkeypatch.setattr(core, "_native_hashes", lambda *args: {0: "same", 1: "same", 2: "same"})
    core._deduplicate(candidates, None, None, tmp_path, None, None, None)
    assert [item["duplicate_of"] for item in candidates] == [None, 0, 0]


def test_automatic_selection_covers_each_worthwhile_scene_and_keeps_end_card_last():
    candidates = []
    scenes = []
    for scene_id in range(9):
        scenes.append({"id": scene_id})
        item = candidate(scene_id * 24, title_frame(), scene_id)
        item.update(eligible=True, score=90)
        candidates.append(item)
    selected, target = core.select_candidates(candidates, scenes, 6, end_id=192)
    assert target == 9
    assert len(selected) == 9
    assert selected[-1] == 192
    limited, target = core.select_candidates(candidates, scenes, 6, count=3, end_id=192)
    assert target == len(limited) == 3
    assert limited[-1] == 192


@pytest.mark.parametrize("duration,expected", [(6, 6), (15, 10), (30, 15), (45, 20)])
def test_duration_targets_tolerate_fractional_frame_rate_timing(duration, expected):
    assert core.duration_target(duration + 0.5 / 23.976, fps=23.976) == expected


def test_missing_models_fail_before_any_runtime_network_operation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Vision startup attempted network access")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with pytest.raises(MissingModelsError, match="setup_models.py"):
        LocalVision(tmp_path / "absent-models")


def test_real_local_text_detector_detects_title_and_measures_its_blur(vision):
    clear = title_frame()
    sharp_regions = vision.analyze(clear)["text"]
    blurred_regions = vision.analyze(cv2.GaussianBlur(clear, (0, 0), 3))["text"]
    assert sharp_regions and blurred_regions
    assert max(item["sharpness"] for item in sharp_regions) > max(item["sharpness"] for item in blurred_regions)
    assert all(0 <= item["confidence"] <= 1 and 0 <= item["contrast"] <= 1 for item in sharp_regions)
    assert all(0 <= x < 960 and 0 <= y < 540 for item in sharp_regions for x, y in item["polygon"])
    assert vision.analyze(np.zeros_like(clear))["text"] == []


def test_legible_text_region_can_rescue_a_frame_with_little_global_detail(vision):
    # A sparse title has little edge energy averaged over the whole canvas.
    # Absolute global blur gating must still respect a sharp detected subject.
    image = cv2.GaussianBlur(title_frame(), (0, 0), 3)
    detected = vision.analyze(image)
    assert detected["text"]
    candidates = [candidate(index, image, vision=detected) for index in (10, 11, 12)]
    score(candidates)
    assert any(item["eligible"] for item in candidates)


def test_tiny_face_landmarks_receive_neutral_expression_scores():
    # Exercise the public contract with an unreliable tiny detection instead of
    # relying on a pretrained detector to repeatedly find a five-pixel face.
    analyzer = LocalVision.__new__(LocalVision)
    analyzer._mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    coefficients = [SimpleNamespace(category_name=name, score=value) for name, value in
                    [("eyeBlinkLeft", 1), ("eyeBlinkRight", 1), ("mouthSmileLeft", 0), ("mouthSmileRight", 0)]]
    result = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=.5, y=.5), SimpleNamespace(x=.51, y=.51)]],
                             face_blendshapes=[coefficients])
    analyzer._landmarker = SimpleNamespace(detect=lambda image: result)
    warnings = []
    faces = analyzer._faces(title_frame(), cv2.cvtColor(title_frame(), cv2.COLOR_BGR2GRAY), warnings)
    assert len(faces) == 1
    assert faces[0]["confidence"] == 0
    assert faces[0]["open_eyes"] == faces[0]["smile"] == 0.5
    assert warnings


def test_native_ten_bit_hashes_retain_changes_lost_in_eight_bit_previews(tmp_path):
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("FFmpeg not available for the native precision fixture")
    width = height = 64
    samples = width * height * 3 // 2
    first = np.full(samples, 512, dtype="<u2")
    changed = first.copy()
    changed[100] += 1  # One native 10-bit LSB, invisible after uint8 conversion.
    source = tmp_path / "native_10bit.mkv"
    subprocess.run([executable, "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "yuv420p10le",
                    "-video_size", "64x64", "-framerate", "1", "-i", "pipe:0", "-c:v", "ffv1", "-level", "3",
                    "-pix_fmt", "yuv420p10le", "-color_primaries", "bt709", "-colorspace", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv", str(source)],
                   input=first.tobytes() * 2 + changed.tobytes(), check=True, capture_output=True)
    video = core.probe_video(source)
    assert video["pix_fmt"] == "yuv420p10le"
    hashes = core._native_hashes(source, video, [0, 1, 2], tmp_path, None)
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    assert np.array_equal(core._source_frame(source, video, 0), core._source_frame(source, video, 2))


def test_scan_finds_distinct_shots_despite_camera_motion_inside_each_shot(tmp_path):
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("FFmpeg not available for scene detection fixture")
    frames = []
    for color in ((180, 55, 45), (40, 175, 50), (45, 55, 180)):
        scene = np.full((96, 160, 3), color, dtype=np.uint8)
        for x in range(10, 160, 25):
            cv2.rectangle(scene, (x, 15), (x+8, 75), (225, 225, 225), -1)
        for frame_index in range(30):
            frames.append(np.roll(scene, frame_index, axis=1))
    source = tmp_path / "three_shots.mkv"
    subprocess.run([executable, "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
                    "-video_size", "160x96", "-framerate", "30", "-i", "pipe:0", "-c:v", "ffv1",
                    "-pix_fmt", "yuv444p", "-color_primaries", "bt709", "-colorspace", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv", str(source)],
                   input=b"".join(frame.tobytes() for frame in frames), check=True, capture_output=True)
    video = core.probe_video(source)
    indexes, _, scenes, _ = core._decode_scan(source, video, tmp_path, None, None)
    boundaries = [scene["start_frame"] for scene in scenes[1:]]
    assert any(abs(index-30) <= 2 for index in boundaries)
    assert any(abs(index-60) <= 2 for index in boundaries)
    assert len(scenes) <= 5, "Smooth camera movement should not become a sequence of artificial cuts"
    assert all(any(scene["start_frame"] <= index <= scene["end_frame"] for index in indexes)
               for scene in scenes)


def test_real_local_face_model_detects_two_prominent_faces(vision):
    source = Path(r"D:\Filmkraft\Tools\Extract_stills\input\Fatty15_RickiLake2_Amazon_15_HD_Social_H264.mp4")
    executable = shutil.which("ffmpeg")
    if not source.exists() or not executable:
        pytest.skip("User-provided video unavailable for face acceptance fixture")
    pixels = subprocess.check_output([executable, "-v", "error", "-ss", "1.5", "-i", str(source),
                                      "-vf", "scale=960:540", "-frames:v", "1", "-f", "rawvideo",
                                      "-pix_fmt", "bgr24", "pipe:1"])
    frame = np.frombuffer(pixels, np.uint8).reshape(540, 960, 3)
    first = vision.analyze(frame)["faces"]
    assert first
    x, y, w, h = first[0]["box"]
    crop = frame[max(0, y-h//4):min(540, y+h+h//4), max(0, x-w//4):min(960, x+w+w//4)]
    pair = vision.analyze(np.concatenate([crop, crop], axis=1))["faces"]
    assert len(pair) == 2
    assert all(face["confidence"] > 0 for face in pair)
    assert all(0 <= face["open_eyes"] <= 1 and 0 <= face["smile"] <= 1 for face in pair)
