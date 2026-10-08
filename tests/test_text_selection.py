"""Selection outcomes and real OCR/export coverage for changing wording."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stills_tool import core
from stills_tool.text_states import assign_text_states
from stills_tool.types import Options


def reading(text, box=None):
    return {
        "text": text, "recognition_confidence": 0.99, "confidence": 0.99,
        "contrast": 0.9, "clipped": False, "complete_geometry": True,
        "box": box or [100, 180, max(100, len(text) * 12), 35],
    }


def candidate(index, time, text, *, logo=False, scene_id=0, score=90):
    texts = [reading(text)]
    if logo:
        texts.insert(0, reading("Logo", [40, 40, 400, 70]))
        texts[-1]["box"] = [100, 460, max(100, len(text) * 12), 12]
    return {
        "id": index, "frame_index": index, "timestamp": time, "scene_id": scene_id,
        "score": score, "tier": "High", "eligible": True, "duplicate_of": None, "reasons": [],
        "vision": {"text": texts, "faces": []},
        "metrics": {"sharpness": 120, "text_stability": 1.0, "open_eyes": 1.0},
    }


def wording_candidates(values, *, logo=False):
    candidates = []
    for state, text in enumerate(values):
        for offset in range(2):
            index = state * 12 + offset * 6
            candidates.append(candidate(index, state * 0.5 + offset * 0.25, text,
                                        logo=logo, score=90 - offset))
    scenes = [{"id": 0, "start": 0, "end": len(values) * 0.5,
               "start_frame": 0, "end_frame": len(values) * 12 - 1}]
    states = assign_text_states(candidates, scenes, {"width": 960, "height": 540, "fps": 24})
    return candidates, scenes, states


def test_five_complete_phrases_in_one_shot_grow_automatic_target():
    candidates, scenes, states = wording_candidates([
        "ALPHA MESSAGE", "BRAVO COPY", "CLEAR CHOICE", "DOCTOR RECOMMENDED", "FINAL WORDING",
    ])
    selected, target = core.select_candidates(candidates, scenes, base_target=2)
    chosen = {candidate["text_state_id"] for candidate in candidates if candidate["id"] in selected}
    assert target == 5 and len(selected) == 5
    assert chosen == {state["id"] for state in states}


def test_persistent_logo_cannot_hide_a_smaller_subtitle_update():
    values = ["First subtitle", "Second subtitle", "Third subtitle"]
    candidates, scenes, states = wording_candidates(values, logo=True)
    selected, target = core.select_candidates(candidates, scenes, base_target=1)
    assert target == len(selected) == 3
    assert {candidate["text_state_id"] for candidate in candidates if candidate["id"] in selected} == {
        state["id"] for state in states
    }


def test_one_word_and_digit_changes_all_receive_a_selection():
    candidates, scenes, _states = wording_candidates(["SALE", "SAFE", "PRICE $25", "PRICE $75"])
    selected, target = core.select_candidates(candidates, scenes, base_target=2)
    wording = {tuple(candidate["text_contents"]) for candidate in candidates if candidate["id"] in selected}
    assert wording == {("sale",), ("safe",), ("price $25",), ("price $75",)}
    assert target == len(selected) == 4


def test_explicit_count_remains_a_cap_even_with_five_complete_phrases():
    candidates, scenes, _states = wording_candidates(["ALPHA", "BRAVO", "CLEAR", "DELTA", "ECHO"])
    end_id = candidates[-1]["id"]
    selected, target = core.select_candidates(candidates, scenes, base_target=2, count=2, end_id=end_id)
    assert target == 2 and len(selected) <= 2
    assert end_id == selected[-1]


def test_incomplete_text_is_excluded_even_if_raw_score_and_eligibility_are_stale():
    candidates, scenes, _states = wording_candidates(["COMPLETE ALPHA", "COMPLETE BRAVO"])
    bad = candidate(100, 2.0, "INCOMPL", score=100)
    bad.update(text_ready=False, text_contents=["incompl"], text_state_id=99,
               text_reason="Detected word crop may omit letters", eligible=True)
    candidates.append(bad)
    selected, target = core.select_candidates(candidates, scenes, base_target=2, end_id=100)
    assert 100 not in selected
    assert target == len(selected) == 2


def test_scene_and_word_coverage_share_frames_instead_of_adding_two_quotas():
    candidates = []
    scenes = []
    for scene_id, text in enumerate(["ALPHA", "BRAVO", "CLEAR"]):
        scenes.append({"id": scene_id, "start": scene_id, "end": scene_id + 1,
                       "start_frame": scene_id * 24, "end_frame": scene_id * 24 + 23})
        for offset in (4, 10):
            candidates.append(candidate(scene_id * 24 + offset, scene_id + offset / 24, text, scene_id=scene_id))
    states = assign_text_states(candidates, scenes, {"width": 960, "height": 540, "fps": 24})
    selected, target = core.select_candidates(candidates, scenes, base_target=2)
    assert target == len(selected) == 3
    assert {item["scene_id"] for item in candidates if item["id"] in selected} == {0, 1, 2}
    assert len(states) == 3


def test_exact_repeated_alpha_can_cover_two_occurrences_with_one_image():
    candidates, scenes, states = wording_candidates(["ALPHA", "BRAVO", "ALPHA"])
    canonical_a = candidates[0]["id"]
    canonical_b = candidates[2]["id"]
    for item in candidates:
        if item["text_contents"] == ["alpha"] and item["id"] != canonical_a:
            item["duplicate_of"] = canonical_a
        if item["text_contents"] == ["bravo"] and item["id"] != canonical_b:
            item["duplicate_of"] = canonical_b
    selected, target = core.select_candidates(candidates, scenes, base_target=2)
    core._record_text_coverage(candidates, states, selected)
    assert selected == [canonical_a, canonical_b] and target == 2
    assert len(states) == 3
    assert all(state["selected_frame_id"] in selected for state in states)
    assert states[2]["selected_frame_id"] == canonical_a
    assert states[2]["covered_by_duplicate"]


def make_real_word_video(directory: Path, words: tuple[str, ...]) -> tuple[Path, list[np.ndarray]]:
    ffmpeg = core.find_tool("ffmpeg")
    source = directory / "wording.mkv"
    images = []
    for word in words:
        image = np.full((1080, 1920, 3), 235, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, height), _baseline = cv2.getTextSize(word, font, 4, 10)
        cv2.putText(image, word, ((1920 - width) // 2, (1080 + height) // 2),
                    font, 4, (15, 15, 15), 10, cv2.LINE_AA)
        images.append(image)
    command = [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", "1920x1080", "-framerate", "25", "-i", "pipe:0",
        "-vf", "setparams=range=full:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=gbr",
        "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for image in images:
            data = image.tobytes()
            for _ in range(10):  # Exactly 0.4 seconds per complete title.
                process.stdin.write(data)
        process.stdin.close()
        errors = process.stderr.read().decode("utf-8", "replace")
        process.wait(timeout=30)
        assert process.returncode == 0, errors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    return source, images


def assert_export_pixels(result: dict, images: list[np.ndarray]) -> None:
    for file in result["files"]:
        image = cv2.imread(file["path"], cv2.IMREAD_UNCHANGED)
        assert image is not None and image.shape == (1080, 1920, 3)
        source_index = file["frame_index"] // 10
        np.testing.assert_allclose(image, images[source_index], atol=1)


def test_real_native_ocr_three_short_titles_exceed_duration_base_count(tmp_path):
    source, images = make_real_word_video(tmp_path, ("ALPHA", "BRAVO", "CLEAR"))
    analysis = core.analyze_video(source, Options(output=tmp_path / "output"))
    assert len(analysis["scenes"]) == 1, "Wording changes must be covered inside one continuous shot"
    assert analysis["base_target"] == 2
    assert [state["text"] for state in analysis["text_states"]] == [["alpha"], ["bravo"], ["clear"]]
    assert len(analysis["selected_ids"]) == analysis["target"] == 3
    result = core.export_analysis(analysis, Options(report=True))
    assert result["selected_count"] == 3
    assert {tuple(file["recognized_text"]) for file in result["files"]} == {("alpha",), ("bravo",), ("clear",)}
    assert all(file["text_ready"] for file in result["files"])
    assert result["files"][-1]["recognized_text"] == ["clear"], "The ending still must retain the final complete wording"
    assert_export_pixels(result, images)


def test_real_native_ocr_repeated_title_reuses_exact_pixels_and_reports_coverage(tmp_path):
    source, images = make_real_word_video(tmp_path, ("ALPHA", "BRAVO", "ALPHA"))
    analysis = core.analyze_video(source, Options(output=tmp_path / "output"))
    assert [state["text"] for state in analysis["text_states"]] == [["alpha"], ["bravo"], ["alpha"]]
    result = core.export_analysis(analysis, Options(report=True))
    assert result["selected_count"] == 2
    pixels = [cv2.imread(file["path"], cv2.IMREAD_UNCHANGED) for file in result["files"]]
    assert len({hashlib.sha256(image.tobytes()).hexdigest() for image in pixels}) == 2
    assert {state for file in result["files"] for state in file["text_state_ids"]} == {0, 1, 2}
    assert all(state["selected_frame_id"] is not None for state in analysis["text_states"])
    assert any(state["covered_by_duplicate"] for state in analysis["text_states"])
    scores = json.loads((Path(result["output_dir"]) / "scores.json").read_text())
    assert len(scores["text_states"]) == 3
    assert all(state["selected_frame_id"] is not None for state in scores["text_states"])
    assert_export_pixels(result, images)
