"""Native subject focus must influence extraction decisions and reports."""

from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from stills_tool import core


SCENE = {"id": 0, "start_frame": 0, "end_frame": 99, "start": 0, "end": 100/24}
VIDEO = {"width": 768, "height": 512, "fps": 24}


def candidate(index, native, *, faces=None, text=None, state=None):
    return {
        "id": index, "frame_index": index, "scene_id": 0, "timestamp": index/24,
        "text_state_id": state, "text_ready": True,
        "duplicate_of": None,
        "metrics": {"sharpness": 200, "mean_luma": 110, "contrast": 40,
                    "clipped_fraction": 0, "dark_fraction": 0, "edge_fraction": 0.05,
                    "native_focus": native},
        "vision": {"faces": faces or [], "text": text or []},
    }


def face():
    return {"box": [280, 128, 208, 208], "confidence": 1,
            "open_eyes": 0.95, "smile": 0.1, "sharpness": 100}


def text_region(wording="ALL THE WORDS"):
    return {"box": [100, 80, 400, 60], "confidence": 0.95,
            "recognition_confidence": 0.95, "contrast": 0.9,
            "text": wording, "sharpness": 100}


def score(candidates):
    core._score_candidates(candidates, [SCENE], VIDEO)


def test_two_high_focus_candidates_do_not_tie_from_percentile_saturation():
    # Legacy/global values are identical and would both saturate the old term.
    # A subtle native advantage must still survive into final ordering.
    candidates = [candidate(10, {"scene": 18, "face": None, "text": None, "noise": 0}),
                  candidate(11, {"scene": 20, "face": None, "text": None, "noise": 0})]
    score(candidates)
    assert candidates[1]["score"] > candidates[0]["score"]
    assert candidates[1]["metrics"]["focus_quality"] > candidates[0]["metrics"]["focus_quality"]
    selected, _ = core.select_candidates(candidates, [SCENE], 1)
    assert selected == [11]


@pytest.mark.parametrize("state", [None, 0])
def test_rounded_display_scores_keep_finer_native_focus_for_selection(state):
    candidates = [candidate(10, {"scene": 18.000, "face": None, "text": None, "noise": 0}, state=state),
                  candidate(11, {"scene": 18.001, "face": None, "text": None, "noise": 0}, state=state)]
    score(candidates)
    assert candidates[0]["score"] == candidates[1]["score"]
    assert candidates[1]["metrics"]["focus_quality"] > candidates[0]["metrics"]["focus_quality"]
    assert core.select_candidates(candidates, [SCENE], 1)[0] == [11]


def test_sharper_face_wins_with_equally_detailed_backgrounds():
    candidates = [candidate(10, {"scene": 20, "face": 5, "text": None, "noise": 0}, faces=[face()]),
                  candidate(11, {"scene": 20, "face": 17, "text": None, "noise": 0}, faces=[face()])]
    score(candidates)
    assert candidates[1]["score"] > candidates[0]["score"]
    assert core.select_candidates(candidates, [SCENE], 1)[0] == [11]


def test_native_readable_text_focus_selects_the_sharper_same_wording_state():
    candidates = [candidate(10, {"scene": 20, "face": None, "text": 8, "noise": 0},
                            text=[text_region()], state=0),
                  candidate(11, {"scene": 20, "face": None, "text": 18, "noise": 0},
                            text=[text_region()], state=0)]
    score(candidates)
    assert candidates[1]["score"] > candidates[0]["score"]
    assert core.select_candidates(candidates, [SCENE], 1)[0] == [11]


def test_different_wording_density_does_not_define_another_states_reference():
    original = [candidate(10, {"scene": 18, "face": None, "text": 8, "noise": 0},
                          text=[text_region("SHORT")], state=0),
                candidate(11, {"scene": 20, "face": None, "text": 12, "noise": 0},
                          text=[text_region("SHORT")], state=0)]
    score(original)
    original_scores = [item["score"] for item in original]
    expanded = deepcopy(original) + [
        candidate(index, {"scene": 90, "face": None, "text": 90, "noise": 0},
                  text=[text_region("A MUCH LONGER AND DENSER HEADLINE")], state=1)
        for index in (20, 21, 22)
    ]
    score(expanded)
    assert [item["score"] for item in expanded[:2]] == original_scores


def busy_background_pair():
    y, x = np.indices((512, 768))
    tiles = ((x//16 + y//16) % 2 * 210 + 20).astype(np.uint8)
    frame = np.repeat(tiles[:, :, None], 3, axis=2)
    subject = np.full((208, 208, 3), 75, np.uint8)
    cv2.ellipse(subject, (104, 104), (75, 90), 0, 0, 360, (155, 175, 195), -1, cv2.LINE_AA)
    cv2.circle(subject, (75, 80), 10, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(subject, (133, 80), 10, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.ellipse(subject, (104, 123), (31, 25), 0, 15, 165, (15, 15, 15), 3, cv2.LINE_AA)
    sharp = frame.copy()
    sharp[128:336, 280:488] = subject
    soft = frame.copy()
    soft[128:336, 280:488] = cv2.GaussianBlur(subject, (0, 0), 4)
    return soft, sharp


def test_native_face_roi_survives_busy_background_and_controls_ranking():
    candidates = []
    for index, frame in zip((10, 11), busy_background_pair()):
        faces = [face()]
        measured = core._native_focus(frame, faces, [])
        candidates.append(candidate(index, measured, faces=faces))
        report = faces[0]["focus_evidence"]
        assert report["measurement_size"] == [208, 208]
        assert measured["measurement"] == "native source pixels"
        assert len(measured["tiles"]) == 9
        assert all(component in report for component in ("laplacian", "tenengrad", "edge_acutance", "noise"))
        # Report values remain JSON-compatible without custom array encoders.
        json.dumps({"native_focus": measured, "face": faces[0]})
    assert candidates[1]["metrics"]["native_focus"]["face"] > candidates[0]["metrics"]["native_focus"]["face"]
    score(candidates)
    assert candidates[1]["score"] > candidates[0]["score"]


def test_unreliable_face_regions_cannot_control_native_subject_quality():
    _, frame = busy_background_pair()
    unreliable = face()
    unreliable["confidence"] = 0
    measurement = core._native_focus(frame, [unreliable], [])
    assert measurement["face"] is None, "Tiny/cropped unreliable face detections must stay neutral"


def test_native_text_rois_keep_fine_detail_instead_of_resizing_to_128_pixels():
    frame = np.full((512, 768, 3), 75, np.uint8)
    cv2.putText(frame, "WORDING 123", (100, 235), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (220, 220, 220), 1, cv2.LINE_AA)
    box = [90, 200, 310, 55]
    sharp_regions = [dict(text_region(), box=box)]
    soft_regions = [dict(text_region(), box=box)]
    sharp = core._native_focus(frame, [], sharp_regions)
    soft = core._native_focus(cv2.GaussianBlur(frame, (0, 0), 2), [], soft_regions)
    assert sharp["text"] > soft["text"]
    assert sharp_regions[0]["focus_evidence"]["measurement_size"] == [310, 55]
    assert soft_regions[0]["focus_evidence"]["measurement_size"] == [310, 55]


def test_candidate_pipeline_measures_source_rois_before_writing_lossy_previews(tmp_path, monkeypatch):
    soft, sharp = busy_background_pair()
    (tmp_path / "previews").mkdir()
    def source_frames(*args):
        yield 10, soft
        yield 11, sharp
    monkeypatch.setattr(core, "_source_frames", source_frames)

    class Analyzer:
        def analyze(self, frame):
            assert frame.shape == (256, 384, 3)
            half = face()
            half["box"] = [140, 64, 104, 104]
            return {"faces": [half], "text": [], "warnings": []}

    video = dict(VIDEO, frames=[{"timestamp": index/24, "pts": index} for index in range(100)])
    screening = core._cheap_metrics(cv2.resize(sharp, (384, 256)))
    records = [deepcopy(screening) for _ in range(100)]
    candidates = core._analyze_native_candidates(None, video, [10, 11], records, [SCENE],
                                                (384, 256), tmp_path, Analyzer(), None, None)
    assert len(candidates) == 2
    for item in candidates:
        face_report = item["vision"]["faces"][0]
        assert face_report["box"] == [140, 64, 104, 104]
        assert face_report["focus_evidence"]["measurement_size"] == [208, 208]
        assert (tmp_path / item["preview"]).is_file()
    assert candidates[1]["metrics"]["native_focus"]["face"] > candidates[0]["metrics"]["native_focus"]["face"]


def test_unreadable_words_are_not_rescued_by_a_high_focus_number():
    values = [candidate(10, {"scene": 200, "face": None, "text": 200, "noise": 0},
                        text=[text_region("FRAGM")], state=None)]
    values[0]["text_ready"] = False
    values[0]["text_reason"] = "Words are incomplete"
    score(values)
    assert not values[0]["eligible"]
    assert core.select_candidates(values, [SCENE], 6)[0] == []
