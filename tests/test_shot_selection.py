"""Shot coverage is independent of duration, wording count and duplicate pixels."""
from pathlib import Path

import numpy as np

from stills_tool import core
from stills_tool.types import Options
from test_core import encode, ffmpeg, neutral_vision, patterned_frames, read_pixels


def candidate(index, shot, score=90, ready=True, eligible=True):
    return {"id": index, "frame_index": index, "shot_id": shot, "scene_id": shot,
            "timestamp": index / 24, "score": score, "text_ready": ready,
            "eligible": eligible, "metrics": {}, "reasons": [], "duplicate_of": None}


def test_automatic_selects_one_readable_frame_per_shot_without_padding():
    shots = [{"id": i} for i in range(3)]
    frames = [candidate(1, 0), candidate(2, 0, 100, ready=False),
              candidate(25, 1, 60), candidate(30, 1, 80), candidate(51, 2)]
    frames[-1]["duplicate_of"] = 1  # Repeated footage is still a separate shot.
    selected, target = core.select_shot_candidates(frames, shots)
    assert selected == [1, 30, 51] and target == 3
    assert all(s["selection_status"] == "recommended" for s in shots)


def test_bad_shot_is_flagged_and_preserved_instead_of_silently_dropped():
    shots = [{"id": 0}, {"id": 1}]
    frames = [candidate(4, 0), candidate(25, 1, 49, ready=False, eligible=False)]
    selected, target = core.select_shot_candidates(frames, shots)
    assert selected == [4, 25] and target == 2
    assert shots[1]["selection_status"] == "fallback"
    assert "review recommended" in frames[1]["reasons"][-1]


def test_selected_readable_fallback_reports_its_wording_even_if_overall_quality_is_low():
    frame = candidate(2, 0, 49, ready=True, eligible=False)
    frame["text_contents"] = ["complete title"]
    states = [{"id": 0, "candidate_ids": [2]}]
    core._record_text_coverage([frame], states, [2])
    assert states[0]["selected_frame_id"] == 2


def test_fallback_preserves_blink_and_transition_penalties_hidden_by_readability_cap():
    blink = candidate(1, 0, 49, ready=False, eligible=False)
    blink.update(quality_score=58)
    blink["metrics"]["focus_quality"] = .67
    clear = candidate(5, 0, 49, ready=False, eligible=False)
    clear.update(quality_score=79)
    clear["metrics"]["focus_quality"] = .65
    assert core.select_shot_candidates([blink, clear], [{"id": 0}])[0] == [5]


def test_settled_frame_beats_higher_score_from_scrolling_blur_without_ocr_words():
    moving = candidate(185, 0, 85)
    moving["selection_quality"] = {"soft": True}
    settled = candidate(214, 0, 79)
    settled["selection_quality"] = {"soft": False}
    assert core.select_shot_candidates([moving, settled], [{"id": 0}])[0] == [214]


def test_visible_open_face_beats_uncertain_face_with_slightly_higher_focus():
    hidden = candidate(123, 0, 49, ready=False, eligible=False)
    hidden.update(quality_score=78, selection_quality={"face_uncertain": True})
    visible = candidate(136, 0, 49, ready=False, eligible=False)
    visible.update(quality_score=75, selection_quality={"face_uncertain": False})
    assert core.select_shot_candidates([hidden, visible], [{"id": 0}])[0] == [136]


def test_scoring_and_selection_keep_open_eyes_when_both_wording_scores_are_capped():
    from test_focus_integration import candidate as focus_candidate, face, SCENE, VIDEO
    blink_face, open_face = face(), face()
    blink_face["open_eyes"] = .4
    open_face["open_eyes"] = .99
    frames = [focus_candidate(10, {"scene": 20, "face": 9, "text": None, "noise": 0}, faces=[blink_face]),
              focus_candidate(11, {"scene": 20, "face": 8, "text": None, "noise": 0}, faces=[open_face])]
    for frame in frames:
        frame["text_ready"] = False
    core._score_candidates(frames, [SCENE], VIDEO)
    assert [frame["score"] for frame in frames] == [49, 49]
    assert frames[1]["quality_score"] > frames[0]["quality_score"]
    assert core.select_shot_candidates(frames, [dict(SCENE)])[0] == [11]


def test_count_is_a_cap_on_shots_with_explicit_missing_coverage():
    shots = [{"id": i} for i in range(5)]
    frames = [candidate(i * 24, i) for i in range(5)]
    selected, target = core.select_shot_candidates(frames, shots, count=3)
    assert selected == [0, 48, 96] and target == 3
    assert core._shot_coverage(frames, shots, selected)["missing_shot_ids"] == [1, 3]
    assert shots[1]["selection_status"] == "omitted_count_limit"
    selected, target = core.select_shot_candidates(frames, shots, count=20)
    assert len(selected) == target == 5


def test_real_repeated_shot_and_two_frame_insert_both_export(tmp_path, ffmpeg, neutral_vision):
    first, second = patterned_frames(2)
    second = second[:, :, ::-1].copy()
    frames = [first] * 24 + [second] * 2 + [first] * 24
    source = encode(ffmpeg, tmp_path / "repeated-insert.mkv", frames)
    identity = core.fingerprint(source)
    analysis = core.analyze_video(source, Options(output=tmp_path / "output"))
    assert [(s["start_frame"], s["end_frame"]) for s in analysis["shots"]] == [(0, 23), (24, 25), (26, 49)]
    assert len(analysis["selected_ids"]) == analysis["target"] == 3
    assert analysis["schema_version"] == 2 and analysis["covered_shot_ids"] == [0, 1, 2]
    result = core.export_analysis(analysis, Options(report=True))
    assert [f["shot_id"] for f in result["files"]] == [0, 1, 2]
    assert result["missing_shot_ids"] == []
    pixels = [read_pixels(f["path"]) for f in result["files"]]
    np.testing.assert_array_equal(pixels[0], pixels[2])
    assert not np.array_equal(pixels[0], pixels[1])
    assert core.fingerprint(source) == identity
    assert all(Path(f["path"]).is_file() for f in result["files"])


def test_sharpest_shortlisted_moment_is_found_outside_initial_time_samples():
    records = [{"contrast": 40, "edge_fraction": .1, "screen_focus": 1.0} for _ in range(120)]
    records[53]["screen_focus"] = 9.0
    shots = [{"id": 0, "start_frame": 0, "end_frame": 119, "start": 0, "end": 5}]
    assert 53 in core._shot_shortlist(records, shots, {"fps": 24})
