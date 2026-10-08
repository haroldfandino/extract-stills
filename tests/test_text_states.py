"""Temporal text fixtures retain wording changes and reject incomplete moments."""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stills_tool.text_states import assign_text_states, normalize_text


def region(text, confidence=0.97, *, box=None, clipped=False, contrast=0.90):
    return {
        "text": text, "recognition_confidence": confidence, "confidence": 0.95,
        "box": box or [100, 160, max(35, len(text) * 15), 35],
        "clipped": clipped, "contrast": contrast,
    }


def candidate(index, time, regions, *, stability=1.0):
    return {
        "id": index, "frame_index": index, "timestamp": time, "scene_id": 0,
        "score": 90, "eligible": True, "duplicate_of": None,
        "vision": {"text": regions}, "metrics": {"sharpness": 120, "text_stability": stability},
    }


def assign(candidates, *, fps=24):
    end = max((item["timestamp"] for item in candidates), default=0) + 1 / fps
    scenes = [{"id": 0, "start": 0.0, "end": end, "start_frame": 0,
               "end_frame": max((item["frame_index"] for item in candidates), default=0)}]
    return assign_text_states(candidates, scenes, {"width": 960, "height": 540, "fps": fps})


def test_alpha_bravo_alpha_keeps_every_continuous_occurrence():
    values = ["ALPHA"] * 3 + ["BRAVO"] * 3 + ["ALPHA"] * 3
    candidates = [candidate(index, index * 0.25, [region(text)]) for index, text in enumerate(values)]
    states = assign(candidates)
    assert [state["text"] for state in states] == [["alpha"], ["bravo"], ["alpha"]]
    assert [item["text_state_id"] for item in candidates] == [0] * 3 + [1] * 3 + [2] * 3
    assert all(item["text_ready"] for item in candidates)
    assert [state["candidate_ids"] for state in states] == [list(range(3)), list(range(3, 6)), list(range(6, 9))]


def test_persistent_logo_does_not_hide_changing_small_subtitle():
    candidates = []
    for index, text in enumerate(["First wording"] * 2 + ["Second wording"] * 2):
        logo = region("FILMKRAFT", box=[60, 30, 430, 90])
        subtitle = region(text, box=[200, 465, 150, 12])
        candidates.append(candidate(index, index * 0.25, [logo, subtitle]))
    states = assign(candidates)
    assert len(states) == 2
    assert states[0]["text"] == ["filmkraft", "first wording"]
    assert states[1]["text"] == ["filmkraft", "second wording"]
    assert all(item["text_ready"] for item in candidates)


def test_typewriter_prefixes_are_not_complete_text_states():
    values = [("H", 0), ("HE", 0.05), ("HEL", 0.1), ("HELL", 0.15),
              ("HELLO", 0.20), ("HELLO", 0.30), ("HELLO", 0.40)]
    candidates = [candidate(index, time, [region(text)]) for index, (text, time) in enumerate(values)]
    states = assign(candidates)
    assert [state["text"] for state in states] == [["hello"]]
    assert all(not item["text_ready"] and item["text_state_id"] is None for item in candidates[:4])
    assert all(item["text_ready"] for item in candidates[4:])
    assert "fragment" in candidates[2]["text_reason"]


def test_short_confirmed_whole_word_change_and_price_digit_change_are_preserved():
    values = ["PRICE $25", "PRICE $25", "PRICE $75", "PRICE $75"]
    candidates = [candidate(index, index * 0.04, [region(text)]) for index, text in enumerate(values)]
    states = assign(candidates)
    assert [state["text"] for state in states] == [["price $25"], ["price $75"]]
    assert all(item["text_ready"] for item in candidates)
    words = [candidate(index, index * 0.04, [region(text)])
             for index, text in enumerate(["SALE", "SALE", "SAFE", "SAFE"])]
    assert [state["text"] for state in assign(words)] == [["sale"], ["safe"]]


def test_meaningful_plural_change_is_not_inferred_to_be_typewriting():
    candidates = [candidate(index, index * 0.04, [region(text)])
                  for index, text in enumerate(["DOG", "DOG", "DOGS", "DOGS"])]
    assert [state["text"] for state in assign(candidates)] == [["dog"], ["dogs"]]
    assert all(item["text_ready"] for item in candidates)


def test_prefix_that_is_held_long_enough_can_be_complete_wording():
    values = [("SALE", 0), ("SALE", 0.15), ("SALE NOW", 0.30), ("SALE NOW", 0.45)]
    candidates = [candidate(index, time, [region(text)]) for index, (text, time) in enumerate(values)]
    assert [state["text"] for state in assign(candidates)] == [["sale"], ["sale now"]]


def test_short_rolling_caption_fragments_wait_for_complete_pause():
    values = [
        ("A GREAT PRODUCT", 0), ("A GREAT PRODUCT", 0.04),
        ("GREAT PRODUCT FOR", 0.08), ("GREAT PRODUCT FOR", 0.12),
        ("PRODUCT FOR YOU", 0.16), ("PRODUCT FOR YOU", 0.20),
        ("A GREAT PRODUCT FOR YOU", 0.24), ("A GREAT PRODUCT FOR YOU", 0.40),
    ]
    candidates = [candidate(index, time, [region(text, box=[90, 160, 380, 35])])
                  for index, (text, time) in enumerate(values)]
    states = assign(candidates)
    assert [state["text"] for state in states] == [["a great product for you"]]
    assert all(not item["text_ready"] for item in candidates[:6])
    assert all(item["text_ready"] for item in candidates[6:])


def test_weak_alphabetic_ocr_flicker_uses_consensus_but_not_as_legibility_proof():
    candidates = [
        candidate(0, 0, [region("BRAVO")]),
        candidate(1, 0.10, [region("BRAVO")]),
        candidate(2, 0.20, [region("BRAVQ", confidence=0.62)]),
        candidate(3, 0.30, [region("BRAVO")]),
        candidate(4, 0.40, [region("BRAVO")]),
    ]
    states = assign(candidates)
    assert len(states) == 1 and states[0]["text"] == ["bravo"]
    assert candidates[2]["text_contents"] == ["bravo"]
    assert candidates[2]["vision"]["text"][0]["text"] == "BRAVQ"
    assert not candidates[2]["text_ready"]
    assert candidates[2]["text_state_id"] == 0


def test_confident_single_letter_change_is_never_smoothed_away():
    candidates = [candidate(index, index * 0.10, [region(text)])
                  for index, text in enumerate(["PRICE", "PRICE", "PRIDE", "PRIDE", "PRICE", "PRICE"])]
    assert [state["text"] for state in assign(candidates)] == [["price"], ["pride"], ["price"]]


def test_empty_ocr_flicker_does_not_create_logo_only_state():
    candidates = []
    for index, text in enumerate(["HELLO", "HELLO", "", "HELLO", "HELLO"]):
        candidates.append(candidate(index, index * 0.1, [
            region("LOGO", box=[50, 30, 90, 35]),
            region(text, confidence=0.0 if not text else 0.97, box=[100, 160, 200, 35]),
        ]))
    states = assign(candidates)
    assert len(states) == 1
    assert states[0]["text"] == ["logo", "hello"]
    assert not candidates[2]["text_ready"]


def test_early_unreadable_crop_on_a_later_readable_track_is_not_a_text_free_frame():
    candidates = [
        candidate(0, 0, [region("", confidence=0.0, box=[100, 160, 80, 35])]),
        candidate(1, 0.05, [region("HEL", confidence=0.40, box=[100, 160, 80, 35])]),
        candidate(2, 0.10, [region("HELLO", box=[100, 160, 80, 35])]),
        candidate(3, 0.20, [region("HELLO", box=[100, 160, 80, 35])]),
    ]
    states = assign(candidates)
    assert [state["text"] for state in states] == [["hello"]]
    assert not candidates[0]["text_ready"] and candidates[0]["text_complete"] == 0
    assert not candidates[1]["text_ready"]


def test_unreadable_persistent_logo_cannot_block_readable_caption_forever():
    candidates = [
        candidate(index, index * 0.1, [
            region("L0GQ", confidence=0.25, box=[50, 30, 90, 35]), region("Complete caption"),
        ]) for index in range(4)
    ]
    states = assign(candidates)
    assert states[0]["text"] == ["complete caption"]
    assert all(item["text_ready"] for item in candidates)
    assert all("not certified" in item["text_reason"] for item in candidates)


def test_intact_repositioned_words_remain_readable_in_one_continuous_state():
    candidates = [
        candidate(index, index * 0.25, [
            region("WHOLE WORDS", box=[80 + index * 90, 160, 190, 35]),
        ], stability=0.35) for index in range(4)
    ]
    states = assign(candidates)
    assert len(states) == 1
    assert all(item["text_ready"] for item in candidates)
    assert "settled" in candidates[0]["text_reason"]


def test_edge_clipped_word_is_excluded_until_it_is_fully_in_view():
    candidates = [
        candidate(index, index * 0.1, [
            region("WHOLE WORD", clipped=index < 2, box=[0 if index < 2 else 80, 160, 240, 35]),
        ]) for index in range(5)
    ]
    states = assign(candidates)
    assert len(states) == 1
    assert all(not item["text_ready"] for item in candidates[:2])
    assert all(item["text_ready"] for item in candidates[2:])
    assert candidates[0]["text_complete"] == 0


def test_detector_padding_at_edge_is_not_mistaken_for_clipped_glyphs():
    candidates = [candidate(index, index * 0.1, [dict(
        region("VISIBLE"), box=[0, 160, 160, 35],
        raw_polygon=[[12, 165], [150, 165], [150, 190], [12, 190]],
    )]) for index in range(3)]
    assert len(assign(candidates)) == 1
    assert all(item["text_ready"] for item in candidates)


def test_confident_artificial_crop_cannot_certify_complete_letters():
    candidates = [
        candidate(index, index * 0.1, [dict(
            region("COMPLETE WORDS", clipped=False),
            complete_geometry=index >= 2,
        )]) for index in range(4)
    ]
    assert len(assign(candidates)) == 1
    assert all(not item["text_ready"] for item in candidates[:2])
    assert all(item["text_ready"] for item in candidates[2:])
    assert "omit letters" in candidates[0]["text_reason"]


def test_prominent_caption_cannot_inherit_an_unreliable_tall_background_track():
    candidates = [
        candidate(index, index * 0.1, [
            region("Stable heading", box=[100, 100, 400, 60]),
            region(f"O{index}", confidence=0.95 if index == 0 else 0.25,
                   box=[625, 277, 35, 102]),
        ]) for index in range(8)
    ]
    candidates += [
        candidate(10 + index, 1.0 + index * 0.1, [
            region("Clear", box=[539, 258, 225, 62]),
            region("details", box=[770, 256, 122, 52]),
        ]) for index in range(4)
    ]
    states = assign(candidates)
    assert states[-1]["text"] == ["clear details"]
    assert all(item["text_ready"] and "clear details" in item["text_contents"] for item in candidates[-4:])


def test_fading_text_prefers_observed_contrast_plateau():
    candidates = [candidate(index, index * 0.1, [
        region("COMPLETE WORDING", contrast=contrast),
    ]) for index, contrast in enumerate([0.03, 0.15, 0.50, 0.90, 0.92, 0.90])]
    assert len(assign(candidates)) == 1
    assert all(not item["text_ready"] for item in candidates[:3])
    assert all(item["text_ready"] for item in candidates[3:])
    assert candidates[2]["text_complete"] < candidates[3]["text_complete"]


def test_normalization_preserves_meaningful_accents_digits_and_word_order():
    assert normalize_text("  ¡SÍ!  Paga $２５  ") == "sí paga $25"
    candidates = [candidate(index, index * 0.1, [region(text)])
                  for index, text in enumerate(["Si", "Si", "Sí", "Sí", "Dog bites man", "Dog bites man",
                                               "Man bites dog", "Man bites dog"])]
    assert [state["text"] for state in assign(candidates)] == [
        ["si"], ["sí"], ["dog bites man"], ["man bites dog"],
    ]


def test_missing_recognition_fields_and_no_text_remain_neutral():
    candidates = [candidate(0, 0, []), candidate(1, 0.1, [{
        "box": [10, 10, 100, 25], "contrast": 0.9, "confidence": 0.99,
    }])]
    assert assign(candidates) == []
    assert all(item["text_ready"] and item["text_state_id"] is None for item in candidates)
    assert all(item["text_complete"] == 1 for item in candidates)


def test_single_observation_requires_confirmation_except_actual_one_frame_shot():
    candidates = [candidate(0, 0, [region("ALPHA")]), candidate(10, 0.5, [region("BRAVO")])]
    assert assign(candidates) == []
    assert all(not item["text_ready"] for item in candidates)
    one = [candidate(7, 0.25, [region("QUICK COMPLETE TITLE")])]
    scenes = [{"id": 0, "start_frame": 7, "end_frame": 7, "start": 0.25, "end": 0.29}]
    states = assign_text_states(one, scenes, {"width": 960, "height": 540, "fps": 24})
    assert len(states) == 1 and one[0]["text_ready"]


def test_candidate_order_does_not_change_state_identity_or_mutate_ocr_records():
    original = [candidate(index, index * 0.1, [region(text)])
                for index, text in enumerate(["ALPHA", "ALPHA", "BRAVO", "BRAVO"])]
    shuffled = copy.deepcopy(original[::-1])
    expected = assign(original)
    actual = assign(shuffled)
    assert actual == expected
    assert shuffled[-1]["vision"]["text"][0]["text"] == "ALPHA"


def test_split_and_merged_line_regions_have_one_stable_wording_identity():
    candidates = [
        candidate(0, 0, [region("Benefits that support:", box=[100, 100, 420, 40])]),
        candidate(1, 0.1, [region("Benefits that support:", box=[100, 100, 420, 40])]),
        candidate(2, 0.2, [
            region("Benefits", box=[100, 101, 150, 39]),
            region("that", box=[257, 98, 72, 43]),
            region("support:", box=[336, 102, 180, 40]),
        ]),
        candidate(3, 0.3, [
            region("Benefits that", box=[100, 100, 235, 41]),
            region("support:", box=[335, 102, 180, 40]),
        ]),
    ]
    states = assign(candidates)
    assert len(states) == 1 and states[0]["text"] == ["benefits that support"]
    assert all(item["text_ready"] for item in candidates)


def test_boundary_quote_and_footer_comma_jitter_do_not_change_words():
    values = ['FATTY"', "fatty", "Fatty'", "FATTY"]
    candidates = [candidate(index, index * 0.1, [region(value)])
                  for index, value in enumerate(values)]
    assert [state["text"] for state in assign(candidates)] == [["fatty"]]
    assert normalize_text("Costs $25,000.50.") == "costs $25,000.50"


def test_uncertain_superscript_does_not_hide_complete_headline_or_change_state():
    candidates = []
    for index, small in enumerate(["15", "15", "a", "15", ""]):
        regions = [region("Complete headline", box=[100, 160, 420, 60])]
        if small:
            regions.append(region(small, confidence=0.80 if small == "a" else 0.99,
                                  box=[505, 160, 14, 10]))
        candidates.append(candidate(index, index * 0.1, regions))
    states = assign(candidates)
    assert len(states) == 1 and states[0]["text"] == ["complete headline"]
    assert all(item["text_ready"] for item in candidates)


def test_genuine_tiny_price_and_short_subtitle_changes_are_preserved():
    for values in (["25", "25", "75", "75"], ["NY", "NY", "LA", "LA"]):
        candidates = [
            candidate(index, index * 0.1, [
                region("Logo", box=[50, 50, 400, 70]),
                region(value, box=[520, 165, 25, 10]),
            ]) for index, value in enumerate(values)
        ]
        assert len(assign(candidates)) == 2
        assert all(item["text_ready"] for item in candidates)


def test_multiline_entrance_waits_for_complete_paragraph():
    candidates = []
    lines = [
        ("See how good", [100, 150, 250, 40]),
        ("you can feel", [100, 200, 230, 40]),
        ("at any age", [100, 250, 220, 40]),
    ]
    for index, (time, count) in enumerate([(0, 1), (0.1, 1), (0.3, 2), (0.4, 2), (0.6, 3), (0.7, 3)]):
        candidates.append(candidate(index, time, [region(text, box=box) for text, box in lines[:count]]))
    states = assign(candidates)
    assert len(states) == 1 and states[0]["text"] == ["see how good", "you can feel", "at any age"]
    assert all(not item["text_ready"] for item in candidates[:4])
    assert all(item["text_ready"] for item in candidates[4:])


def test_multiline_entrance_can_include_intermediate_partially_typed_lines():
    texts = [
        (0, ["See how good"]), (0.1, ["See how good"]),
        (0.2, ["See how good", "you ca"]), (0.25, ["See how good", "you can"]),
        (0.3, ["See how good", "you can feel"]), (0.4, ["See how good", "you can feel"]),
        (0.5, ["See how good", "you can feel", "at any"]),
        (0.6, ["See how good", "you can feel", "at any age"]),
        (0.7, ["See how good", "you can feel", "at any age"]),
    ]
    candidates = [
        candidate(index, time, [
            region(text, box=[100, 150 + row * 50, 260, 40]) for row, text in enumerate(lines)
        ]) for index, (time, lines) in enumerate(texts)
    ]
    states = assign(candidates)
    assert len(states) == 1
    assert states[0]["text"] == ["see how good", "you can feel", "at any age"]
    assert all(not item["text_ready"] for item in candidates[:-2])


def test_camera_scene_cut_does_not_turn_unchanged_complete_words_into_fragment():
    candidates = [candidate(index, index * 0.1, [
        region("Trusted by", box=[100 - index * 2, 150, 180, 40]),
        region("Ricki Lake", box=[550 + index * 2, 150, 180, 40]),
    ]) for index in range(6)]
    for item in candidates[2:]:
        item["scene_id"] = 1
    scenes = [
        {"id": 0, "start_frame": 0, "end_frame": 1, "start": 0, "end": 0.2},
        {"id": 1, "start_frame": 2, "end_frame": 5, "start": 0.2, "end": 0.6},
    ]
    states = assign_text_states(candidates, scenes, {"width": 960, "height": 540, "fps": 24})
    assert len(states) == 1
    assert all(item["text_ready"] for item in candidates)


def test_real_lossless_video_and_local_ocr_track_changed_wording(tmp_path):
    from stills_tool import core, vision
    if not hasattr(vision.LocalVision, "recognize_text"):
        pytest.skip("Native OCR recognizer is not available")
    try:
        ffmpeg = core.find_tool("ffmpeg")
        model = vision.LocalVision()
    except (RuntimeError, FileNotFoundError) as error:
        pytest.skip(f"Local integration dependencies unavailable: {error}")
    frames = []
    for word in ("ALPHA", "BRAVO", "ALPHA"):
        image = np.full((360, 640, 3), 235, np.uint8)
        cv2.putText(image, word, (105, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (15, 15, 15), 7, cv2.LINE_AA)
        frames.extend([image] * 24)
    source = tmp_path / "wording.mkv"
    command = [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", "640x360", "-framerate", "24", "-i", "pipe:0",
        "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source),
    ]
    subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames),
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    capture = cv2.VideoCapture(str(source))
    sample = {6, 18, 30, 42, 54, 66}
    candidates = []
    try:
        for index in range(72):
            ok, image = capture.read()
            assert ok
            if index in sample:
                detections = model.analyze(image)["text"]
                recognized = model.recognize_text(image, detections)
                candidates.append(candidate(index, index / 24, recognized))
    finally:
        capture.release()
        model.close()
    states = assign_text_states(candidates, [
        {"id": 0, "start_frame": 0, "end_frame": 71, "start": 0, "end": 3},
    ], {"width": 640, "height": 360, "fps": 24})
    assert [state["text"] for state in states] == [["alpha"], ["bravo"], ["alpha"]]
    assert all(item["text_ready"] for item in candidates)
