"""Pixel evidence must remove OCR flicker while preserving real word changes."""

import base64
import json
from pathlib import Path
import zlib

import cv2
import numpy as np
import pytest

from stills_tool.text_appearance import (
    MAX_PATCH_PIXELS, attach_text_appearance, release_text_appearance,
    same_text_appearance, text_appearance_evidence,
)


def _frame(text="C15:0", size=(180, 640)):
    frame = np.full((*size, 3), 245, np.uint8)
    cv2.putText(frame, text, (120, 95), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (10, 10, 10), 3, cv2.LINE_AA)
    return frame


def _region(text, box=(112, 48, 235, 64)):
    return {"text": text, "box": list(box), "clipped": False, "complete_geometry": True}


def test_lossless_native_evidence_survives_json_cache_replay_and_detector_box_jitter():
    frame = _frame()
    first = attach_text_appearance(frame, [_region("C15:0")])[0]
    second = attach_text_appearance(frame, [_region("C15:O", (114, 46, 232, 68))])[0]
    replay = json.loads(json.dumps([first, second]))
    assert same_text_appearance(*replay)
    assert text_appearance_evidence(*replay)["mean_delta"] == 0
    # Surrounding analysis boxes can be rescaled while native capsule coordinates
    # stay self-contained; comparison never relies on the outer box dimensions.
    for region in replay:
        region["box"] = [value / 2 for value in region["box"]]
    assert same_text_appearance(*replay)
    capsule = first["appearance"]
    left, top, right, bottom = capsule["bounds"]
    decoded = np.frombuffer(zlib.decompress(base64.b64decode(capsule["data"])), np.uint8)
    original = frame[top:bottom, left:right]
    assert np.array_equal(decoded.reshape(original.shape), original)


@pytest.mark.parametrize("before,after", [("$25", "$75"), ("cat", "car"), ("C15:0", "C15:O")])
def test_real_digit_single_letter_and_zero_o_changes_remain_distinct(before, after):
    first = attach_text_appearance(_frame(before), [_region(before)])[0]
    second = attach_text_appearance(_frame(after), [_region(after)])[0]
    evidence = text_appearance_evidence(first, second)
    assert not evidence["same"]
    assert evidence["max_tile_delta"] > 20


def test_one_small_native_glyph_change_is_not_diluted_by_a_long_text_region():
    first = np.full((80, 1800, 3), 245, np.uint8)
    second = first.copy()
    second[30, 800] = 10
    region = _region("long fine print", (10, 20, 1750, 30))
    records = [attach_text_appearance(frame, [region])[0] for frame in (first, second)]
    assert not same_text_appearance(*records), "One changed native stroke must survive whole-line averaging"
    assert text_appearance_evidence(*records)["max_pixel_delta"] == 235


def test_tiny_codec_noise_is_allowed_without_hiding_a_character_stroke():
    first = _frame()
    second = first.copy()
    second[60:65, 200:205] -= 1
    records = [attach_text_appearance(frame, [_region("caption")])[0] for frame in (first, second)]
    assert same_text_appearance(*records)
    second[60:65, 200:205] -= 30
    changed = attach_text_appearance(second, [_region("caption")])[0]
    assert not same_text_appearance(records[0], changed)


def _opaque_title(text, seed):
    # Smooth moving background has no unexplained coherent stroke support.
    x = np.linspace(0, 100, 640)
    gray = np.tile(np.uint8(40+seed*10+x), (160, 1))
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    cv2.putText(frame, text, (120, 95), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def test_opaque_white_title_matches_across_changing_smooth_backgrounds():
    records = [attach_text_appearance(_opaque_title("TITLE", seed), [_region("TITLE")])[0]
               for seed in (1, 2)]
    evidence = text_appearance_evidence(*records)
    assert evidence["same"] and evidence["reason"] == "opaque_native_glyph_paint_matches"


def test_opaque_black_title_has_symmetric_proof_on_moving_smooth_light_backgrounds():
    records = []
    for seed in (1, 2):
        gray = np.tile(np.uint8(np.linspace(80+seed*10, 210+seed*10, 640)), (160, 1))
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        cv2.putText(frame, "TITLE", (120, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        records.append(attach_text_appearance(frame, [_region("TITLE")])[0])
    evidence = text_appearance_evidence(*records)
    assert evidence["same"] and evidence["paint"] == "clipped_black"


def test_sparse_saturated_texture_is_not_coherent_glyph_paint():
    rng = np.random.default_rng(9)
    images = []
    pixels = np.indices((160, 640))
    speckle = (pixels[0] % 3 == 0) & (pixels[1] % 3 == 0)
    for _ in range(2):
        frame = rng.integers(60, 210, (160, 640, 3), dtype=np.uint8)
        frame[speckle] = 255
        images.append(frame)
    records = [attach_text_appearance(frame, [_region("texture")])[0] for frame in images]
    assert not same_text_appearance(*records)


@pytest.mark.parametrize("scale", [.35, 1.0])
def test_thin_small_native_price_changes_are_never_hidden_by_foreground_proof(scale):
    records = []
    for text, seed in (("$25", 1), ("$75", 2)):
        rng = np.random.default_rng(seed)
        gray = cv2.GaussianBlur(rng.integers(30, 200, (120, 360), dtype=np.uint8), (5, 5), 0)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        cv2.putText(frame, text, (100, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
        box = (95, 40, 100, 35) if scale == 1.0 else (95, 50, 35, 20)
        records.append(attach_text_appearance(frame, [_region(text, box)])[0])
    assert not same_text_appearance(*records)


def test_thin_glyph_cores_alone_do_not_prove_equivalence_on_a_moving_background():
    records = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        gray = cv2.GaussianBlur(rng.integers(30, 200, (120, 360), dtype=np.uint8), (5, 5), 0)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        cv2.putText(frame, "$25", (100, 65), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        records.append(attach_text_appearance(frame, [_region("$25", (95, 45, 45, 25))])[0])
    assert not same_text_appearance(*records)


@pytest.mark.parametrize("before,after", [("$25", "$75"), ("cat", "car"), ("C15:0", "C15:O")])
def test_changed_opaque_letters_on_moving_backgrounds_remain_distinct(before, after):
    records = [attach_text_appearance(_opaque_title(text, seed), [_region(text)])[0]
               for text, seed in ((before, 1), (after, 2))]
    assert not same_text_appearance(*records)


def test_local_change_outside_opaque_title_paint_stays_visible_to_strict_evidence():
    frame = _opaque_title("TITLE", 1)
    other = frame.copy()
    # The paint comparison may ignore moving background; it must never ignore
    # removing a foreground stroke simply because most letters remain unchanged.
    paint = np.argwhere(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) == 255)
    y, x = paint[len(paint)//2]
    other[y, x] = 80
    records = [attach_text_appearance(image, [_region("TITLE")])[0] for image in (frame, other)]
    assert not same_text_appearance(*records)


def test_missing_full_glyph_union_coverage_never_claims_equivalence():
    frame = _frame("COMPLETE WORDS")
    first = attach_text_appearance(frame, [_region("COMPLETE", (112, 48, 110, 64))])[0]
    second = attach_text_appearance(frame, [_region("COMPLETE WORDS", (112, 48, 450, 64))])[0]
    assert not same_text_appearance(first, second)
    assert text_appearance_evidence(first, second)["reason"] == "complete_glyph_union_is_not_covered"


def test_merged_and_split_regions_compare_using_composite_native_patches():
    frame = _frame("ONE TWO")
    merged = attach_text_appearance(frame, [_region("ONE TWO", (110, 48, 220, 64))])[0]
    split = attach_text_appearance(frame, [_region("ONE", (110, 48, 110, 64)),
                                           _region("TWO", (220, 48, 110, 64))])
    assert same_text_appearance(merged, split)
    assert same_text_appearance({"regions": split}, merged)
    changed = _frame("ONE TEN")
    other = attach_text_appearance(changed, [_region("ONE", (110, 48, 110, 64)),
                                           _region("TEN", (220, 48, 110, 64))])
    assert not same_text_appearance(merged, other)


def test_oversized_or_clipped_geometry_has_no_pixel_equivalence_proof():
    image = np.zeros((800, 1200, 3), np.uint8)
    large = attach_text_appearance(image, [_region("large", (0, 0, 1200, 600))])[0]
    assert not large["appearance"]["available"]
    assert large["appearance"]["reason"] == "native_patch_exceeds_size_limit"
    assert "data" not in large["appearance"]
    clipped = attach_text_appearance(_frame(), [dict(_region("cut"), clipped=True)])[0]
    assert not same_text_appearance(clipped, clipped)


def test_malformed_and_oversized_encoded_evidence_is_rejected():
    record = attach_text_appearance(_frame(), [_region("caption")])[0]
    invalid = json.loads(json.dumps(record))
    invalid["appearance"]["data"] = "not base64!"
    assert not same_text_appearance(record, invalid)
    bomb = json.loads(json.dumps(record))
    bomb["appearance"]["data"] = base64.b64encode(zlib.compress(b"x" * (MAX_PATCH_PIXELS+1))).decode()
    assert not same_text_appearance(record, bomb)
    invalid["appearance"]["bounds"] = [0, 0, -1, 3]
    assert not same_text_appearance(record, invalid)


def test_production_payload_release_preserves_geometry_and_prevents_false_replay_proof():
    records = attach_text_appearance(_frame(), [_region("caption")])
    candidate = {"id": 1, "vision": {"text": records}, "text_ready": True}
    before_bounds = list(records[0]["appearance"]["bounds"])
    release_text_appearance([candidate])
    capsule = records[0]["appearance"]
    assert "data" not in capsule
    assert capsule["bounds"] == before_bounds
    assert capsule["payload_discarded"] and not capsule["available"]
    assert candidate["text_ready"]
    assert not same_text_appearance(records[0], records[0])


@pytest.mark.parametrize("frames,box,words", [
    ((98, 108), (1144, 720, 150, 48), ("FAST@MPANY", "FASTOMPANY")),
    ((232, 233), (185, 480, 295, 120), ("C15:0", "C15:O")),
])
def test_real_native_ocr_variants_have_matching_original_pixels(frames, box, words):
    root = Path(__file__).resolve().parents[1] / "validation/beta3_samples/Fatty15_RickiLake2_Amazon_30_HD_Social_H264"
    paths = [next(root.glob(f"run*/*frame_{index:08d}.png"), None) for index in frames]
    if any(path is None for path in paths):
        pytest.skip("Cached native Fatty30 stills unavailable")
    records = [attach_text_appearance(cv2.imread(str(path)), [_region(word, box)])[0]
               for path, word in zip(paths, words)]
    assert same_text_appearance(*records)


@pytest.mark.parametrize("frames", [(155, 173), (179, 181)])
def test_real_mixbook_busy_background_has_no_complete_glyph_support_proof(frames):
    source = Path(__file__).resolve().parents[1] / "validation/beta3_enriched/Mixbook_Juniper_Birthday_30_4K_Social_H264/analysis.json"
    if not source.is_file():
        pytest.skip("Enriched native Mixbook logo evidence unavailable")
    analysis = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for index in frames:
        candidate = next(item for item in analysis["candidates"] if item["frame_index"] == index)
        records.append(next(item for item in candidate["vision"]["text"] if "mixbook" in item["text"].lower()))
    assert not same_text_appearance(*records), "Version-1 grayscale snapshots cannot prove unchanged colored glyphs"


def test_isoluminant_colored_price_changes_cannot_match_as_identical_grayscale():
    frames = []
    for text in ("$25", "$75"):
        frame = np.full((140, 400, 3), (255, 0, 0), dtype=np.uint8)
        cv2.putText(frame, text, (80, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 97), 3, cv2.LINE_8)
        frames.append(frame)
    assert np.array_equal(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY))
    records = [attach_text_appearance(frame, [_region(text, (70, 45, 130, 60))])[0]
               for text, frame in zip(("$25", "$75"), frames)]
    assert not same_text_appearance(*records)


def test_old_grayscale_capsules_never_claim_full_color_equivalence():
    record = attach_text_appearance(_frame(), [_region("caption")])[0]
    record["appearance"]["version"] = 1
    record["appearance"]["encoding"] = "gray8-zlib-base64"
    assert not same_text_appearance(record, record)


@pytest.mark.parametrize("color", [(0, 255, 255), (180, 180, 180), (130, 130, 130)])
@pytest.mark.parametrize("before,after", [("$25", "$75"), ("cat", "car")])
def test_matching_white_heading_cannot_hide_a_changed_unsaturated_price_or_word(color, before, after):
    records = []
    for text, seed in ((before, 1), (after, 2)):
        gray = np.random.default_rng(seed).integers(30, 180, (140, 600), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        cv2.putText(frame, "PRICE", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        records.append(attach_text_appearance(frame, [_region("PRICE "+text, (35, 45, 330, 60))])[0])
    assert not same_text_appearance(*records)


def test_colored_middle_word_cannot_hide_between_matching_white_words():
    records = []
    for text, seed in (("cat", 1), ("car", 2)):
        gray = np.tile(np.uint8(np.linspace(40+seed*10, 150+seed*10, 700)), (160, 1))
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        for word, x, color in (("BUY", 40, (255, 255, 255)), (text, 170, (180, 180, 180)),
                               ("NOW", 300, (255, 255, 255))):
            cv2.putText(frame, word, (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        records.append(attach_text_appearance(frame, [_region("BUY "+text+" NOW", (30, 45, 380, 64))])[0])
    assert not same_text_appearance(*records)
