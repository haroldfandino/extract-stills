"""Texture rejection must preserve genuine numeric labels and prices."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from stills_tool.text_appearance import attach_text_appearance
from stills_tool.text_pattern import text_pattern_evidence


def evidence(frame, text="000000", confidence=.6, box=None):
    region = {"text": text, "confidence": confidence, "clipped": False,
              "complete_geometry": True,
              "box": box or [0, 0, frame.shape[1], frame.shape[0]]}
    return text_pattern_evidence(attach_text_appearance(frame, [region])[0])


def repeated_vertical_digits():
    image = np.full((220, 120, 3), 245, np.uint8)
    for index in range(6):
        cv2.putText(image, "0", (40, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (10, 10, 10), 1, cv2.LINE_AA)
    return image


def test_printed_vertical_repeated_digits_have_strong_periodic_support():
    result = evidence(repeated_vertical_digits())
    assert result["periodicity"] > .8
    assert result["best_lag"] == 30
    assert not result["unlikely_repeated_glyphs"]


def test_weak_nonperiodic_vertical_texture_is_not_a_confirmed_zero_column():
    noise = np.random.default_rng(8).integers(40, 230, (152, 66), dtype=np.uint8)
    image = np.repeat(noise[:, :, None], 3, axis=2)
    result = evidence(image, "00000")
    assert result["periodicity"] < .35
    assert result["unlikely_repeated_glyphs"]
    assert not evidence(image, "00000", confidence=.95)["unlikely_repeated_glyphs"]


@pytest.mark.parametrize("text", ["$25", "$75", "123456", "2026", "Price 00000"])
def test_prices_mixed_digits_and_ordinary_words_remain_neutral(text):
    assert not evidence(repeated_vertical_digits(), text)["unlikely_repeated_glyphs"]


def test_horizontal_numeric_reading_is_never_rejected_for_missing_periodicity():
    image = np.full((50, 250, 3), 235, np.uint8)
    cv2.putText(image, "00000", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    result = evidence(image, "00000")
    assert result["reading_axis"] == "horizontal"
    assert not result["unlikely_repeated_glyphs"]


def test_missing_native_evidence_never_rejects_a_reading():
    assert not text_pattern_evidence({"text": "00000", "box": [0, 0, 66, 152]})["unlikely_repeated_glyphs"]


def test_real_blouse_lace_does_not_support_the_repeated_zero_reading():
    root = Path(__file__).resolve().parents[1] / "validation/beta3_samples/Fatty15_RickiLake2_Amazon_30_HD_Social_H264"
    path = next(root.glob("run*/*frame_00000102.png"), None)
    if path is None:
        pytest.skip("Local native blouse fixture unavailable")
    result = evidence(cv2.imread(str(path)), "00000", box=[214, 476, 66, 152])
    assert result["periodicity"] < .2
    assert result["unlikely_repeated_glyphs"]
