"""Conservative evidence for repeated-glyph OCR on narrow texture regions."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .text_appearance import decode_text_appearance


def text_pattern_evidence(region: dict) -> dict:
    """Measure repeated strokes without rejecting ordinary digits or prices.

    This diagnostic applies only to four or more identical observed characters.
    Printed repetitions should repeat along their reading axis. A weak detector
    finding a narrow vertical strip of nonperiodic texture has little support
    for its repeated-character OCR reading. Strong detections, horizontal text,
    mixed digits, prices and unavailable pixels remain neutral.
    """
    text = "".join(str(region.get("text", "")).split())
    if len(text) < 4 or len(set(text)) != 1:
        return {"available": False, "unlikely_repeated_glyphs": False,
                "reason": "not_a_repeated_character_reading"}
    decoded = decode_text_appearance(region)
    if decoded is None:
        return {"available": False, "unlikely_repeated_glyphs": False,
                "reason": "native_appearance_unavailable"}
    capsule, patch = decoded
    left, top, right, bottom = capsule["glyph_bounds"]
    origin_x, origin_y = capsule["bounds"][:2]
    crop = patch[top-origin_y:bottom-origin_y, left-origin_x:right-origin_x]
    gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop).astype(np.float32)
    height, width = gray.shape
    vertical = height > width * 1.5
    length = height if vertical else width
    pitch = length / len(text)
    minimum = max(3, math.floor(pitch * .6))
    maximum = min(length // 3, math.ceil(pitch * 1.4))
    if min(width, height) < 8 or minimum > maximum:
        return {"available": False, "unlikely_repeated_glyphs": False,
                "reason": "insufficient_native_pattern_support"}
    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 2)
    correlations = []
    for lag in range(minimum, maximum + 1):
        first, second = ((high_pass[:-lag, :], high_pass[lag:, :]) if vertical else
                         (high_pass[:, :-lag], high_pass[:, lag:]))
        energy = float(np.mean(first * first) * np.mean(second * second))
        correlation = float(np.mean(first * second) / math.sqrt(energy)) if energy > 1e-6 else 0
        correlations.append((correlation, lag))
    correlation, lag = max(correlations)
    try:
        detector_confidence = float(region.get("confidence", 1))
    except (TypeError, ValueError):
        detector_confidence = 1
    if not math.isfinite(detector_confidence):
        detector_confidence = 1
    unlikely = vertical and detector_confidence < .8 and correlation < .35
    return {"available": True, "unlikely_repeated_glyphs": bool(unlikely),
            "reason": "weak_nonperiodic_vertical_texture" if unlikely else "repetition_is_not_rejected",
            "reading_axis": "vertical" if vertical else "horizontal",
            "native_size": [width, height], "character_count": len(text),
            "expected_pitch": round(pitch, 3), "periodicity": round(correlation, 5),
            "best_lag": lag, "detector_confidence": detector_confidence}
