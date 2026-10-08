"""Offline Latin-script line recognition using PaddlePaddle's pinned ONNX model.

The model and its exact character dictionary are verified by LocalVision before
this module loads them. No Paddle framework, YAML package or network is needed.
Recognition confidence measures model certainty, not whether a sentence is
complete; the temporal selector must also check stability and crop boundaries.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_characters(config_path: Path) -> list[str]:
    """Read only the scalar character list from the pinned inference.yml.

    This intentionally supports the small YAML subset used by that immutable
    file, rather than interpreting arbitrary YAML tags or constructing objects.
    Duplicate entries are meaningful class indexes and must stay in order.
    """
    content = config_path.read_text(encoding="utf-8")
    if "  name: CTCLabelDecode\n" not in content:
        raise ValueError("Text recognition dictionary requires CTCLabelDecode")
    marker = "  character_dict:\n"
    if content.count(marker) != 1:
        raise ValueError("Text recognition dictionary has no unique character list")
    characters = []
    for line in content.split(marker, 1)[1].splitlines():
        if not line.startswith("  - "):
            if not line.strip():
                continue
            break
        scalar = line[4:].strip()
        if scalar.startswith("'") and scalar.endswith("'"):
            value = scalar[1:-1].replace("''", "'")
        elif scalar.startswith('"') and scalar.endswith('"'):
            value = json.loads(scalar)
        else:
            value = scalar
        if not isinstance(value, str) or len(value) != 1:
            raise ValueError("Text recognition dictionary contains a non-character entry")
        characters.append(value)
    if len(characters) != 836 or not all(char in characters for char in "áéíóúñü¿"):
        # The pinned v5 Latin ONNX output has 838 classes: blank, 836 entries,
        # space. Reject silent class-index drift instead of recognizing garbage.
        raise ValueError("Text recognition dictionary does not match the pinned Latin model")
    return [""] + characters + [" "]


def decode_ctc(prediction: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """Greedy CTC decoding; repeated letters separated by blank are retained."""
    scores = np.asarray(prediction)
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != len(characters):
        raise RuntimeError("Text recognizer returned an incompatible CTC output")
    if scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise RuntimeError("Text recognizer returned an invalid probability sequence")
    # This official export includes softmax. Do not softmax probabilities again.
    if float(scores.min()) < -1e-5 or float(scores.max()) > 1.00001:
        raise RuntimeError("Text recognizer returned values outside the probability range")
    indexes = scores[0].argmax(axis=1)
    probabilities = scores[0].max(axis=1)
    keep = np.ones(indexes.shape, dtype=bool)
    keep[1:] = indexes[1:] != indexes[:-1]
    keep &= indexes != 0
    text = "".join(characters[int(index)] for index in indexes[keep]).strip()
    confidence = float(np.mean(probabilities[keep])) if text else 0.0
    return text, float(np.clip(confidence, 0, 1))


def _polygon(region: dict[str, Any]) -> np.ndarray:
    points = region.get("recognition_polygon", region.get("polygon"))
    if points is None:
        x, y, width, height = region["box"]
        points = [[x, y], [x+width-1, y], [x+width-1, y+height-1], [x, y+height-1]]
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("Text regions require a finite four-point polygon")
    return polygon


def region_is_clipped(region: dict[str, Any], frame_shape: tuple[int, ...]) -> bool:
    """Check detector support before its padding and coordinate clamping."""
    if "clipped" in region:
        return bool(region["clipped"])
    height, width = frame_shape[:2]
    points = np.asarray(region.get("raw_polygon", _polygon(region)), dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Text regions require finite raw detector geometry")
    # A detector contour touching the canvas cannot prove all letters are visible.
    return bool((points[:, 0] <= 1).any() or (points[:, 0] >= width-2).any()
                or (points[:, 1] <= 1).any() or (points[:, 1] >= height-2).any())


def rectify_region(frame: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    """Perspective-rectify a quadrilateral without using low-res previews."""
    points = _polygon(region)
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    points = points[np.argsort(angle)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if abs(float(cv2.contourArea(points))) < 4:
        raise ValueError("Text region polygon is degenerate")
    width = round(max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3])))
    height = round(max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1])))
    if min(width, height) < 2:
        raise ValueError("Text region is too small to rectify")
    # The input is bounded by a real decoded frame; reject unreasonable external
    # regions instead of allocating arbitrarily large warps.
    if width > frame.shape[1] * 2 or height > frame.shape[0] * 2:
        raise ValueError("Text region lies beyond the supplied frame")
    destination = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(frame, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    if height / width >= 1.5:
        crop = np.ascontiguousarray(np.rot90(crop))
    return crop


class TextRecognizer:
    """One CPU worker's local ONNX recognizer; no face inference is performed."""

    def __init__(self, model_path: Path, config_path: Path):
        import onnxruntime as ort

        self.characters = load_characters(config_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1:3] != [3, 48]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or len(outputs[0].shape) != 3 or outputs[0].shape[-1] != len(self.characters)):
            raise ValueError("Text recognition ONNX shape does not match the pinned Latin model")
        self.input_name = inputs[0].name

    @staticmethod
    def prepare(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        ratio_width = max(1, math.ceil(48 * width / height))
        tensor_width = min(3200, max(320, math.ceil(ratio_width / 8) * 8))
        resized_width = min(tensor_width, ratio_width)
        resized = cv2.resize(crop, (resized_width, 48), interpolation=cv2.INTER_LINEAR)
        # RecResizeImg: BGR, [0,255] -> [-1,1], NCHW, normalized-zero padding.
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.zeros((1, 3, 48, tensor_width), dtype=np.float32)
        tensor[0, :, :, :resized_width] = normalized.transpose(2, 0, 1)
        return tensor

    def _line(self, crop: np.ndarray) -> tuple[str, float]:
        tensor = self.prepare(crop)
        prediction = self.session.run(None, {self.input_name: tensor})[0]
        return decode_ctc(prediction, self.characters)

    def _oriented_line(self, crop: np.ndarray) -> tuple[str, float]:
        text, confidence = self._line(crop)
        # No extra orientation model is needed for occasional upside-down
        # titles. Keep the ordinary reading when confidence is already high.
        if confidence < 0.85:
            rotated_text, rotated_confidence = self._line(cv2.rotate(crop, cv2.ROTATE_180))
            if rotated_confidence > confidence + 0.10:
                text, confidence = rotated_text, rotated_confidence
        return text, confidence

    def recognize(self, frame: np.ndarray, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for region in regions:
            item = dict(region)
            item["clipped"] = region_is_clipped(region, frame.shape)
            crop = rectify_region(frame, region)
            text, confidence = self._oriented_line(crop)
            # DB can shrink a superscript/short-number crop inside the glyphs.
            # Limited native-pixel context repairs incomplete local crops such
            # as a visible "15" initially read as "a". Large words never expand
            # here, preserving neighboring word boundaries and their geometry.
            crop_height, crop_width = crop.shape[:2]
            if crop_height <= 40 and crop_width <= 96 and (len(text) <= 3 or confidence < .90):
                margin = min(12, max(4, round(crop_height * .4)))
                center, (width, height), angle = cv2.minAreaRect(_polygon(region))
                context_polygon = cv2.boxPoints((center, (width+2*margin, height+2*margin), angle))
                context_crop = rectify_region(frame, {"polygon": context_polygon})
                context_text, context_confidence = self._oriented_line(context_crop)
                extension = (bool(text) and text in context_text and len(context_text) > len(text)
                             and context_confidence >= confidence - .01)
                if context_confidence >= .90 and (context_confidence > confidence + .05 or extension):
                    text, confidence = context_text, context_confidence
                    item["recognition_polygon"] = context_polygon.tolist()
            item.update(text=text, recognition_confidence=confidence)
            result.append(item)
        return result
