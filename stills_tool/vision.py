"""Offline, CPU-only face expression and text-region measurements.

Coordinates are in the supplied BGR analysis image. Sharpness is the variance of
the Laplacian on a normalized ROI; it is a relative metric, not a blur probability.
Text confidence is DB foreground confidence. MediaPipe does not expose a raw
per-face probability, so face confidence describes geometric reliability.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


MODEL_INFO = {
    "face": "MediaPipe Face Landmarker float16/1 (Apache-2.0)",
    "text": "PP-OCRv5 mobile detector ONNX (Apache-2.0)",
    "device": "CPU",
    "network_access": False,
    "face_confidence": "geometric reliability; not a model probability",
}


class MissingModelsError(RuntimeError):
    """A required local model is missing, invalid, or cannot be loaded."""


def _resources() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sharpness(gray: np.ndarray) -> float:
    if gray.size == 0 or min(gray.shape[:2]) < 3:
        return 0.0
    # Fixed crop dimensions make regional scores comparable across candidates.
    height, width = gray.shape[:2]
    scale = 128.0 / max(height, width)
    normalized = cv2.resize(gray, (max(3, round(width * scale)), max(3, round(height * scale))))
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    return float(cv2.Laplacian(normalized, cv2.CV_32F).var())


def _box(points: np.ndarray, width: int, height: int) -> list[int]:
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    x0, y0 = max(0, min(width - 1, x0)), max(0, min(height - 1, y0))
    x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


class LocalVision:
    """Load only verified local models; never fetch assets at runtime.

    Explicit model_dir takes precedence over STILLS_MODEL_DIR and bundled assets.
    Instances are intended for one analysis worker and are not thread-safe.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self._landmarker = None
        self._session = None
        self._closed = False
        root = _resources()
        self.model_dir = Path(model_dir or os.environ.get("STILLS_MODEL_DIR", root / "models")).resolve()
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = root / "models" / "manifest.json"
        hint = (f'Run python scripts/setup_models.py --model-dir "{self.model_dir}" during development, '
                "or reinstall the complete offline application.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported model manifest schema")
            entries = {entry["id"]: entry for entry in manifest["models"]}
            paths = {}
            for model_id in ("face_landmarker", "text_detector"):
                entry = entries[model_id]
                if Path(entry["filename"]).name != entry["filename"]:
                    raise ValueError("model manifest contains an unsafe filename")
                path = self.model_dir / entry["filename"]
                if not path.is_file():
                    raise MissingModelsError(f"Missing local model: {path}. {hint}")
                if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
                    raise MissingModelsError(f"Local model verification failed: {path}. {hint}")
                paths[model_id] = path
            self.model_info = dict(MODEL_INFO, models=manifest["models"])
        except MissingModelsError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MissingModelsError(f"Cannot read model manifest {manifest_path}: {exc}. {hint}") from exc
        try:
            import mediapipe as mp
            import onnxruntime as ort

            # Newer MediaPipe binaries document telemetry. They must be audited
            # before changing this pin, not silently used by the offline app.
            if mp.__version__ != "0.10.21":
                raise RuntimeError(f"Offline face runtime requires mediapipe==0.10.21; found {mp.__version__}")
            self._mp = mp
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(paths["face_landmarker"]),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=20,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            session_options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(paths["text_detector"]),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._input = self._session.get_inputs()[0]
        except Exception as exc:
            self.close()
            raise MissingModelsError(f"Could not load the offline CPU vision runtime: {exc}. {hint}") from exc

    def analyze(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LocalVision has been closed")
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8:
            raise ValueError("LocalVision requires a uint8 BGR analysis frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 2:
            raise ValueError("LocalVision requires a nonempty H x W x 3 BGR analysis frame")
        frame_bgr = np.ascontiguousarray(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        warnings: list[str] = []
        faces = self._faces(frame_bgr, gray, warnings)
        text = self._text(frame_bgr, gray)
        return {"faces": faces, "text": text, "warnings": warnings}

    def _faces(self, frame: np.ndarray, gray: np.ndarray, warnings: list[str]) -> list[dict]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        faces = []
        for index, landmarks in enumerate(result.face_landmarks):
            points = np.array([(point.x * width, point.y * height) for point in landmarks], dtype=np.float32)
            if not np.isfinite(points).all():
                warnings.append("A face had invalid landmarks and was ignored.")
                continue
            box = _box(points, width, height)
            x, y, w, h = box
            extent = points.max(axis=0) - points.min(axis=0)
            raw_area = max(1.0, float(extent[0] * extent[1]))
            visible = min(1.0, w * h / raw_area)
            # Confidence is zero for unreliable geometry, causing the selector
            # to treat expressions neutrally, rather than invent open eyes.
            tiny = min(w, h) < 40 or w * h < width * height * 0.001
            oversized = w * h > width * height * 0.85
            outside = bool((points[:, 0] < 0).any() or (points[:, 0] >= width).any()
                           or (points[:, 1] < 0).any() or (points[:, 1] >= height).any())
            confidence = 0.0 if tiny or oversized or visible < 0.9 or outside else 1.0
            coefficients = {}
            if index < len(result.face_blendshapes):
                coefficients = {item.category_name: float(item.score) for item in result.face_blendshapes[index]}
            available = all(name in coefficients for name in ("eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight"))
            if not available:
                confidence = 0.0
            open_eyes = 1.0 - max(coefficients.get("eyeBlinkLeft", 0.5), coefficients.get("eyeBlinkRight", 0.5))
            smile = (coefficients.get("mouthSmileLeft", 0.0) + coefficients.get("mouthSmileRight", 0.0)) / 2
            faces.append({
                "box": box,
                "confidence": confidence,
                "open_eyes": float(np.clip(open_eyes if confidence else 0.5, 0, 1)),
                "smile": float(np.clip(smile if confidence else 0.5, 0, 1)),
                "sharpness": _sharpness(gray[y:y+h, x:x+w]),
            })
        if any(face["confidence"] == 0 for face in faces):
            warnings.append("Small, cropped or unreliable faces received neutral expression scores.")
        if len(faces) == 20:
            warnings.append("Face detection reached the 20-face limit; additional faces may be unscored.")
        return faces

    def _text(self, frame: np.ndarray, gray: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        # Follow the official inference.yml: BGR, long side 960, multiples of
        # 32, ImageNet mean/std, then NCHW. Do not swap to RGB here.
        scale = 960.0 / max(height, width)
        target_h = max(32, int(round(height * scale / 32)) * 32)
        target_w = max(32, int(round(width * scale / 32)) * 32)
        input_shape = self._input.shape
        if len(input_shape) == 4:
            if isinstance(input_shape[2], int) and input_shape[2] > 0:
                target_h = input_shape[2]
            if isinstance(input_shape[3], int) and input_shape[3] > 0:
                target_w = input_shape[3]
        resized = cv2.resize(frame, (target_w, target_h)).astype(np.float32) / 255.0
        resized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        prediction = np.asarray(self._session.run(None, {self._input.name: tensor})[0]).squeeze()
        if prediction.ndim != 2 or not np.isfinite(prediction).all():
            raise RuntimeError("Text detector returned an invalid probability map")
        map_h, map_w = prediction.shape
        bitmap = np.uint8(prediction > 0.3) * 255
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Largest candidates first prevents tiny speckles consuming the cap.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1000]
        detections = []
        for contour in contours:
            rectangle = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rectangle
            if min(rw, rh) < 3:
                continue
            quadrilateral = cv2.boxPoints(rectangle)
            left, top, right, bottom = cv2.boundingRect(quadrilateral.astype(np.int32))
            right += left
            bottom += top
            left, top = max(0, left), max(0, top)
            right, bottom = min(map_w, right), min(map_h, bottom)
            if right <= left or bottom <= top:
                continue
            local_mask = np.zeros((bottom-top, right-left), dtype=np.uint8)
            cv2.fillPoly(local_mask, [(quadrilateral - [left, top]).astype(np.int32)], 1)
            confidence = float(cv2.mean(prediction[top:bottom, left:right], mask=local_mask)[0])
            if confidence < 0.6:
                continue
            # DB's area/perimeter expansion of a rotated rectangle. Parallel
            # edge offsets yield this expanded minimum-area rectangle directly.
            offset = rw * rh * 1.5 / max(2 * (rw + rh), 1e-6)
            expanded = cv2.boxPoints(((cx, cy), (rw + 2*offset, rh + 2*offset), angle))
            expanded[:, 0] = np.clip(expanded[:, 0] * width / map_w, 0, width-1)
            expanded[:, 1] = np.clip(expanded[:, 1] * height / map_h, 0, height-1)
            box = _box(expanded, width, height)
            x, y, w, h = box
            region = gray[y:y+h, x:x+w]
            polygon_mask = np.zeros(region.shape, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [(expanded - [x, y]).astype(np.int32)], 1)
            samples = region[polygon_mask != 0]
            contrast = float((np.percentile(samples, 90) - np.percentile(samples, 10)) / 255) if samples.size else 0.0
            detections.append({
                "box": box,
                "polygon": [[round(float(px), 2), round(float(py), 2)] for px, py in expanded],
                "confidence": float(np.clip(confidence, 0, 1)),
                "sharpness": _sharpness(region),
                "contrast": float(np.clip(contrast, 0, 1)),
            })
        return sorted(detections, key=lambda item: (item["box"][1], item["box"][0]))

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._session = None
        self._closed = True

    def __enter__(self) -> "LocalVision":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
