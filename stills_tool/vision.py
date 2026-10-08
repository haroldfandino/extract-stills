"""Offline, CPU-only face expression and text-region measurements.

Coordinates are in the supplied BGR analysis image. Sharpness is the variance of
the Laplacian on a normalized ROI; it is a relative metric, not a blur probability.
Text detection confidence is DB foreground confidence. Text recognition
confidence is the mean emitted-character CTC probability, not a calibrated
legibility/completeness probability. MediaPipe does not expose a raw per-face
probability, so face confidence describes geometric reliability.
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
    "text_recognition": "PP-OCRv5 Latin mobile recognizer ONNX (Apache-2.0)",
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


def _border_ink(gray: np.ndarray, expanded: np.ndarray, contrast: float) -> bool:
    """Detect glyph strokes cut by the canvas when DB support shrinks inward.

    Only examine borders crossed by the original, unclamped padded polygon.
    Uniform padding at the border does not mean clipped letters. Temporal text
    agreement is still needed when a crop falls exactly between two letters.
    """
    height, width = gray.shape
    x0, y0 = np.floor(expanded.min(axis=0)).astype(int)
    x1, y1 = np.ceil(expanded.max(axis=0)).astype(int)
    left, right = max(0, x0), min(width, x1+1)
    top, bottom = max(0, y0), min(height, y1+1)
    edges = []
    if x0 <= 0 and bottom > top:
        edges.append(gray[top:bottom, 0])
    if x1 >= width-1 and bottom > top:
        edges.append(gray[top:bottom, width-1])
    if y0 <= 0 and right > left:
        edges.append(gray[0, left:right])
    if y1 >= height-1 and right > left:
        edges.append(gray[height-1, left:right])
    threshold = max(.08, contrast * .3)
    return any((float(np.percentile(edge, 95))-float(np.percentile(edge, 5))) / 255 >= threshold
               for edge in edges if edge.size >= 3)


class LocalVision:
    """Load only verified local models; never fetch assets at runtime.

    Explicit model_dir takes precedence over STILLS_MODEL_DIR and bundled assets.
    Instances are intended for one analysis worker and are not thread-safe.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self._landmarker = None
        self._session = None
        self._recognizer = None
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
            for model_id in ("face_landmarker", "text_detector", "text_recognizer", "text_dictionary"):
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
            self._model_paths = paths
            self._model_hint = hint
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

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8:
            raise ValueError("LocalVision requires a uint8 BGR analysis frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 2:
            raise ValueError("LocalVision requires a nonempty H x W x 3 BGR analysis frame")
        return np.ascontiguousarray(frame_bgr)

    def analyze(self, frame_bgr: np.ndarray, recognize: bool = False) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LocalVision has been closed")
        frame_bgr = self._validate_frame(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        warnings: list[str] = []
        faces = self._faces(frame_bgr, gray, warnings)
        text = self._text(frame_bgr, gray)
        if recognize:
            text = self.recognize_text(frame_bgr, text)
        return {"faces": faces, "text": text, "warnings": warnings}

    def recognize_text(self, frame_bgr: np.ndarray, regions: list[dict] | None = None) -> list[dict]:
        """Recognize native-frame crops without running face inference again.

        Supplied regions must use this frame's coordinates. A caller analyzing
        downscaled previews must scale boxes and both polygon fields first.
        The model loads lazily, but all bundled assets are verified at startup.
        """
        if self._closed:
            raise RuntimeError("LocalVision has been closed")
        frame_bgr = self._validate_frame(frame_bgr)
        if regions is None:
            regions = self._text(frame_bgr, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        if not regions:
            return []
        if self._recognizer is None:
            try:
                from .text_ocr import TextRecognizer

                self._recognizer = TextRecognizer(self._model_paths["text_recognizer"],
                                                  self._model_paths["text_dictionary"])
            except Exception as exc:
                raise MissingModelsError(f"Could not load the offline CPU text recognizer: {exc}. "
                                         f"{self._model_hint}") from exc
        return self._recognizer.recognize(frame_bgr, regions)

    def refine_text_regions(self, frame_bgr: np.ndarray, regions: list[dict]) -> list[dict]:
        """Redetect native text bands whose preview boxes can cut whole words.

        Same-baseline pieces and long small-print lines benefit from a higher
        effective detector resolution. Individual large captions keep their
        original geometry. Crops retain context, but only detections overlapping
        the source band replace it; neighboring rows do not leak into the band.
        Artificial crop clipping is reported separately from video clipping.
        """
        if self._closed:
            raise RuntimeError("LocalVision has been closed")
        frame_bgr = self._validate_frame(frame_bgr)
        if not regions:
            return []
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bands: list[list[tuple[int, dict]]] = []
        untouched = []
        for index, region in sorted(enumerate(regions), key=lambda pair: (
                pair[1]["box"][1] + pair[1]["box"][3] / 2, pair[1]["box"][0])):
            x, y, w, h = region["box"]
            if min(w, h) <= 0 or w / h < .85:
                untouched.append(dict(region))
                continue
            candidates = []
            for band_index, band in enumerate(bands):
                band_height = float(np.median([item[1]["box"][3] for item in band]))
                baseline = float(np.median([item[1]["box"][1] + item[1]["box"][3] / 2 for item in band]))
                distance = abs(y + h / 2 - baseline)
                if max(h, band_height) / min(h, band_height) <= 1.8 and distance <= .45 * min(h, band_height):
                    candidates.append((distance, band_index))
            if candidates:
                bands[min(candidates)[1]].append((index, region))
            else:
                bands.append([(index, region)])

        output = untouched
        for band in bands:
            originals = [item[1] for item in band]
            median_height = float(np.median([item["box"][3] for item in originals]))
            fine_print = any(item["box"][3] <= 40 and item["box"][2] / item["box"][3] >= 12
                             for item in originals)
            if len(originals) == 1 and not fine_print:
                output.extend(dict(item) for item in originals)
                continue
            left = min(item["box"][0] for item in originals)
            top = min(item["box"][1] for item in originals)
            right = max(item["box"][0] + item["box"][2] for item in originals)
            bottom = max(item["box"][1] + item["box"][3] for item in originals)
            margin = min(96, max(32, round(median_height)))
            chosen = []
            for attempt in range(2):
                context = margin * (attempt + 1)
                x0, y0 = max(0, int(left-context)), max(0, int(top-context))
                x1, y1 = min(width, int(right+context)), min(height, int(bottom+context))
                if x1 <= x0 or y1 <= y0:
                    break
                crop = frame_bgr[y0:y1, x0:x1]
                local_regions = self._text(crop, gray[y0:y1, x0:x1])
                translated = []
                for local in local_regions:
                    item = dict(local)
                    rx, ry, rw, rh = local["box"]
                    item["box"] = [int(rx+x0), int(ry+y0), int(rw), int(rh)]
                    ix, iy, iw, ih = item["box"]
                    center_y = iy + ih / 2
                    overlap = max(0, min(ix+iw, right)-max(ix, left)) * max(0, min(iy+ih, bottom)-max(iy, top))
                    if (not top-.2*median_height <= center_y <= bottom+.2*median_height
                            or overlap / max(1, iw*ih) < .1):
                        continue
                    for key in ("polygon", "raw_polygon"):
                        if key in local:
                            item[key] = [[float(px+x0), float(py+y0)] for px, py in local[key]]
                    raw = np.asarray(item.get("raw_polygon", item["polygon"]), np.float32)
                    raw_source_clip = bool((raw[:, 0] <= 1).any() or (raw[:, 0] >= width-2).any()
                                           or (raw[:, 1] <= 1).any() or (raw[:, 1] >= height-2).any())
                    padded = np.asarray(item["polygon"], np.float32)
                    source_clip = raw_source_clip or _border_ink(gray, padded, item.get("contrast", 0))
                    # Local detector clipping may refer to an artificial crop
                    # edge. It does not establish that original video text is cut.
                    artificial_clip = bool(local.get("clipped", False) and not source_clip)
                    item.update(clipped=bool(source_clip), refined_native=True,
                                complete_geometry=not artificial_clip and not source_clip,
                                refinement_source_indices=[index for index, _ in band])
                    if artificial_clip:
                        item["geometry_warning"] = "Text reaches a regional crop boundary; complete native glyphs are uncertain."
                    translated.append(item)
                chosen = translated
                if not any(not item["complete_geometry"] and not item["clipped"] for item in chosen):
                    break
            # Avoid replacing a whole band when redetection loses an original
            # region. A DB merge may cover several source pieces, or vice versa.
            def covered(original: dict) -> bool:
                ox, oy, ow, oh = original["box"]
                area = 0
                for item in chosen:
                    ix, iy, iw, ih = item["box"]
                    area += max(0, min(ox+ow, ix+iw)-max(ox, ix)) * max(0, min(oy+oh, iy+ih)-max(oy, iy))
                return area / max(1, ow*oh) >= .20
            if chosen and all(covered(item) for item in originals):
                # DB's expanded rectangles can overlap the next word even when
                # the raw detector supports are disjoint. Bound OCR context at
                # the middle of that gap so "benefits that" cannot acquire the
                # first "s" of a separately detected "support". Keep detection
                # geometry unchanged for scoring and source-coordinate traces.
                horizontal = sorted(chosen, key=lambda item: item["box"][0])
                for previous, following in zip(horizontal, horizontal[1:]):
                    first = np.asarray(previous["raw_polygon"], np.float32)
                    second = np.asarray(following["raw_polygon"], np.float32)
                    first_right, second_left = float(first[:, 0].max()), float(second[:, 0].min())
                    if first_right > second_left:
                        continue
                    boundary = (first_right+second_left) / 2
                    first_crop = np.asarray(previous.get("recognition_polygon", previous["polygon"]), np.float32)
                    second_crop = np.asarray(following.get("recognition_polygon", following["polygon"]), np.float32)
                    if first_crop[:, 0].max() > boundary:
                        first_crop[:, 0] = np.minimum(first_crop[:, 0], boundary)
                        previous["recognition_polygon"] = first_crop.tolist()
                    if second_crop[:, 0].min() < boundary:
                        second_crop[:, 0] = np.maximum(second_crop[:, 0], boundary)
                        following["recognition_polygon"] = second_crop.tolist()
                output.extend(chosen)
            else:
                output.extend(dict(item) for item in originals)

        deduplicated = []
        for item in sorted(output, key=lambda region: (not region.get("refined_native", False),
                                                       region["box"][1], region["box"][0])):
            x, y, w, h = item["box"]
            duplicate = False
            for previous in deduplicated:
                px, py, pw, ph = previous["box"]
                intersection = max(0, min(x+w, px+pw)-max(x, px)) * max(0, min(y+h, py+ph)-max(y, py))
                if intersection / max(1, w*h+pw*ph-intersection) >= .65:
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append(item)
        return sorted(deduplicated, key=lambda item: (item["box"][1], item["box"][0]))

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
            # Retain raw detector support before DB padding or canvas clamping.
            # An expanded box crossing a border does not imply clipped letters.
            raw_polygon = quadrilateral * [width / map_w, height / map_h]
            clipped = bool((quadrilateral[:, 0] <= 1).any() or (quadrilateral[:, 0] >= map_w-2).any()
                           or (quadrilateral[:, 1] <= 1).any() or (quadrilateral[:, 1] >= map_h-2).any())
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
            expanded[:, 0] *= width / map_w
            expanded[:, 1] *= height / map_h
            unclamped = expanded.copy()
            expanded[:, 0] = np.clip(expanded[:, 0], 0, width-1)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, height-1)
            box = _box(expanded, width, height)
            x, y, w, h = box
            region = gray[y:y+h, x:x+w]
            polygon_mask = np.zeros(region.shape, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [(expanded - [x, y]).astype(np.int32)], 1)
            samples = region[polygon_mask != 0]
            contrast = float((np.percentile(samples, 90) - np.percentile(samples, 10)) / 255) if samples.size else 0.0
            clipped = clipped or _border_ink(gray, unclamped, contrast)
            detections.append({
                "box": box,
                "polygon": [[round(float(px), 2), round(float(py), 2)] for px, py in expanded],
                "raw_polygon": [[round(float(px), 2), round(float(py), 2)] for px, py in raw_polygon],
                "clipped": clipped,
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
        self._recognizer = None
        self._closed = True

    def __enter__(self) -> "LocalVision":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
