"""Relative, noise-resistant focus measurements on ORIGINAL supplied pixels.

No resize or upsample is performed. Compare focus within a scene/subject track
at a consistent pixel scale; separately measure native-resolution face/text ROIs
when previews hide blur. A sharp background does not prove a sharp subject.

Report units (luminance is normalized to [0, 1]):
* laplacian: noise-corrected multiscale RMS second derivative, levels/pixel^2.
* tenengrad: noise-corrected multiscale RMS gradient, levels/pixel.
* edge_acutance: contrast-normalized fine edge slope, approximately 1/pixel.
* noise: robust additive-noise standard deviation, normalized levels.
* focus: a continuous relative index, NOT a probability or universal 0-100 grade.
* reliability: noise/sample support, NOT the probability an image is sharp.

Derivative energy is averaged over coherent edge support rather than all pixels,
so extra background texture or blank padding contributes less density bias. The
contrast normalization reduces (but cannot eliminate) dependence on brightness
and edge contrast. Components and reliability are retained for calibration.

This combines established operator families with our own weighting, not a
published calibrated quality model. Pertuz et al. show focus measures depend on
noise, contrast and window size (doi:10.1016/j.patcog.2012.11.011). Immerkær's
noise estimator motivates the 3x3 separable second-difference mask, with robust
MAD/flat-region selection here; fine repeated textures can still resemble noise
(doi:10.1006/cviu.1996.0060). OpenCV supplies Gaussian, Scharr and Laplacian ops:
https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
"""

from __future__ import annotations

from functools import lru_cache
import math

import cv2
import numpy as np

_SCALES = (0.6, 1.2, 2.4)
_WEIGHTS = (0.6, 0.3, 0.1)
_NOISE_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _derivatives(gray: np.ndarray, sigma: float):
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0, scale=1/32, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1, scale=1/32, borderType=cv2.BORDER_REFLECT_101)
    gradient_squared = gx*gx + gy*gy
    laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    return smoothed, gradient_squared, laplacian


@lru_cache(maxsize=3)
def _noise_gains(sigma: float) -> tuple[float, float]:
    # Impulse response gives the exact noise variance gain of each implemented
    # smoothing/derivative chain, including OpenCV's selected Gaussian support.
    impulse = np.zeros((41, 41), np.float32)
    impulse[20, 20] = 1
    _, gradient_squared, laplacian = _derivatives(impulse, sigma)
    return float(gradient_squared.sum()), float(np.square(laplacian).sum())


def _empty(noise: float = 0.0, shape=(0, 0)) -> dict:
    return {"focus": 0.0, "laplacian": 0.0, "tenengrad": 0.0,
            "edge_acutance": 0.0, "noise": float(noise), "scale_retention": 0.0,
            "edge_fraction": 0.0, "edge_count": 0, "reliability": 0.0,
            "measurement_size": [int(shape[1]), int(shape[0])], "scales": []}


def measure_focus(frame_bgr: np.ndarray) -> dict:
    """Measure focus without making any resizing or subject/wording assumptions.

    Input is a nonempty uint8 BGR ndarray. Tiny (<8 px on a side), flat, gradient
    or noise-only images with insufficient coherent edges return focus=0 and
    low reliability; this is insufficient focus evidence, not proof of blur.
    Callers should not mix native ROI values with resized global measurements
    as though they were the same quantity, nor use focus to infer complete text.
    Fine periodic textures, aliasing, sharpening halos and changing ROI content
    can affect ordering. Use temporal tracks, subject coverage and OCR readability
    independently; zero focus may mean absent evidence rather than poor optics.
    """
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8:
        raise ValueError("Focus measurement requires a uint8 BGR image")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 1:
        raise ValueError("Focus measurement requires a nonempty H x W x 3 BGR image")
    height, width = frame_bgr.shape[:2]
    if min(height, width) < 8:
        return _empty(shape=(height, width))
    gray = cv2.cvtColor(np.ascontiguousarray(frame_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    coarse, coarse_energy, _ = _derivatives(gray, 1.2)
    coarse_gradient = np.sqrt(coarse_energy)
    # Borders can invent gradients in very tight subject crops. Exclude a small
    # native-pixel rim instead of padding/resizing the image.
    interior = np.zeros(gray.shape, np.bool_)
    rim = min(3, (min(height, width)-1)//2)
    interior[rim:height-rim, rim:width-rim] = True
    flat = interior & (coarse_gradient <= np.percentile(coarse_gradient[interior], 35))
    response = cv2.filter2D(gray, cv2.CV_32F, _NOISE_MASK, borderType=cv2.BORDER_REFLECT_101)
    residual = response[flat]
    noise = float(np.median(np.abs(residual - np.median(residual))) / (6 * 0.67448975)) if residual.size else 0.0
    noise = max(0.0, min(noise, 1.0))

    # Local contrast at a coarse scale suppresses random fine-noise excursions.
    # Keep a native 17-pixel neighborhood (or the largest valid smaller odd size).
    neighborhood = min(17, min(height, width)//2*2-1)
    kernel = np.ones((max(3, neighborhood), max(3, neighborhood)), np.uint8)
    local_contrast = cv2.dilate(coarse, kernel) - cv2.erode(coarse, kernel)
    gradient_gain, _ = _noise_gains(1.2)
    minimum_gradient = max(0.004, 2.8 * noise * math.sqrt(gradient_gain))
    support = interior & (coarse_gradient > minimum_gradient) & (local_contrast > max(0.025, 4*noise))
    count = int(support.sum())
    if count < 16:
        return _empty(noise, (height, width))
    contrast = max(float(np.median(local_contrast[support])), 0.025)
    noise_reliability = 1 / (1 + (3 * noise / contrast)**2)
    sample_reliability = min(1.0, count/64)
    reliability = float(noise_reliability * sample_reliability)
    scales = []
    acutance = 0.0
    laplacian_energy = tenengrad_energy = 0.0
    for sigma, weight in zip(_SCALES, _WEIGHTS):
        _, energy, lap = _derivatives(gray, sigma)
        ten_gain, lap_gain = _noise_gains(sigma)
        ten_squared = max(0.0, float(np.mean(energy[support])) - noise*noise*ten_gain)
        lap_squared = max(0.0, float(np.mean(np.square(lap[support]))) - noise*noise*lap_gain)
        tenengrad_energy += weight * ten_squared
        laplacian_energy += weight * lap_squared
        scales.append({"sigma": sigma, "tenengrad": math.sqrt(ten_squared), "laplacian": math.sqrt(lap_squared)})
        if sigma == _SCALES[0]:
            corrected_gradient = np.sqrt(np.maximum(energy[support] - noise*noise*ten_gain, 0))
            normalized_slope = corrected_gradient / np.maximum(local_contrast[support], 0.025)
            acutance = float(np.percentile(normalized_slope, 65))

    tenengrad = math.sqrt(tenengrad_energy)
    laplacian = math.sqrt(laplacian_energy)
    fine, broad = scales[0]["tenengrad"], scales[-1]["tenengrad"]
    retention = max(0.0, min(1.0, 1-broad/max(fine, 1e-9)))
    focus = (100 * (0.6*acutance + 0.25*laplacian/contrast + 0.15*tenengrad/contrast)
             * (0.65 + 0.35*retention) * reliability)
    return {"focus": float(max(0, focus)), "laplacian": float(laplacian), "tenengrad": float(tenengrad),
            "edge_acutance": acutance, "noise": noise, "scale_retention": retention,
            "edge_fraction": count/(height*width), "edge_count": count, "reliability": reliability,
            "measurement_size": [width, height], "scales": scales}


def focus_score(frame_bgr: np.ndarray) -> float:
    """Return the continuous relative focus index; see measure_focus for units."""
    return measure_focus(frame_bgr)["focus"]
