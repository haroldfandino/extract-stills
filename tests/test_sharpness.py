"""Focus outcomes, including failure modes of single resized Laplacian scores."""

import cv2
import numpy as np
import pytest

from stills_tool.sharpness import focus_score, measure_focus


def text_image():
    image = np.full((240, 480, 3), 70, np.uint8)
    cv2.putText(image, "SHARP STILL", (15, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (220, 220, 220), 3, cv2.LINE_AA)
    return image


def motion_blur(image, length, angle):
    kernel = np.zeros((length, length), np.float32)
    center = (length-1)/2
    delta = (np.cos(np.deg2rad(angle))*center, np.sin(np.deg2rad(angle))*center)
    cv2.line(kernel, (round(center-delta[0]), round(center-delta[1])),
             (round(center+delta[0]), round(center+delta[1])), 1, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_directional_motion_blur_loses_focus_in_each_orientation(angle):
    image = text_image()
    assert focus_score(image) > focus_score(motion_blur(image, 15, angle)) * 1.3


def test_focus_decreases_as_defocus_increases():
    image = text_image()
    scores = [focus_score(image)] + [focus_score(cv2.GaussianBlur(image, (0, 0), sigma))
                                    for sigma in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_noisy_soft_frame_does_not_outrank_clean_sharp_frame():
    clean = text_image()
    soft = cv2.GaussianBlur(clean, (0, 0), 4)
    noise = np.random.default_rng(42).normal(0, 25, soft.shape)
    noisy = np.clip(soft.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    clean_measurement, noisy_measurement = measure_focus(clean), measure_focus(noisy)
    assert clean_measurement["focus"] > noisy_measurement["focus"] * 2
    assert noisy_measurement["noise"] > clean_measurement["noise"]
    assert noisy_measurement["reliability"] < clean_measurement["reliability"]


def test_noise_alone_cannot_create_reliable_focus():
    noise = np.random.default_rng(123).normal(128, 15, (240, 480, 3))
    measurement = measure_focus(np.clip(noise, 0, 255).astype(np.uint8))
    assert measurement["noise"] > 0
    assert measurement["focus"] < 1
    assert measurement["reliability"] < 0.5


def test_subject_roi_ranking_is_unaffected_by_sharp_background():
    # A busy background cannot be used to infer foreground focus. The caller
    # supplies tracked subject ROIs, measured at their original pixel scale.
    sharp_subject = text_image()
    soft_subject = cv2.GaussianBlur(sharp_subject, (0, 0), 4)
    y, x = np.indices((480, 960))
    checker = ((x//16 + y//16) % 2 * 210 + 20).astype(np.uint8)
    frame = np.repeat(checker[:, :, None], 3, axis=2)
    frame[120:360, 240:720] = soft_subject
    soft_roi = frame[120:360, 240:720].copy()
    frame[120:360, 240:720] = sharp_subject
    sharp_roi = frame[120:360, 240:720]
    assert focus_score(sharp_roi) > focus_score(soft_roi) * 2
    assert focus_score(soft_roi) == pytest.approx(focus_score(soft_subject))


def test_native_measurement_detects_blur_hidden_by_analysis_downsampling():
    native = np.full((960, 960, 3), 96, np.uint8)
    for y in range(80, 900, 40):
        cv2.putText(native, "NATIVE DETAILS AB123", (80, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(native, (0, 0), 1.2)
    native_gap = focus_score(native) - focus_score(blurred)
    preview_gap = abs(focus_score(cv2.resize(native, (120, 120), interpolation=cv2.INTER_AREA))
                      - focus_score(cv2.resize(blurred, (120, 120), interpolation=cv2.INTER_AREA)))
    assert native_gap > 10
    assert native_gap > preview_gap * 5
    assert measure_focus(native)["measurement_size"] == [960, 960]


def test_tiny_crops_are_not_upsampled_into_false_focus_evidence():
    tiny = cv2.resize(text_image(), (40, 6), interpolation=cv2.INTER_AREA)
    measurement = measure_focus(tiny)
    assert measurement["focus"] == 0
    assert measurement["reliability"] == 0
    assert measurement["measurement_size"] == [40, 6]


def test_partial_but_sharp_text_requires_a_separate_completeness_check():
    # Focus cannot certify wording completeness: cropping a sharp title leaves
    # sharp edges. OCR/state tracking must reject incomplete wording separately.
    image = text_image()
    partial = image[:, :150]
    assert focus_score(image) > 0
    assert focus_score(partial) > focus_score(image)*0.5


def test_contrast_and_blank_padding_do_not_dominate_edge_focus():
    image = text_image()
    padded = np.pad(image, ((100, 100), (200, 200), (0, 0)), constant_values=70)
    reduced_contrast = np.uint8(70 + (image.astype(np.float32)-70)*0.5)
    assert focus_score(padded) == pytest.approx(focus_score(image), rel=0.12)
    assert focus_score(reduced_contrast) == pytest.approx(focus_score(image), rel=0.15)


def test_report_components_are_finite_and_explicit():
    measurement = measure_focus(text_image())
    for component in ("focus", "laplacian", "tenengrad", "edge_acutance", "noise", "reliability"):
        assert np.isfinite(measurement[component])
        assert measurement[component] >= 0
    assert measurement["edge_count"] > 16
    assert len(measurement["scales"]) == 3
    assert focus_score(text_image()) == measurement["focus"]


@pytest.mark.parametrize("size", [(1, 1), (4, 12), (50, 90)])
def test_constant_inputs_have_no_focus_evidence(size):
    measurement = measure_focus(np.full((*size, 3), 128, np.uint8))
    assert measurement["focus"] == 0


@pytest.mark.parametrize("value", [None, np.zeros((10, 10), np.uint8),
                                   np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint16)])
def test_invalid_analysis_pixels_fail_actionably(value):
    with pytest.raises(ValueError, match="BGR"):
        measure_focus(value)
