"""Edit boundaries survive motion, flashes and short advertising inserts."""
import cv2
import numpy as np
import pytest

from stills_tool.shots import ShotDetector


def picture(seed, height=128, width=224):
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = rng.integers(15, 180, 3)
    for _ in range(30):
        point = (int(rng.integers(width)), int(rng.integers(height)))
        cv2.circle(image, point, int(rng.integers(4, 25)), tuple(map(int, rng.integers(20, 255, 3))), -1)
    return image


def detect(frames, fps=24):
    detector = ShotDetector(fps)
    for index, frame in enumerate(frames):
        assert detector.process_frame(index, frame) == []
    return detector.finish(len(frames)), detector


def test_hard_cuts_and_two_frame_insert():
    a, b, c = (picture(i) for i in range(3))
    cuts, _ = detect([a] * 15 + [b] * 2 + [c] * 15)
    assert cuts == [15, 17]


def test_one_frame_montage_is_not_consolidated_into_a_wipe():
    a, b, c, d = (picture(i) for i in (1, 2, 3, 4))
    assert detect([a] * 12 + [b] + [c] + [d] * 12)[0] == [12, 13, 14]


def test_blank_leader_and_tail_do_not_create_empty_shots():
    image = picture(1)
    blank = np.zeros_like(image)
    assert detect([blank] * 6 + [image] * 20 + [blank] * 6)[0] == []


def test_brief_gradient_entrance_is_attached_to_first_shot():
    gradient = np.tile(np.linspace(80, 180, 224, dtype=np.uint8), (128, 1))
    entrance = np.repeat(gradient[..., None], 3, axis=2)
    assert detect([entrance] * 3 + [picture(7)] * 24)[0] == []


def test_continuous_blurred_object_movement_inside_static_canvas():
    base = picture(6, height=180, width=74)
    frames = []
    for index in range(32):
        y = int(28 + 20 * np.tanh((index - 16) / 2))
        inset = base[y:y + 120].copy()
        if 14 <= index <= 18:
            inset = cv2.GaussianBlur(inset, (3, 15), 0, sigmaY=4)
        frame = np.full((128, 224, 3), 240, np.uint8)
        frame[4:124, 100:174] = inset
        cv2.putText(frame, "BRAND", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 60, 50), 1)
        frames.append(frame)
    assert detect(frames)[0] == []


def test_same_palette_different_camera_angle():
    a = picture(7)
    b = cv2.flip(a, 1)
    cuts, _ = detect([a] * 16 + [b] * 16)
    assert cuts == [16]


def test_portrait_inset_camera_cut_on_an_unchanged_branded_canvas():
    first, second = picture(3, height=120, width=64), picture(8, height=120, width=64)
    frames = []
    for inset in [first] * 15 + [second] * 15:
        frame = np.full((128, 224, 3), 240, np.uint8)
        frame[4:124, 100:164] = inset
        cv2.putText(frame, "BRAND", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 60, 50), 1)
        frames.append(frame)
    assert detect(frames)[0] == [15]


@pytest.mark.parametrize("value", [0, 255])
def test_exposure_flash_returns_to_same_camera_view(value):
    a = picture(12)
    flash = np.full_like(a, value)
    cuts, detector = detect([a] * 16 + [flash] * 2 + [a] * 16)
    assert cuts == []
    assert any("flash" in x.get("reason", "") for x in detector.diagnostics)


def test_flash_leading_to_a_different_view_keeps_edit():
    a, b = picture(3), picture(8)
    cuts, _ = detect([a] * 16 + [np.full_like(a, 255)] + [b] * 16)
    assert cuts
    assert all(16 <= x <= 17 for x in cuts)


def test_small_subtitle_and_logo_changes_are_not_camera_cuts():
    frames = []
    for index in range(48):
        frame = picture(2)
        cv2.rectangle(frame, (12, 100), (200, 120), (0, 0, 0), -1)
        cv2.putText(frame, "PRICE 25" if index < 24 else "PRICE 75", (16, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.rectangle(frame, (index % 20, 0), (index % 20 + 12, 10), (255, 0, 0), -1)
        frames.append(frame)
    assert detect(frames)[0] == []


def test_fast_horizontal_pan_is_one_shot():
    base = picture(24, width=480)
    frames = [base[:, 6 * i:6 * i + 224].copy() for i in range(40)]
    assert detect(frames)[0] == []


def test_slow_zoom_is_one_shot():
    base = picture(34, height=256, width=448)
    frames = []
    for index in range(40):
        crop = index + 1
        frames.append(cv2.resize(base[crop:-crop, crop:-crop], (224, 128)))
    assert detect(frames)[0] == []


def test_cross_dissolve_yields_one_midpoint_boundary():
    a, b = picture(5), picture(15)
    transition = [cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
                  for alpha in np.linspace(0, 1, 12)]
    cuts, detector = detect([a] * 16 + transition + [b] * 16)
    assert len(cuts) == 1
    assert 19 <= cuts[0] <= 24
    assert any(x["kind"] == "dissolve" and x["accepted"] for x in detector.diagnostics)


def test_fade_through_black_yields_one_edit():
    a, b = picture(20), picture(21)
    fade = [np.uint8(a.astype(float) * alpha) for alpha in np.linspace(1, 0, 8)]
    fade += [np.uint8(b.astype(float) * alpha) for alpha in np.linspace(0, 1, 8)]
    cuts, _ = detect([a] * 16 + fade + [b] * 16)
    assert len(cuts) == 1
    assert 21 <= cuts[0] <= 26


def test_uniform_color_edit_is_a_shot():
    a = np.full((128, 224, 3), (55, 70, 180), np.uint8)
    b = np.full_like(a, (140, 190, 20))
    assert detect([a] * 12 + [b] * 12)[0] == [12]


def test_empty_single_and_finalisation_contract():
    detector = ShotDetector()
    assert detector.finish() == []
    cuts, detector = detect([picture(1)] * 10)
    assert cuts == detector.finish() == []
    with pytest.raises(RuntimeError):
        detector.process_frame(10, picture(1))


def test_does_not_keep_full_video_frames():
    _, detector = detect([picture(1)] * 200)
    assert len(detector._history) <= 24
    assert max(x[1].shape[1] for x in detector._history) <= 320
