"""Content recognition, native crop geometry and offline asset regressions."""

import hashlib
import json
from pathlib import Path
import socket

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from stills_tool.text_ocr import TextRecognizer, decode_ctc, load_characters, rectify_region, region_is_clipped
from stills_tool.vision import LocalVision, MissingModelsError


ROOT = Path(__file__).resolve().parents[1]


def _title(text="BEST STILLS ARE READY"):
    frame = np.full((540, 960, 3), 55, np.uint8)
    cv2.putText(frame, text, (50, 260), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (250, 250, 250), 3, cv2.LINE_AA)
    return frame


@pytest.fixture(scope="module")
def recognizer():
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
    if any(not (ROOT / "models" / item["filename"]).is_file() for item in manifest["models"]):
        pytest.skip("Provision all local assets with python scripts/setup_models.py")
    with LocalVision() as analyzer:
        yield analyzer


def test_exact_upstream_dictionary_preserves_unicode_duplicate_indexes_and_space():
    dictionary = ROOT / "models" / "pp_ocrv5_latin_rec.yml"
    if not dictionary.exists():
        pytest.skip("Pinned character dictionary unavailable")
    characters = load_characters(dictionary)
    assert len(characters) == 838
    assert characters[0] == "" and characters[-1] == " "
    assert "ñ" in characters and "é" in characters
    assert characters.count("ñ") == 2, "Duplicate upstream dictionary entries have distinct class indexes"
    assert "'" in characters and '"' in characters and "\\" in characters


def test_dictionary_rejects_mismatched_or_unsafe_configuration(tmp_path):
    dictionary = tmp_path / "dictionary.yml"
    dictionary.write_text("PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - !!python/object:bad\n")
    with pytest.raises(ValueError, match="non-character"):
        load_characters(dictionary)
    dictionary.write_text("PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - a\n")
    with pytest.raises(ValueError, match="does not match"):
        load_characters(dictionary)


def test_ctc_retains_repeated_letters_spaces_and_accented_words():
    characters = [""] + list(dict.fromkeys("café mañana"))
    wanted = "café mañana"
    indexes = [0]
    for char in wanted:
        index = characters.index(char)
        indexes.extend([index, index, 0])
    probability = np.zeros((1, len(indexes), len(characters)), np.float32)
    for position, index in enumerate(indexes):
        probability[0, position, index] = .97
    text, confidence = decode_ctc(probability, characters)
    assert text == wanted
    assert confidence == pytest.approx(.97)
    probability[:] = 0
    probability[:, :, 0] = 1
    assert decode_ctc(probability, characters) == ("", 0)


@pytest.mark.parametrize("prediction", [np.zeros((1, 4, 9)), np.full((1, 4, 2), np.nan),
                                         np.full((1, 4, 2), 2), np.zeros((1, 0, 2))])
def test_ctc_rejects_invalid_runtime_outputs(prediction):
    with pytest.raises(RuntimeError):
        decode_ctc(prediction, ["", "a"])


def test_preprocessing_preserves_native_line_aspect_and_bgr_channel_order():
    crop = np.zeros((24, 240, 3), np.uint8)
    crop[:, :, 0] = 255
    tensor = TextRecognizer.prepare(crop)
    assert tensor.shape == (1, 3, 48, 480)
    assert tensor.dtype == np.float32
    assert np.all(tensor[0, 0] == 1) and np.all(tensor[0, 2] == -1)
    short = TextRecognizer.prepare(crop[:, :24])
    assert short.shape == (1, 3, 48, 320)
    assert np.all(short[:, :, :, 48:] == 0)


def test_clipping_uses_raw_support_instead_of_expanded_padded_polygon():
    padded = {"box": [0, 30, 130, 60], "polygon": [[0, 30], [129, 30], [129, 89], [0, 89]],
              "raw_polygon": [[12, 42], [117, 42], [117, 77], [12, 77]]}
    assert not region_is_clipped(padded, (160, 320, 3))
    cropped = dict(padded, raw_polygon=[[0, 42], [117, 42], [117, 77], [0, 77]])
    assert region_is_clipped(cropped, (160, 320, 3))
    assert region_is_clipped(dict(padded, clipped=True), (160, 320, 3))


def test_perspective_crop_retains_native_resolution_and_unordered_quad():
    frame = np.full((160, 320, 3), 90, np.uint8)
    cv2.rectangle(frame, (40, 40), (239, 89), (250, 250, 250), -1)
    region = {"polygon": [[239, 89], [40, 89], [40, 40], [239, 40]]}
    crop = rectify_region(frame, region)
    assert crop.shape == (49, 199, 3)
    assert float(crop.mean()) > 240
    with pytest.raises(ValueError, match="degenerate"):
        rectify_region(frame, {"polygon": [[40, 40]] * 4})
    with pytest.raises(ValueError, match="finite"):
        rectify_region(frame, {"polygon": [[float("nan"), 40]] * 4})


@pytest.mark.parametrize("damaged_id", ["text_recognizer", "text_dictionary"])
def test_missing_or_tampered_recognition_asset_fails_before_runtime_startup(tmp_path, monkeypatch, damaged_id):
    def forbidden(*args, **kwargs):
        pytest.fail("Asset verification attempted runtime networking")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    entries = []
    for model_id in ("face_landmarker", "text_detector", "text_recognizer", "text_dictionary"):
        path = tmp_path / (model_id + ".asset")
        path.write_bytes(b"valid")
        entries.append({"id": model_id, "filename": path.name, "size_bytes": 5,
                        "sha256": hashlib.sha256(b"valid").hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": entries}))
    (tmp_path / (damaged_id + ".asset")).write_bytes(b"wrong")
    with pytest.raises(MissingModelsError, match="verification failed.*setup_models.py"):
        LocalVision(tmp_path)
    (tmp_path / (damaged_id + ".asset")).unlink()
    with pytest.raises(MissingModelsError, match="Missing local model.*setup_models.py"):
        LocalVision(tmp_path)


def test_real_offline_recognition_reads_multiword_english_and_accented_spanish(recognizer, monkeypatch):
    from matplotlib import font_manager

    def forbidden(*args, **kwargs):
        pytest.fail("Recognition attempted runtime networking")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    image = Image.new("RGB", (1280, 720), (55, 55, 55))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 48)
    for text, y in [("BEST STILLS ARE READY", 200), ("Español: mañana, acción y café", 350)]:
        draw.text((60, y), text, font=font, fill=(250, 250, 250))
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    recognized = recognizer.recognize_text(frame)
    assert [item["text"] for item in recognized] == ["BEST STILLS ARE READY", "Español: mañana, acción y café"]
    assert all(item["recognition_confidence"] > .9 and not item["clipped"] for item in recognized)
    assert recognizer.recognize_text(np.zeros_like(frame)) == []


def test_recognition_on_supplied_native_regions_does_not_repeat_detection_or_faces(recognizer, monkeypatch):
    image = _title()
    regions = recognizer.analyze(image)["text"]

    def forbidden(*args, **kwargs):
        pytest.fail("Existing native regions reran vision detection")

    monkeypatch.setattr(recognizer, "_text", forbidden)
    monkeypatch.setattr(recognizer, "_faces", forbidden)
    result = recognizer.recognize_text(image, regions)
    assert result[0]["text"] == "BEST STILLS ARE READY"
    for key in ("box", "polygon", "raw_polygon", "confidence", "contrast", "sharpness"):
        assert result[0][key] == regions[0][key]
    blurred = recognizer.recognize_text(cv2.GaussianBlur(image, (0, 0), 12), regions)
    assert blurred[0]["recognition_confidence"] < result[0]["recognition_confidence"] * .7
    assert recognizer.recognize_text(image, []) == []


def test_upside_down_and_vertical_titles_are_recognized(recognizer):
    image = _title()
    upside_down = recognizer.recognize_text(cv2.rotate(image, cv2.ROTATE_180))
    assert " ".join(item["text"] for item in upside_down) == "BEST STILLS ARE READY"
    vertical = recognizer.recognize_text(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    assert " ".join(item["text"] for item in vertical) == "BEST STILLS ARE READY"


def test_border_clipped_title_is_flagged_by_original_detector_geometry(recognizer):
    image = _title()
    cropped = image[:, 95:]
    result = recognizer.recognize_text(cropped)
    assert result and any(item["clipped"] for item in result)


def test_public_recognition_validates_input_and_closed_state(recognizer):
    with pytest.raises(ValueError, match="uint8"):
        recognizer.recognize_text(np.zeros((32, 32, 3), np.float32))
    with pytest.raises(ValueError, match="nonempty"):
        recognizer.recognize_text(np.zeros((32, 32), np.uint8))
    closed = LocalVision.__new__(LocalVision)
    closed._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        closed.recognize_text(_title())


def test_small_number_context_recovers_glyphs_outside_a_tight_detector_crop(recognizer):
    frame = np.full((160, 320, 3), 245, np.uint8)
    cv2.putText(frame, "15", (125, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (8, 8, 8), 3, cv2.LINE_AA)
    region = {"box": [133, 58, 35, 22], "confidence": .8, "clipped": False}
    recognizer.recognize_text(frame, [])  # Public empty operation remains cheap.
    if recognizer._recognizer is None:
        recognizer.recognize_text(frame)
    raw_text, _ = recognizer._recognizer._line(rectify_region(frame, region))
    assert raw_text != "15", "The regression fixture must exercise an incomplete detector crop"
    result = recognizer.recognize_text(frame, [region])[0]
    assert result["text"] == "15" and result["recognition_confidence"] > .95
    assert result["box"] == region["box"] and not result["clipped"]
    assert "recognition_polygon" in result


def test_context_padding_never_clears_original_canvas_clipping(recognizer):
    frame = np.full((160, 320, 3), 245, np.uint8)
    cv2.putText(frame, "15", (-15, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (8, 8, 8), 3, cv2.LINE_AA)
    region = {"box": [0, 58, 30, 22], "confidence": .8,
              "raw_polygon": [[0, 58], [29, 58], [29, 79], [0, 79]]}
    result = recognizer.recognize_text(frame, [region])[0]
    assert result["clipped"], "Additional recognition padding must never certify cut canvas glyphs"


def test_native_fatty_superscript_is_fifteen_not_a_letter(recognizer):
    cache = (ROOT / "validation/samples_final/Fatty15_RickiLake2_Amazon_15_HD_Social_H264"
             / "run_20261008_102123/Fatty15_RickiLake2_Amazon_15_HD_Social_H264_still_007_frame_00000204.png")
    if cache.is_file():
        frame = cv2.imread(str(cache))
    else:
        source = Path(r"D:\Filmkraft\Tools\Extract_stills\input\Fatty15_RickiLake2_Amazon_15_HD_Social_H264.mp4")
        if not source.is_file():
            pytest.skip("User-provided video/native cached still unavailable")
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 204)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            pytest.skip("User-provided video codec unavailable")
    assert frame is not None and frame.shape[:2] == (1080, 1920)
    region = {"box": [782, 218, 24, 18], "confidence": .623, "clipped": False}
    result = recognizer.recognize_text(frame, [region])[0]
    assert result["text"] == "15" and result["recognition_confidence"] > .95
    assert result["box"] == region["box"] and not result["clipped"]


def _region(x, y, width, height, clipped=False):
    polygon = [[x, y], [x+width-1, y], [x+width-1, y+height-1], [x, y+height-1]]
    return {"box": [x, y, width, height], "polygon": polygon, "raw_polygon": polygon,
            "confidence": .95, "contrast": .8, "sharpness": 20, "clipped": clipped}


def _geometry_analyzer(detector):
    analyzer = LocalVision.__new__(LocalVision)
    analyzer._closed = False
    analyzer._text = detector
    return analyzer


def test_regional_refinement_translates_native_coordinates_and_retries_artificial_edges():
    calls = []

    def detector(frame, gray):
        calls.append(frame.shape)
        offset = 32 if len(calls) == 1 else 64
        return [_region(offset, offset, 320, 20, clipped=len(calls) == 1)]

    analyzer = _geometry_analyzer(detector)
    result = analyzer.refine_text_regions(np.zeros((300, 700, 3), np.uint8), [_region(100, 80, 320, 20)])
    assert len(calls) == 2
    assert result[0]["box"] == [100, 80, 320, 20]
    assert result[0]["raw_polygon"][0] == [100, 80]
    assert result[0]["refined_native"] and result[0]["complete_geometry"]
    assert not result[0]["clipped"]
    assert "geometry_warning" not in result[0]


def test_persistent_artificial_crop_clipping_is_separate_from_video_clipping():
    calls = []

    def detector(frame, gray):
        calls.append(frame.shape)
        offset = 32 if len(calls) == 1 else 64
        return [_region(offset, offset, 320, 20, clipped=True)]

    analyzer = _geometry_analyzer(detector)
    result = analyzer.refine_text_regions(np.zeros((300, 700, 3), np.uint8), [_region(100, 80, 320, 20)])
    assert len(calls) == 2
    assert not result[0]["clipped"] and not result[0]["complete_geometry"]
    assert "regional crop boundary" in result[0]["geometry_warning"]


def test_refinement_retains_genuine_video_corner_clipping():
    analyzer = _geometry_analyzer(lambda frame, gray: [_region(0, 0, 320, 20, clipped=True)])
    result = analyzer.refine_text_regions(np.zeros((200, 700, 3), np.uint8), [_region(0, 0, 320, 20)])
    assert result[0]["box"] == [0, 0, 320, 20]
    assert result[0]["clipped"] and not result[0]["complete_geometry"]


def test_neighboring_fine_print_rows_do_not_leak_or_duplicate_across_context_crops():
    calls = []

    def detector(frame, gray):
        calls.append(frame.shape)
        ys = (32, 60) if len(calls) == 1 else (4, 32)
        return [_region(32, y, 320, 20) for y in ys]

    analyzer = _geometry_analyzer(detector)
    originals = [_region(100, 80, 320, 20), _region(100, 108, 320, 20)]
    result = analyzer.refine_text_regions(np.zeros((300, 700, 3), np.uint8), originals)
    assert [item["box"] for item in result] == [[100, 80, 320, 20], [100, 108, 320, 20]]
    assert all(item["refined_native"] for item in result)


def test_single_large_caption_and_vertical_text_keep_original_geometry():
    def forbidden(*args, **kwargs):
        pytest.fail("Single large or vertical captions reran the detector")

    analyzer = _geometry_analyzer(forbidden)
    originals = [_region(100, 80, 400, 80), _region(550, 30, 20, 190)]
    result = analyzer.refine_text_regions(np.zeros((300, 700, 3), np.uint8), originals)
    assert sorted(result, key=lambda item: item["box"][0]) == originals
    assert analyzer.refine_text_regions(np.zeros((300, 700, 3), np.uint8), []) == []


def test_native_redetection_recovers_whole_words_from_incomplete_source_boxes(recognizer):
    frame = np.full((720, 1280, 3), 55, np.uint8)
    for text, x in (("COMPLETE", 160), ("WORDS", 480)):
        cv2.putText(frame, text, (x, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (250, 250, 250), 3, cv2.LINE_AA)
    # A preview detector found only part of the first word; contextual native
    # detection must locate all its glyphs without inventing neighboring words.
    originals = [_region(160, 210, 150, 44), _region(480, 210, 165, 44)]
    refined = recognizer.refine_text_regions(frame, originals)
    result = sorted(recognizer.recognize_text(frame, refined), key=lambda item: item["box"][0])
    assert " ".join(item["text"] for item in result) == "COMPLETE WORDS"
    assert all(item["refined_native"] and item["complete_geometry"] for item in result)


@pytest.mark.parametrize("frame_index", [108, 114])
def test_native_band_refinement_preserves_complete_fatty_headline_words(recognizer, frame_index):
    cache_root = ROOT / "validation"
    cache = next(cache_root.glob(f"**/Fatty15_RickiLake2_Amazon_15_HD_Social_H264_still*frame_{frame_index:08d}.png"), None)
    if cache is not None:
        frame = cv2.imread(str(cache))
    else:
        source = Path(r"D:\Filmkraft\Tools\Extract_stills\input\Fatty15_RickiLake2_Amazon_15_HD_Social_H264.mp4")
        if not source.is_file():
            pytest.skip("User-provided Fatty15 video/native still unavailable")
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            pytest.skip("User-provided video codec unavailable")
    if frame is None or frame.shape[:2] != (1080, 1920):
        pytest.skip("Native Fatty15 still unavailable")
    regions = recognizer.analyze(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))["text"]
    native = []
    for region in regions:
        item = dict(region, box=[value*2 for value in region["box"]])
        for key in ("polygon", "raw_polygon"):
            item[key] = [[x*2, y*2] for x, y in region[key]]
        native.append(item)
    refined = recognizer.refine_text_regions(frame, native)
    recognized = recognizer.recognize_text(frame, refined)
    headline = sorted([item for item in recognized if 300 < item["box"][1] < 500], key=lambda item: item["box"][0])
    assert " ".join(item["text"] for item in headline) == "36+ cellular benefits that support:"
    assert all(item["refined_native"] and item["complete_geometry"] and not item["clipped"] for item in headline)
    footer = [item for item in recognized if "These statements" in item["text"]]
    assert footer and footer[0]["text"].endswith("Food and Drug Administration.")
