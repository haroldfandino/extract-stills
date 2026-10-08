"""Conservative native-pixel evidence for OCR wording changes.

An OCR character alternative is not a visual change when the complete source
glyph area contains the same pixels. Preserve lossless native BGR patches
so temporal consensus can verify that fact during JSON analysis-cache replay.
Never downsample patches or use perceptual hashes as proof of unchanged letters.
"""

from __future__ import annotations

import base64
from functools import lru_cache
import math
import zlib

import cv2
import numpy as np


CONTEXT_PIXELS = 12
MAX_PATCH_PIXELS = 262144
MAX_PATCH_WIDTH = 8192
MAX_PATCH_HEIGHT = 512
MAX_PATCH_BYTES = MAX_PATCH_PIXELS * 3


def _bounds(region: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    polygon = region.get("recognition_polygon", region.get("polygon"))
    if polygon is not None:
        points = np.asarray(polygon, np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            return None
        left, top = np.floor(points.min(axis=0)).astype(int)
        right, bottom = np.ceil(points.max(axis=0)).astype(int) + 1
    else:
        try:
            x, y, w, h = (float(value) for value in region["box"])
        except (ValueError, TypeError, KeyError):
            return None
        if not all(math.isfinite(value) for value in (x, y, w, h)) or min(w, h) <= 0:
            return None
        left, top = math.floor(x), math.floor(y)
        right, bottom = math.ceil(x+w), math.ceil(y+h)
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(width, int(right)), min(height, int(bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def attach_text_appearance(frame_bgr: np.ndarray, regions: list[dict]) -> list[dict]:
    """Return region copies with bounded, JSON-serializable native evidence.

    Region coordinates must use the supplied native frame. Capsule coordinates
    deliberately remain native after callers scale the surrounding detection
    boxes: each capsule includes its frame size and is internally self-contained.
    Oversized patches remain unavailable rather than losing fine character detail
    through resizing. This function never writes files or performs inference.
    """
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
        raise ValueError("Text appearance requires a uint8 H x W x 3 BGR native frame")
    height, width = frame_bgr.shape[:2]
    if min(height, width) < 2:
        raise ValueError("Text appearance requires a nonempty native frame")
    if not regions:
        return []
    result = []
    for source in regions:
        region = dict(source)
        capsule = {"version": 2, "frame_size": [width, height], "available": False}
        bounds = _bounds(source, width, height)
        if bounds is None:
            capsule["reason"] = "invalid_detection_geometry"
        else:
            left, top, right, bottom = bounds
            capsule["glyph_bounds"] = [left, top, right, bottom]
            polygon = source.get("recognition_polygon", source.get("polygon"))
            if polygon is not None:
                capsule["recognition_polygon"] = [[float(x), float(y)] for x, y in polygon]
            x0, y0 = max(0, left-CONTEXT_PIXELS), max(0, top-CONTEXT_PIXELS)
            x1, y1 = min(width, right+CONTEXT_PIXELS), min(height, bottom+CONTEXT_PIXELS)
            w, h = x1-x0, y1-y0
            capsule["bounds"] = [x0, y0, x1, y1]
            if w*h > MAX_PATCH_PIXELS or w > MAX_PATCH_WIDTH or h > MAX_PATCH_HEIGHT:
                capsule["reason"] = "native_patch_exceeds_size_limit"
            elif source.get("clipped", False) or source.get("complete_geometry", True) is False:
                capsule["reason"] = "glyph_geometry_is_incomplete"
            else:
                patch = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1])
                compressed = zlib.compress(patch.tobytes(), level=3)
                capsule.update(available=True, encoding="bgr8-zlib-base64",
                               data=base64.b64encode(compressed).decode("ascii"))
        region["appearance"] = capsule
        result.append(region)
    return result


capture_text_appearance = attach_text_appearance


@lru_cache(maxsize=128)
def _decode(data: str, width: int, height: int) -> np.ndarray:
    if (min(width, height) < 1 or width*height > MAX_PATCH_PIXELS
            or width > MAX_PATCH_WIDTH or height > MAX_PATCH_HEIGHT):
        raise ValueError("Text appearance patch exceeds its native size limit")
    if not isinstance(data, str) or len(data) > ((MAX_PATCH_BYTES+1024+2)//3)*4:
        raise ValueError("Text appearance encoded patch is too large")
    compressed = base64.b64decode(data, validate=True)
    # Bound decompression before checking length, even when cache input is not
    # trusted. An oversized compressed payload must never allocate without bound.
    if len(compressed) > MAX_PATCH_BYTES+1024:
        raise ValueError("Text appearance compressed patch is too large")
    decoder = zlib.decompressobj()
    byte_count = width*height*3
    raw = decoder.decompress(compressed, byte_count+1)
    if len(raw) != byte_count or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Text appearance patch has an invalid decoded size")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def _capsule(region: dict) -> tuple[dict, np.ndarray] | None:
    appearance = region.get("appearance")
    if (not isinstance(appearance, dict) or appearance.get("version") != 2
            or not appearance.get("available") or appearance.get("encoding") != "bgr8-zlib-base64"):
        return None
    if region.get("clipped", False) or region.get("complete_geometry", True) is False:
        return None
    try:
        bounds = appearance["bounds"]
        glyph = appearance["glyph_bounds"]
        size = appearance["frame_size"]
        if (len(bounds) != 4 or len(glyph) != 4 or len(size) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in bounds+glyph+size)):
            return None
        left, top, right, bottom = bounds
        if not 0 <= left < right <= size[0] or not 0 <= top < bottom <= size[1]:
            return None
        if not left <= glyph[0] < glyph[2] <= right or not top <= glyph[1] < glyph[3] <= bottom:
            return None
        patch = _decode(appearance["data"], right-left, bottom-top)
        return appearance, patch
    except (ValueError, TypeError, KeyError, zlib.error):
        return None


def decode_text_appearance(region: dict) -> tuple[dict, np.ndarray] | None:
    """Read a validated lossless native BGR capsule; returned pixels are immutable."""
    return _capsule(region)


def _components(value: dict | list[dict]) -> list[tuple[dict, np.ndarray]] | None:
    if isinstance(value, list):
        result = []
        for region in value:
            decoded = _components(region)
            if decoded is None:
                return None
            result.extend(decoded)
        return result or None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("appearance"), dict):
        decoded = _capsule(value)
        return [decoded] if decoded is not None else None
    if isinstance(value.get("regions"), list):
        return _components(value["regions"])
    return None


def _paint(components: list[tuple[dict, np.ndarray]], bounds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = bounds
    pixels = np.zeros((bottom-top, right-left, 3), np.uint8)
    coverage = np.zeros(pixels.shape[:2], bool)
    for capsule, patch in components:
        x0, y0, x1, y1 = capsule["bounds"]
        x0, y0, x1, y1 = max(x0, left), max(y0, top), min(x1, right), min(y1, bottom)
        if x1 <= x0 or y1 <= y0:
            continue
        bx, by = capsule["bounds"][:2]
        local = patch[y0-by:y1-by, x0-bx:x1-bx]
        target = pixels[y0-top:y1-top, x0-left:x1-left]
        occupied = coverage[y0-top:y1-top, x0-left:x1-left]
        if np.any(target[occupied] != local[occupied]):
            raise ValueError("Overlapping native appearance patches disagree")
        target[:] = local
        occupied[:] = True
    return pixels, coverage


def _clipped_paint_evidence(first: np.ndarray, second: np.ndarray, glyph_area: np.ndarray) -> dict | None:
    """Prove fixed opaque glyph paint while its photographic background moves.

    Compare the UNION of both saturated paint masks, never their intersection:
    a new or removed character must contribute changed pixels. Small threshold
    flicker is permitted only at an existing stroke's one-pixel boundary, with
    at most eight intensity steps in every channel. No alignment or erosion is used.
    """
    delta = np.abs(first.astype(np.float32)-second.astype(np.float32)).max(axis=2)
    total = int(glyph_area.sum())
    for name, mask_a, mask_b in (
            ("clipped_white", (first >= 248).all(axis=2), (second >= 248).all(axis=2)),
            ("clipped_black", (first <= 7).all(axis=2), (second <= 7).all(axis=2))):
        mask_a &= glyph_area
        mask_b &= glyph_area
        counts = int(mask_a.sum()), int(mask_b.sum())
        if any(count < 32 or not .03 <= count / total <= .60 for count in counts):
            continue
        coherent = True
        for mask in (mask_a, mask_b):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            component_sizes = stats[1:, cv2.CC_STAT_AREA]
            if (component_sizes.size == 0 or component_sizes.max() < 16
                    or component_sizes[component_sizes >= 4].sum() < mask.sum() * .80):
                coherent = False
                break
            distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            thickness = np.zeros(len(stats), np.float32)
            np.maximum.at(thickness, labels.ravel(), distances.ravel())
            # Saturated cores of thin strokes can omit a meaningful serif or
            # tiny digit. Such components do not establish unchanged lettering.
            if any(thickness[index+1] < 1.5 for index, area in enumerate(component_sizes) if area >= 4):
                coherent = False
                break
        if not coherent:
            continue
        # Matching one color is not proof for an entire line: an unchanged white
        # PRICE can sit beside a changed yellow amount or an unsaturated word.
        # Any coherent stroke support away from the verified cores makes that
        # subset proof incomplete. Busy photographic texture may also trigger
        # this guard; conservatively decline rather than erase a real change.
        complete_support = True
        for pixels, mask in ((first, mask_a), (second, mask_b)):
            distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
            edges = np.zeros(pixels.shape[:2], bool)
            for channel in cv2.split(pixels):
                edges |= cv2.Canny(channel, 12, 32) != 0
            unproved = (edges & glyph_area & (distance > 3)).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(unproved, connectivity=8)
            if np.any(stats[1:, cv2.CC_STAT_AREA] >= 4):
                complete_support = False
                break
        if not complete_support:
            continue
        union, difference = mask_a | mask_b, mask_a ^ mask_b
        disagreement = float(difference.sum() / union.sum())
        if disagreement > .01:
            continue
        if difference.any():
            distance_a = cv2.distanceTransform((~mask_a).astype(np.uint8), cv2.DIST_L2, 3)
            distance_b = cv2.distanceTransform((~mask_b).astype(np.uint8), cv2.DIST_L2, 3)
            if max(float(distance_b[mask_a].max()), float(distance_a[mask_b].max())) > 1.001:
                continue
        mean, maximum = float(delta[union].mean()), float(delta[union].max())
        painted_delta = np.where(union, delta, 0)
        tile_size = (min(8, delta.shape[1]), min(8, delta.shape[0]))
        local = float(cv2.blur(painted_delta, tile_size, borderType=cv2.BORDER_REPLICATE).max())
        if mean <= .5 and maximum <= 8 and local <= 2:
            return {"paint": name, "paint_pixels": int(union.sum()),
                    "paint_mask_difference": round(disagreement, 6),
                    "mean_delta": round(mean, 6), "max_tile_delta": round(local, 6),
                    "max_pixel_delta": round(maximum, 6)}
    return None


def text_appearance_evidence(first: dict | list[dict], second: dict | list[dict]) -> dict:
    """Explain a pixel-equivalence decision without guessing OCR spelling.

    Require both patches to cover the union of both detected glyph bounds. No
    translation, scaling, affine fitting or erosion can conceal a changed letter.
    Equal pixels with jittering detector boxes compare in fixed source coordinates.
    """
    decoded_first, decoded_second = _components(first), _components(second)
    if decoded_first is None or decoded_second is None:
        return {"same": False, "reason": "native_appearance_unavailable"}
    components = decoded_first+decoded_second
    frame_size = components[0][0]["frame_size"]
    if any(capsule["frame_size"] != frame_size for capsule, _ in components):
        return {"same": False, "reason": "different_native_frame_size"}
    glyphs = [capsule["glyph_bounds"] for capsule, _ in components]
    left, top = min(glyph[0] for glyph in glyphs), min(glyph[1] for glyph in glyphs)
    right, bottom = max(glyph[2] for glyph in glyphs), max(glyph[3] for glyph in glyphs)
    if (right-left)*(bottom-top) > MAX_PATCH_PIXELS:
        return {"same": False, "reason": "native_union_exceeds_size_limit"}
    bounds = [left, top, right, bottom]
    try:
        pixels_a, coverage_a = _paint(decoded_first, bounds)
        pixels_b, coverage_b = _paint(decoded_second, bounds)
    except ValueError:
        return {"same": False, "reason": "overlapping_component_patches_disagree"}
    glyph_area = np.zeros(pixels_a.shape[:2], bool)
    for x0, y0, x1, y1 in glyphs:
        glyph_area[y0-top:y1-top, x0-left:x1-left] = True
    if not np.all((coverage_a & coverage_b)[glyph_area]):
        return {"same": False, "reason": "complete_glyph_union_is_not_covered"}
    delta = np.abs(pixels_a.astype(np.float32)-pixels_b.astype(np.float32)).max(axis=2)
    delta[~glyph_area] = 0
    mean = float(delta[glyph_area].mean())
    maximum = float(delta[glyph_area].max())
    tile_size = (min(8, delta.shape[1]), min(8, delta.shape[0]))
    local = float(cv2.blur(delta, tile_size, borderType=cv2.BORDER_REPLICATE).max())
    same = mean <= .5 and local <= 2.0 and maximum <= 8.0
    if not same:
        foreground = _clipped_paint_evidence(pixels_a, pixels_b, glyph_area)
        if foreground is not None:
            return {"same": True, "reason": "opaque_native_glyph_paint_matches",
                    "comparison_bounds": bounds, **foreground}
    return {"same": bool(same), "reason": "native_pixels_match" if same else "native_glyph_pixels_differ",
            "mean_delta": round(mean, 6), "max_tile_delta": round(local, 6),
            "max_pixel_delta": round(maximum, 6), "comparison_bounds": [left, top, right, bottom]}


def same_text_appearance(first: dict | list[dict], second: dict | list[dict]) -> bool:
    """Return True only when native pixels prove unchanged observed glyphs."""
    return text_appearance_evidence(first, second)["same"]


def clear_text_appearance_cache() -> None:
    """Release cached decoded pixels when an analysis worker finishes."""
    _decode.cache_clear()


def release_text_appearance(records: dict | list[dict]) -> None:
    """Strip native payloads in place after final selection and coverage.

    Accept candidate lists, individual regions or canonical line dictionaries.
    Preserve bounds, frame size and computed diagnostics while explicitly marking
    the removed pixels unavailable. Enriched diagnostic caches should retain the
    payloads until all tracker replay work has finished.
    """
    if isinstance(records, list):
        for record in records:
            release_text_appearance(record)
        clear_text_appearance_cache()
        return
    if not isinstance(records, dict):
        return
    appearance = records.get("appearance")
    if isinstance(appearance, dict) and "data" in appearance:
        appearance.pop("data")
        appearance.update(available=False, payload_discarded=True,
                          reason="payload_released_after_selection")
    vision = records.get("vision")
    if isinstance(vision, dict):
        release_text_appearance(vision.get("text", []))
    if isinstance(records.get("regions"), list):
        release_text_appearance(records["regions"])
