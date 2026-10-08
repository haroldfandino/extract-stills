"""Temporal evidence for readable, complete observed text.

OCR confidence and repeated wording are proxies, not proof of original opacity,
typography or semantic completeness. Region geometry stays separate: a stable
logo never substitutes for a changing subtitle. State IDs describe chronological
occurrences, so A -> B -> A creates three states.
"""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher
import math
import re
import unicodedata

from .text_appearance import same_text_appearance
from .text_pattern import text_pattern_evidence

MIN_CONFIDENCE = 0.55
MIN_CONTRAST = 0.08
MIN_CONFIRMED_SPAN = 0.05
BRIEF_FRAGMENT_SECONDS = 0.20


def normalize_text(value: str) -> str:
    """Normalize spacing/case without changing accents, words, numbers or order."""
    tokens = unicodedata.normalize("NFKC", str(value)).casefold().split()
    # OCR often includes an adjacent trademark stroke as a trailing quote.
    # Boundary punctuation is presentation, while embedded decimal/currency
    # characters and every digit remain part of the wording identity.
    return " ".join(token.strip("\"'‘’“”.,:;!?*¡¿ ") for token in tokens).strip()


def _box(region: dict) -> tuple[float, float, float, float]:
    values = region.get("box", (0, 0, 0, 0))
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return 0.0, 0.0, 0.0, 0.0
    return x, y, max(0.0, width), max(0.0, height)


def _match_cost(first: tuple, second: tuple, diagonal: float) -> float | None:
    x, y, width, height = first
    other_x, other_y, other_width, other_height = second
    if not min(width, height, other_width, other_height):
        return None
    if min(height, other_height) / max(height, other_height) < 0.40:
        return None
    if min(width, other_width) / max(width, other_width) < 0.08:
        return None
    aspect, other_aspect = width / height, other_width / other_height
    if min(aspect, other_aspect) / max(aspect, other_aspect) < 0.25:
        return None
    dy = abs(y + height / 2 - other_y - other_height / 2)
    if dy > max(8, 0.75 * min(height, other_height)):
        return None
    dx = abs(x + width / 2 - other_x - other_width / 2)
    left_match = abs(x - other_x) < max(8, min(width, other_width) * 0.25)
    if dx > max(width, other_width, diagonal * 0.18) and not left_match:
        return None
    return (dx + dy * 3 + abs(height - other_height)) / max(1, diagonal)


def _clipped(region: dict, video: dict) -> bool:
    if region.get("clipped", False):
        return True
    # Padding around detected glyphs can touch an edge while all glyphs remain
    # visible. Trust an explicit False from native OCR; inspect raw support only
    # when the provider has not supplied its own completeness measurement.
    if "clipped" in region:
        return False
    polygon = region.get("raw_polygon")
    if not polygon:
        return False
    width, height = video.get("width", 0), video.get("height", 0)
    if not width or not height:
        return False
    try:
        return any(x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1 for x, y in polygon)
    except (ValueError, TypeError):
        return True


def _similar_jitter(first: str, second: str) -> bool:
    if not first or not second or any(character.isdigit() for character in first + second):
        return False
    if len(first.split()) != len(second.split()) or min(len(first), len(second)) < 4:
        return False
    return SequenceMatcher(None, first, second, autojunk=False).ratio() >= 0.80


def _fragment_of(shorter: str, longer: str) -> bool:
    if not shorter or len(shorter) >= len(longer):
        return False
    # Changing a price or other number is meaningful even when one string is a
    # prefix of the other; OCR-based growth must never erase such observations.
    if re.findall(r"\d+(?:[.,]\d+)*", shorter) != re.findall(r"\d+(?:[.,]\d+)*", longer):
        return False
    return longer.startswith(shorter) or longer.endswith(shorter)


def _canonical_lines(regions: list[dict]) -> list[dict]:
    """Join only neighboring OCR segments on the same visual baseline."""
    contained = set()
    for index, child in enumerate(regions):
        x, y, width, height = child["box"]
        for parent in regions:
            if child is parent or not child["text"] or not parent["text"]:
                continue
            px, py, pw, ph = parent["box"]
            if pw * ph <= width * height * 1.10:
                continue
            intersection = max(0, min(x + width, px + pw) - max(x, px))
            intersection *= max(0, min(y + height, py + ph) - max(y, py))
            if intersection / max(1, width * height) < 0.90:
                continue
            if abs(y + height / 2 - py - ph / 2) > min(height, ph) * 0.35:
                continue
            if child["text"] not in parent["text"]:
                continue
            child_capsule, parent_capsule = child.get("appearance", {}), parent.get("appearance", {})
            if not child_capsule.get("available") or not parent_capsule.get("available"):
                continue
            glyph = child_capsule.get("glyph_bounds")
            bounds = parent_capsule.get("bounds", [])
            if not glyph or len(bounds) != 4 or not (
                bounds[0] <= glyph[0] < glyph[2] <= bounds[2]
                and bounds[1] <= glyph[1] < glyph[3] <= bounds[3]
            ):
                continue
            parent_overlap = dict(parent)
            parent_overlap["appearance"] = dict(parent_capsule, glyph_bounds=list(glyph))
            if same_text_appearance(parent_overlap, child):
                contained.add(index)
                break
    regions = [region for index, region in enumerate(regions) if index not in contained]
    unique = []
    for region in sorted(regions, key=lambda item: item["confidence"], reverse=True):
        x, y, width, height = region["box"]
        duplicate = False
        for other in unique:
            ox, oy, ow, oh = other["box"]
            intersection = max(0, min(x + width, ox + ow) - max(x, ox))
            intersection *= max(0, min(y + height, oy + oh) - max(y, oy))
            fraction = intersection / max(1, min(width * height, ow * oh))
            if fraction >= 0.75 and same_text_appearance(region, other):
                duplicate = True
                break
        if not duplicate:
            unique.append(region)
    rows = []
    for region in sorted(unique, key=lambda item: (item["box"][1] + item["box"][3] / 2, item["box"][0])):
        x, y, width, height = region["box"]
        if not region["text"]:
            continue
        center = y + height / 2
        matching = [
            row for row in rows
            if abs(row["center"] - center) <= 0.45 * min(row["height"], height)
            and any(
                max(0, max(x, other["box"][0]) - min(x + width, other["box"][0] + other["box"][2]))
                <= max(12, min(height, other["box"][3]) * 0.9)
                for other in row["regions"]
            )
        ]
        if matching:
            row = min(matching, key=lambda item: abs(item["center"] - center))
        else:
            row = {"center": center, "height": height, "regions": []}
            rows.append(row)
        row["regions"].append(region)
    result = []
    for row in sorted(rows, key=lambda item: item["center"]):
        blocks = []
        for region in sorted(row["regions"], key=lambda item: item["box"][0]):
            x, y, width, height = region["box"]
            previous = blocks[-1] if blocks else None
            if previous and x - previous["box"][0] - previous["box"][2] <= max(12, min(height, previous["box"][3]) * 0.9):
                old_x, old_y, old_w, old_h = previous["box"]
                previous["text"] += " " + region["text"]
                previous["box"] = (
                    old_x, min(old_y, y), max(old_x + old_w, x + width) - old_x,
                    max(old_y + old_h, y + height) - min(old_y, y),
                )
                previous["regions"].append(region)
            else:
                blocks.append({"text": region["text"], "box": region["box"], "regions": [region]})
        result.extend(blocks)
    return result


def _appearance_consensus(entries: list[tuple[dict, dict]]) -> None:
    """Resolve different OCR readings only when original glyph pixels agree."""
    leaders, groups = [], []
    for observation, region in entries:
        if not region["text"] or not region.get("appearance", {}).get("available"):
            continue
        nearby = sorted(leaders, key=lambda pair: abs(
            pair[0]["candidate"]["timestamp"] - observation["candidate"]["timestamp"],
        ))
        matching = next((other for _, other in nearby[:12] if same_text_appearance(region, other)), None)
        if matching is not None:
            group = groups[matching["appearance_class"]]
            group["members"].append(region)
            group["readings"].add(region["raw_text"])
            region["appearance_class"] = matching["appearance_class"]
            if region["text"] != matching["raw_text"]:
                region["text"] = matching["raw_text"]
                region["appearance_consensus"] = True
        else:
            # A fixed, first observed reading anchors each appearance class.
            # Corrected alternatives never become proof references, so neither
            # numeric substitutions nor chains of small changes can propagate.
            region["appearance_class"] = len(groups)
            leaders.append((observation, region))
            groups.append({"text": region["raw_text"], "members": [region], "readings": {region["raw_text"]}})
    alias_labels = {group["text"] for group in groups if len(group["readings"]) > 1}
    collisions = {text for text in alias_labels if sum(group["text"] == text for group in groups) > 1}
    for group in groups:
        if group["text"] in collisions:
            for region in group["members"]:
                region["appearance_identity"] = (region["track"], region["appearance_class"])


def _character_tokens(source: dict) -> list[tuple[str, list[float]]]:
    characters = source.get("recognition_characters", [])
    if not characters or source.get("recognition_details_match") is False:
        return []
    text = "".join(str(item.get("character", "")) for item in characters)
    if normalize_text(text) != normalize_text(source.get("text", "")):
        return []
    tokens, letters, confidences = [], [], []
    for item in characters:
        character = str(item.get("character", ""))
        if character.isspace():
            if letters:
                tokens.append((normalize_text("".join(letters)), confidences))
                letters, confidences = [], []
        else:
            letters.append(character)
            confidences.append(float(item.get("confidence", 0)))
    if letters:
        tokens.append((normalize_text("".join(letters)), confidences))
    return tokens


def _uncertain_edge_consensus(entries: list[tuple[dict, dict]]) -> None:
    readings = defaultdict(int)
    for _, region in entries:
        if region["confidence"] >= 0.9 and region["geometry_complete"]:
            readings[region["raw_text"]] += 1
    proposals = []
    digit_values = {token for _, region in entries for token, _ in region["character_tokens"]
                    if token.isdigit()}
    edge_readings = defaultdict(set)
    for _, region in entries:
        tokens = region["character_tokens"]
        if len(tokens) >= 2:
            edge_readings[" ".join(token for token, _ in tokens[1:])].add(tokens[0][0])
            edge_readings[" ".join(token for token, _ in tokens[:-1])].add(tokens[-1][0])
    for _, region in entries:
        tokens = region["character_tokens"]
        if len(tokens) < 2:
            continue
        for index in (0, len(tokens) - 1):
            token, probabilities = tokens[index]
            if len(token) > 1 or not probabilities or max(probabilities) >= 0.85:
                continue
            if any(character in "$€£¥%" for character in token):
                continue
            if token.isdigit() and len(digit_values) >= 2:
                continue
            remaining = tokens[1:] if index == 0 else tokens[:-1]
            if not all(probabilities and min(probabilities) >= 0.90 for _, probabilities in remaining):
                continue
            main = " ".join(token for token, _ in remaining)
            variants = edge_readings[main]
            graphic_confusion = any(value.isdigit() for value in variants) and any(value.isalpha() for value in variants)
            if readings[main] >= 2 and graphic_confusion:
                proposals.append((region, main))
                break
    repeated = defaultdict(int)
    for region, main in proposals:
        repeated[(region["raw_text"], main)] += 1
    for region, main in proposals:
        if repeated[(region["raw_text"], main)] >= 2:
            region["text"] = main
            region["uncertain_edge_ignored"] = True


def _assign_line_layouts(observations: list[dict], diagonal: float) -> None:
    """Keep independent fields in stable physical slots despite detector order."""
    active = {}
    next_id = 0
    for observation in observations:
        time = observation["candidate"]["timestamp"]
        active = {key: value for key, value in active.items() if time - value["time"] <= 1.0}
        used = set()
        for line in observation["lines"]:
            matches = []
            for key, previous in active.items():
                if key in used:
                    continue
                cost = _match_cost(line["box"], previous["box"], diagonal)
                if cost is not None:
                    matches.append((cost - (0.02 if line["text"] == previous["text"] else 0), key))
            if matches:
                layout_id = min(matches)[1]
            else:
                unchanged = [key for key, previous in active.items()
                             if key not in used and previous["text"] == line["text"]]
                layout_id = unchanged[0] if len(unchanged) == 1 else next_id
                if len(unchanged) != 1:
                    next_id += 1
            used.add(layout_id)
            line["layout_id"] = layout_id
            active[layout_id] = {"box": line["box"], "text": line["text"], "time": time}
        observation["signature"] = tuple(sorted(
            (line["layout_id"], line["text"], tuple(
                region["appearance_identity"] for region in line["regions"] if "appearance_identity" in region
            )) for line in observation["lines"]
        )) or None


def _snapshot_growth(run: dict, neighbor: dict | None, diagonal: float) -> int:
    if neighbor is None:
        return 0
    available = list(neighbor["lines"])
    changes = 0
    for line in run["lines"]:
        matching = []
        for position, other in enumerate(available):
            if line["text"] != other["text"] and not _fragment_of(line["text"], other["text"]):
                continue
            cost = _match_cost(line["box"], other["box"], diagonal)
            if cost is not None:
                matching.append((cost, position))
        if not matching:
            return 0
        other = available.pop(min(matching)[1])
        changes += line["text"] != other["text"]
    if not changes:
        return 0
    changes += len(available)
    same_scenes = ({item["scene"] for item in run["observations"]}
                   & {item["scene"] for item in neighbor["observations"]})
    return 0 if not same_scenes and changes < 2 else changes


def _prominent_phrase_fragment(run: dict, neighbor: dict, diagonal: float, height: float) -> bool:
    for line in run["lines"]:
        if line["box"][3] < max(16, height * 0.035):
            continue
        for other in neighbor["lines"]:
            if not _fragment_of(line["text"], other["text"]):
                continue
            if " " not in line["text"] + other["text"] and not line["text"].endswith("-"):
                continue
            if _match_cost(line["box"], other["box"], diagonal) is not None:
                return True
    return False


def _prefix_transition(first: dict, second: dict, diagonal: float) -> bool:
    return any(
        _fragment_of(line["text"], other["text"])
        and _match_cost(line["box"], other["box"], diagonal) is not None
        for line in first["lines"] for other in second["lines"]
    )


def _line_additions(first: dict, second: dict, diagonal: float) -> bool:
    available = list(second["lines"])
    for line in first["lines"]:
        matches = [
            (cost, index) for index, other in enumerate(available)
            if line["text"] == other["text"]
            and (cost := _match_cost(line["box"], other["box"], diagonal)) is not None
        ]
        if not matches:
            return False
        available.pop(min(matches)[1])
    return bool(available)


def _repair_line_seams(observations: list[dict], diagonal: float) -> None:
    """Align an overlapping detector seam with repeated full-line OCR evidence."""
    references = defaultdict(list)
    for observation in observations:
        for line in observation["lines"]:
            if any(len(region["text"].split()) >= 3 and region["confidence"] >= 0.95
                   and region["geometry_complete"] and not region["clipped"]
                   for region in line["regions"]):
                references[(observation["scene"], line["text"])].append((
                    observation["candidate"]["timestamp"], line,
                ))
    for observation in observations:
        for line in observation["lines"]:
            regions = line["regions"]
            if len(regions) < 2:
                continue
            tokens = line["text"].split()
            offset = 0
            for left, right in zip(regions, regions[1:]):
                left_tokens, right_tokens = left["text"].split(), right["text"].split()
                offset += len(left_tokens)
                if not left_tokens or not right_tokens or offset > len(tokens):
                    continue
                x, _, width, _ = left["box"]
                if right["box"][0] >= x + width:
                    continue
                last, next_word = left_tokens[-1], right_tokens[0]
                proposed = None
                if len(last) > 3 and last[-1:] == next_word[:1] and not any(char.isdigit() for char in last):
                    replacement = tokens.copy()
                    replacement[offset - 1] = last[:-1]
                    proposed = " ".join(replacement)
                elif len(last) == 1 and last == next_word[:1]:
                    proposed = " ".join(tokens[:offset - 1] + tokens[offset:])
                if proposed is None:
                    continue
                matches = [
                    other for time, other in references.get((observation["scene"], proposed), [])
                    if abs(time - observation["candidate"]["timestamp"]) <= 0.8
                    and _match_cost(line["box"], other["box"], diagonal) is not None
                ]
                if len(matches) >= 2:
                    line["text"] = proposed
                    observation["seam_recovered"] = True
                    break


def _run_fragment(run: dict, neighbor: dict | None) -> bool:
    if neighbor is None:
        return False
    if not ({item["scene"] for item in run["observations"]}
            & {item["scene"] for item in neighbor["observations"]}):
        return False
    current, other = run["tracks"], neighbor["tracks"]
    changes = []
    for track, text in current.items():
        if track not in other:
            return False
        if text != other[track]:
            changes.append((text, other[track]))
    return bool(changes) and all(_fragment_of(first, second) for first, second in changes)


def _rolling_transition(first: dict, second: dict) -> bool:
    if first["scene"] != second["scene"]:
        return False
    differences = [(text, second["tracks"].get(track, ""))
                   for track, text in first["tracks"].items()
                   if text != second["tracks"].get(track, "")]
    if len(differences) != 1:
        return False
    before, after = differences[0]
    if any(character.isdigit() for character in before + after):
        return False
    before, after = before.split(), after.split()
    if min(len(before), len(after)) < 2:
        return False
    return any(before[-count:] == after[:count] for count in range(1, min(len(before), len(after))))


def _frame_duration(candidate: dict, video: dict) -> float:
    index = candidate.get("frame_index", candidate.get("id", 0))
    frames = video.get("frames", [])
    if isinstance(index, int) and 0 <= index < len(frames):
        duration = frames[index].get("duration", 0)
        if isinstance(duration, (int, float)) and duration > 0:
            return float(duration)
    fps = video.get("fps", 24)
    return 1 / fps if isinstance(fps, (int, float)) and fps > 0 else 1 / 24


def assign_text_states(candidates: list[dict], scenes: list[dict], video: dict) -> list[dict]:
    """Annotate candidates in place and return chronological text occurrences.

    Input geometry must use video["width"/"height"] coordinates. Observations
    missing recognition fields are treated neutrally for older analyses.
    Unreliable decorative regions never override a reliable small caption.
    """
    ordered = sorted(candidates, key=lambda item: (item["timestamp"], item.get("frame_index", item["id"])))
    if not ordered:
        return []
    diagonal = math.hypot(video.get("width", 960), video.get("height", 540))
    observations = []
    tracks = {}
    next_track = 0
    for candidate in ordered:
        tracks = {
            track_id: previous for track_id, previous in tracks.items()
            if previous["scene"] == candidate["scene_id"]
            and candidate["timestamp"] - previous["time"] <= 1.0
        }
        regions = []
        used = set()
        for source in sorted(candidate.get("vision", {}).get("text", []), key=lambda region: (_box(region)[1], _box(region)[0])):
            text = normalize_text(source.get("text", ""))
            try:
                confidence = float(source.get("recognition_confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            if not math.isfinite(confidence):
                confidence = 0.0
            if "\ufffd" in text:
                confidence = 0.0
            box = _box(source)
            # Detection-only legacy records are not an invented OCR reading.
            if not text and "recognition_confidence" not in source:
                continue
            matches = []
            for track_id, previous in tracks.items():
                if track_id in used or previous["scene"] != candidate["scene_id"]:
                    continue
                if candidate["timestamp"] - previous["time"] > 1.0:
                    continue
                cost = _match_cost(box, previous["box"], diagonal)
                if cost is not None:
                    matches.append((cost, track_id))
            track_id = min(matches)[1] if matches else next_track
            if not matches:
                next_track += 1
            used.add(track_id)
            region = {
                "text": text, "raw_text": text, "confidence": max(0, min(1, confidence)),
                "box": box, "track": track_id, "clipped": _clipped(source, video),
                "geometry_complete": source.get("complete_geometry", True) is not False,
                "geometry_measured": "complete_geometry" in source,
                "appearance": source.get("appearance", {}),
                "complete_geometry": source.get("complete_geometry", True),
                "pattern_evidence": text_pattern_evidence(source),
                "character_tokens": _character_tokens(source),
                "contrast": source.get("contrast"), "recovered": False,
            }
            regions.append(region)
            tracks[track_id] = {"box": box, "time": candidate["timestamp"], "scene": candidate["scene_id"]}
        observations.append({"candidate": candidate, "regions": regions, "scene": candidate["scene_id"]})

    history = defaultdict(list)
    for observation in observations:
        for region in observation["regions"]:
            history[region["track"]].append((observation, region))
    for entries in history.values():
        _appearance_consensus(entries)
        _uncertain_edge_consensus(entries)

    # Repair only weak isolated alphabetic OCR flicker between agreeing readings.
    # Confident alternatives, price changes and consecutive alternatives stay
    # distinct, even when their edit distance is only one character.
    for entries in history.values():
        for position in range(1, len(entries) - 1):
            observation, region = entries[position]
            before_observation, before = entries[position - 1]
            after_observation, after = entries[position + 1]
            if region["confidence"] >= 0.70 or region["clipped"]:
                continue
            if before["text"] != after["text"] or min(before["confidence"], after["confidence"]) < 0.80:
                continue
            if after_observation["candidate"]["timestamp"] - before_observation["candidate"]["timestamp"] > 0.8:
                continue
            if not region["text"] or region["text"] == before["text"] or _similar_jitter(region["text"], before["text"]):
                region["text"] = before["text"]
                region["recovered"] = True

    reliable_tracks = set()
    for track, entries in history.items():
        readings = defaultdict(int)
        for _, region in entries:
            if region["text"] and region["confidence"] >= MIN_CONFIDENCE and not region["recovered"]:
                readings[region["text"]] += 1
        if max(readings.values(), default=0) >= 2 or any(
            region["text"] and region["confidence"] >= 0.85 for _, region in entries
        ):
            reliable_tracks.add(track)
        elif len(entries) == 1 and entries[0][1]["confidence"] >= 0.9:
            scene = next((scene for scene in scenes if scene["id"] == entries[0][0]["scene"]), {})
            if scene.get("start_frame") == scene.get("end_frame") and "start_frame" in scene:
                reliable_tracks.add(track)
    ignored_auxiliary = set()
    area = max(1, video.get("width", 960) * video.get("height", 540))
    for track in reliable_tracks:
        entries = history[track]
        pattern_flags = sum(region["pattern_evidence"].get("unlikely_repeated_glyphs", False) for _, region in entries)
        nonpattern_readings = [
            region for _, region in entries
            if region["text"] and region["confidence"] >= MIN_CONFIDENCE
            and not region["pattern_evidence"].get("unlikely_repeated_glyphs", False)
            and (not region["pattern_evidence"].get("available")
                 or region["pattern_evidence"].get("periodicity", 1) >= 0.35)
        ]
        if pattern_flags >= 2 and not nonpattern_readings:
            ignored_auxiliary.add(track)
            continue
        sizes = [region["box"][2] * region["box"][3] / area for _, region in entries]
        readings = {region["text"] for _, region in entries if region["text"]}
        numeric = {text for text in readings if text.isdigit()}
        numeric_stems = {re.sub(r"\d+(?:[.,]\d+)*", "#", text) for text in readings}
        numeric_change = (len(numeric) >= 2 or
                          bool(readings) and all(any(char.isdigit() for char in text) for text in readings)
                          and len(numeric_stems) == 1)
        repeated_readings = defaultdict(int)
        for _, region in entries:
            if region["confidence"] >= MIN_CONFIDENCE and region["geometry_complete"]:
                repeated_readings[region["text"]] += 1
        confirmed_changes = sum(count >= 2 for count in repeated_readings.values()) >= 2
        repeated_strong = defaultdict(int)
        for _, region in entries:
            if region["confidence"] >= 0.90:
                repeated_strong[region["text"]] += 1
        strong_changes = sum(count >= 2 for count in repeated_strong.values()) >= 2
        large_neighbors = sum(
            any(other["track"] != track and other["box"][2] * other["box"][3] / area > 0.005
                and other["confidence"] >= 0.9 for other in observation["regions"])
            for observation, _ in entries
        )
        # Isolated superscripts and stray glyphs are not allowed to make an
        # otherwise complete headline unreadable. Multiple numeric values
        # remain meaningful, even when the price/caption is very small.
        short_auxiliary = (
            max(sizes, default=1) < 0.0025 and max(map(len, readings), default=100) <= 2
            and len(numeric) <= 1 and large_neighbors >= len(entries) * 0.75
            and not strong_changes
            and not any(character in "$€£¥%" for text in readings for character in text)
        )
        unstable_microtext = (
            max(sizes, default=1) < 0.03 and len(readings) >= 3
            and sum(region["confidence"] for _, region in entries) / len(entries) < 0.70
            and large_neighbors >= len(entries) * 0.75
        )
        dense_microtext = (
            max((region["box"][3] for _, region in entries), default=100) <= 12
            and len(readings) >= 3 and not numeric_change and not confirmed_changes
            and sum(region["confidence"] for _, region in entries) / len(entries) < 0.90
            and large_neighbors >= len(entries) * 0.75
        )
        modern_neighbors = sum(
            any(other.get("refined_native", False) for other in observation["candidate"].get("vision", {}).get("text", []))
            for observation, _ in entries
        )
        uncertified_fallback = (
            modern_neighbors >= len(entries) * 0.75
            and sum(region["geometry_complete"] and region["geometry_measured"]
                    for _, region in entries) < len(entries) * 0.5
            and sum(region["confidence"] for _, region in entries) / len(entries) < 0.90
            and len(readings) <= 1 and not numeric_change
            and not any(character in "$€£¥%" for text in readings for character in text)
        )
        if short_auxiliary or unstable_microtext or dense_microtext or uncertified_fallback:
            ignored_auxiliary.add(track)
    reliable_tracks -= ignored_auxiliary
    for observation in observations:
        relevant = [region for region in observation["regions"] if region["track"] in reliable_tracks]
        observation["ignored_weak"] = len(observation["regions"]) - len(relevant)
        observation["regions"] = relevant
        lines = _canonical_lines(relevant)
        observation["lines"] = lines
        contents = list(dict.fromkeys(line["text"] for line in lines))
        observation["contents"] = contents
        # Layout-independent region identities avoid splitting one continuous
        # message merely because its intact words move across the image.
        observation["signature"] = tuple(contents) if contents else None
    _repair_line_seams(observations, diagonal)
    for observation in observations:
        observation["contents"] = list(dict.fromkeys(line["text"] for line in observation["lines"]))
        observation["signature"] = tuple(observation["contents"]) or None
    _assign_line_layouts(observations, diagonal)

    runs = []
    for observation_index, observation in enumerate(observations):
        observation["index"] = observation_index
        signature = observation["signature"]
        if signature is None:
            continue
        candidate = observation["candidate"]
        preceding = runs[-1] if runs else None
        same = preceding is not None and preceding["signature"] == signature
        # Brief detector misses can bridge a run. A sustained empty interval is
        # a disappearance/reappearance, which is a new occurrence.
        if same:
            last = preceding["observations"][-1]["candidate"]
            between = observations[preceding["observations"][-1]["index"] + 1:observation_index]
            empty_span = (candidate["timestamp"] - between[0]["candidate"]["timestamp"]) if between else 0
            same = empty_span < BRIEF_FRAGMENT_SECONDS
        if not same:
            runs.append({
                "signature": signature, "observations": [], "scene": observation["scene"],
                "tracks": {region["track"]: region["text"] for region in observation["regions"] if region["text"]},
                "lines": observation["lines"],
            })
        runs[-1]["observations"].append(observation)

    scene_lookup = {scene["id"]: scene for scene in scenes}
    run_for_observation = {}
    accepted = []
    rolling = [_rolling_transition(first, second) for first, second in zip(runs, runs[1:])]
    additions = [_line_additions(first, second, diagonal) for first, second in zip(runs, runs[1:])]
    prefixes = [_prefix_transition(first, second, diagonal) for first, second in zip(runs, runs[1:])]
    run_times = [run["observations"][0]["candidate"]["timestamp"] for run in runs]
    confirmed = [
        sum(all(region["text"] and region["confidence"] >= MIN_CONFIDENCE
                and not region["clipped"] and region["geometry_complete"] and not region["recovered"]
                for region in observation["regions"]) for observation in run["observations"]) >= 2
        for run in runs
    ]
    for position, run in enumerate(runs):
        items = run["observations"]
        first, last = items[0]["candidate"], items[-1]["candidate"]
        next_time = (runs[position + 1]["observations"][0]["candidate"]["timestamp"]
                     if position + 1 < len(runs)
                     else scene_lookup.get(items[-1]["scene"], {}).get("end", last["timestamp"] + _frame_duration(last, video)))
        duration = max(0, float(next_time) - first["timestamp"])
        before = runs[position - 1] if position else None
        after = runs[position + 1] if position + 1 < len(runs) else None
        growth = max(_snapshot_growth(run, before, diagonal), _snapshot_growth(run, after, diagonal))
        prefix_signal = _run_fragment(run, before) or _run_fragment(run, after) or growth > 0
        phrase_context = any(
            _fragment_of(line["text"], other["text"])
            and (" " in line["text"] + other["text"] or line["text"].endswith("-"))
            for neighbor in (before, after) if neighbor
            for line in run["lines"] for other in neighbor["lines"]
        )
        prefix_chain = (
            (position + 1 < len(prefixes) and prefixes[position] and prefixes[position + 1])
            or (position > 0 and position < len(prefixes) and prefixes[position - 1] and prefixes[position])
        )
        brief_fragment = duration < BRIEF_FRAGMENT_SECONDS and prefix_signal and (phrase_context or prefix_chain)
        # Several labels appearing together are one entering layout, rather than
        # multiple complete headlines. A held SALE -> SALE NOW still has only
        # one changed line and retains its distinct wording occurrences.
        brief_fragment |= duration < 0.60 and growth >= 2
        dangling = any(line["text"].endswith("-") for line in run["lines"])
        brief_fragment |= dangling and duration < 0.60 and growth > 0
        entering_paragraph = (
            (position + 1 < len(additions) and additions[position] and additions[position + 1])
            or (position > 0 and position < len(additions) and additions[position - 1] and additions[position])
        )
        before_window = range(bisect_left(run_times, first["timestamp"] - 1.2), position)
        after_window = range(position + 1, bisect_right(run_times, first["timestamp"] + 1.2))
        future_additions = [
            runs[index] for index in after_window
            if confirmed[index] and _line_additions(run, runs[index], diagonal)
        ]
        previous_additions = any(
            confirmed[index] and _line_additions(runs[index], run, diagonal)
            for index in before_window
        )
        entering_paragraph |= any(
            len(other["lines"]) >= len(run["lines"]) + 2 for other in future_additions
        ) or (bool(future_additions) and previous_additions)
        brief_fragment |= duration < 0.60 and entering_paragraph
        fragment_window = range(bisect_left(run_times, first["timestamp"] - 0.5),
                                bisect_right(run_times, first["timestamp"] + 0.5))
        brief_fragment |= duration < BRIEF_FRAGMENT_SECONDS and any(
            index != position and _prominent_phrase_fragment(run, runs[index], diagonal, video.get("height", 540))
            for index in fragment_window
        )
        rolling_chain = (
            (position > 0 and position < len(rolling) and rolling[position - 1] and rolling[position])
            or (position + 1 < len(rolling) and rolling[position] and rolling[position + 1])
            or (position >= 2 and rolling[position - 2] and rolling[position - 1])
        )
        brief_fragment |= duration < BRIEF_FRAGMENT_SECONDS and rolling_chain
        span = last["timestamp"] - first["timestamp"] + _frame_duration(last, video)
        confirmations = sum(
            all(region["text"] and region["confidence"] >= MIN_CONFIDENCE
                and not region["clipped"] and region["geometry_complete"] and not region["recovered"]
                for region in item["regions"])
            for item in items
        )
        scene = scene_lookup.get(run["scene"], {})
        single_frame_scene = scene.get("start_frame") == scene.get("end_frame") and "start_frame" in scene
        supported = (confirmations >= 2 and span >= MIN_CONFIRMED_SPAN) or (
            single_frame_scene and confirmations == 1
            and all(region["confidence"] >= 0.90 for region in items[0]["regions"])
        )
        run.update(fragment=brief_fragment, supported=supported)
        peaks = {}
        for item in items:
            for region in item["regions"]:
                if isinstance(region["contrast"], (int, float)):
                    peaks[region["track"]] = max(peaks.get(region["track"], 0), float(region["contrast"]))
        run["contrast_peaks"] = peaks
        for item in items:
            run_for_observation[id(item)] = run
        if not brief_fragment and supported:
            previous = accepted[-1] if accepted else None
            appearance_classes = tuple(sorted(
                region["appearance_identity"] for region in items[0]["regions"]
                if "appearance_identity" in region
            ))
            bridge = (
                previous is not None and previous["text"] == items[0]["contents"]
                and first["timestamp"] - previous["end"] <= 0.4
                and previous["_scene"] == run["scene"]
                and previous["_appearance"] == appearance_classes
            )
            if bridge:
                run["state_id"] = previous["id"]
                previous["candidate_ids"].extend(item["candidate"]["id"] for item in items)
                previous["end"] = last["timestamp"] + _frame_duration(last, video)
            else:
                run["state_id"] = len(accepted)
                accepted.append({
                    "id": run["state_id"], "text": list(items[0]["contents"]),
                    "candidate_ids": [item["candidate"]["id"] for item in items],
                    "start": first["timestamp"], "end": last["timestamp"] + _frame_duration(last, video),
                    "_scene": run["scene"],
                    "_appearance": appearance_classes,
                })

    for observation in observations:
        candidate, regions = observation["candidate"], observation["regions"]
        candidate["text_state_id"] = None
        candidate["text_contents"] = observation["contents"]
        candidate["text_ready"] = True
        candidate["text_complete"] = 1.0
        candidate["text_reason"] = "No reliably recognized text"
        if not observation["signature"]:
            if regions:
                # A track with a later reliable reading is real text evidence,
                # even when this particular early/flickering crop reads empty.
                candidate["text_ready"] = False
                candidate["text_complete"] = 0.0
                candidate["text_reason"] = "Detected text has no reliable complete reading in this frame"
            if observation["ignored_weak"]:
                candidate["text_reason"] += "; unreliable regions were not certified"
            candidate.setdefault("metrics", {})["text_completeness"] = candidate["text_complete"]
            continue
        run = run_for_observation[id(observation)]
        candidate["text_state_id"] = run.get("state_id")
        confidence = min(region["confidence"] for region in regions)
        clipped = any(region["clipped"] for region in regions)
        geometry_incomplete = any(not region["geometry_complete"] for region in regions)
        uncertain = any(not region["text"] or region["confidence"] < MIN_CONFIDENCE or region["recovered"]
                        for region in regions)
        uncertain |= observation.get("seam_recovered", False)
        contrasts = [float(region["contrast"]) for region in regions if isinstance(region["contrast"], (int, float))]
        faint = bool(contrasts and min(contrasts) < MIN_CONTRAST)
        # Contrast changing substantially within one wording occurrence is a
        # fade estimate, not a claim about original alpha/opacity.
        fading = False
        for region in regions:
            if not isinstance(region["contrast"], (int, float)):
                continue
            peak = run["contrast_peaks"].get(region["track"], float(region["contrast"]))
            fading |= float(region["contrast"]) < peak * 0.65
        ready = bool(run.get("state_id") is not None and not clipped and not geometry_incomplete
                     and not uncertain and not faint and not fading)
        candidate["text_ready"] = ready
        complete = confidence
        if clipped or geometry_incomplete or run["fragment"]:
            complete = 0.0
        elif not run["supported"]:
            complete *= 0.5
        elif faint or fading:
            complete *= 0.4
        candidate["text_complete"] = round(complete, 3)
        candidate.setdefault("metrics", {})["text_completeness"] = candidate["text_complete"]
        if run["fragment"]:
            reason = "Brief growing or scrolling text fragment"
        elif clipped:
            reason = "Observed text intersects the viewport edge"
        elif geometry_incomplete:
            reason = "Detected word crop may omit letters"
        elif uncertain:
            reason = "Text reading is uncertain; neighboring consensus does not prove this frame is legible"
        elif not run["supported"]:
            reason = "Insufficient repeated wording to confirm a complete text moment"
        elif faint or fading:
            reason = "Text contrast has not reached a readable plateau"
        else:
            reason = "Repeated readable observed wording; no detected clipping"
            if candidate.get("metrics", {}).get("text_stability", 1.0) < 0.7:
                reason += "; prefer a settled layout when available"
        candidate["text_reason"] = reason
        if observation["ignored_weak"]:
            candidate["text_reason"] += "; unreliable regions were not certified"
        if any(region.get("appearance_consensus") for region in regions):
            candidate["text_reason"] += "; OCR variants aligned by unchanged native glyph pixels"
        if any(region.get("uncertain_edge_ignored") for region in regions):
            candidate["text_reason"] += "; uncertain isolated graphic token was not certified as wording"
    for state in accepted:
        state.pop("_scene", None)
        state.pop("_appearance", None)
    return accepted
