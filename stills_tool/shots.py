"""Streaming edit-boundary detection, independent of still selection and OCR.

Adjacent-frame colour changes propose edits; spatial coverage and registered
image continuity distinguish a new camera view from pans or a changing caption.
Only a short, small-frame history is kept. Boundaries use native frame numbers.
"""
from __future__ import annotations

from collections import deque
import math

import cv2
import numpy as np


def _small(frame, edge=320):
    height, width = frame.shape[:2]
    scale = min(1.0, edge / max(height, width))
    image = cv2.resize(frame, (max(8, round(width * scale)), max(8, round(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
    return cv2.GaussianBlur(image, (3, 3), .8)


def _coverage(difference, threshold=12.0):
    """Equal-area cells prevent a bright subtitle from dominating a frame mean."""
    cells = cv2.resize(difference.astype(np.float32), (8, 6), interpolation=cv2.INTER_AREA)
    return float(np.mean(cells > threshold))


def _spatial(difference):
    cells = cv2.resize(difference.astype(np.float32), (8, 6), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(cells > 12)
    return {"local_strength": float(np.sort(cells.ravel())[-max(4, len(xs)):].mean()),
            "span_x": float((xs.max() - xs.min() + 1) / 8) if len(xs) else 0,
            "span_y": float((ys.max() - ys.min() + 1) / 6) if len(ys) else 0}


def _histogram(image):
    # Joint coarse BGR is deliberately hue-wrap safe and works on gray footage.
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return hist.ravel() / max(float(hist.sum()), 1.0)


def _difference(previous, current, *, register=False):
    first = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    raw = np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean(axis=2)
    corrected = raw
    continuity = 0.0
    registered = False
    if register and float(first.std()) > 4 and float(second.std()) > 4:
        corners = cv2.goodFeaturesToTrack(first, 120, .012, 7, blockSize=5)
        if corners is not None and len(corners) >= 8:
            moved, status, error = cv2.calcOpticalFlowPyrLK(
                first, second, corners, None, winSize=(21, 21), maxLevel=3)
            if moved is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    second, first, moved, None, winSize=(21, 21), maxLevel=3)
                good = ((status.ravel() != 0) & (back_status.ravel() != 0)
                        & (error.ravel() < 22)
                        & (np.linalg.norm(corners[:, 0] - back[:, 0], axis=1) < 1.5))
                if int(good.sum()) >= 8:
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        corners[good], moved[good], method=cv2.RANSAC,
                        ransacReprojThreshold=2.0)
                    if matrix is not None:
                        scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
                        if .75 < scale < 1.35:
                            warped = cv2.warpAffine(previous, matrix, (second.shape[1], second.shape[0]),
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT)
                            valid = cv2.warpAffine(np.ones(first.shape, np.uint8), matrix,
                                                  (second.shape[1], second.shape[0])) > 0
                            if float(valid.mean()) > .55:
                                aligned = np.abs(current.astype(np.float32) - warped.astype(np.float32)).mean(axis=2)
                                aligned[~valid] = 0
                                corrected = aligned
                                continuity = float(good.sum() / len(corners) * inliers.mean())
                                registered = True
    hist = float(cv2.compareHist(_histogram(previous), _histogram(current), cv2.HISTCMP_BHATTACHARYYA))
    raw_spatial = _spatial(raw)
    cells = cv2.resize(raw, (8, 6), interpolation=cv2.INTER_AREA)
    return {"mean": float(raw.mean()), "coverage": _coverage(raw), **_spatial(corrected),
            "raw_span_x": raw_spatial["span_x"], "raw_span_y": raw_spatial["span_y"],
            "active_cells": np.flatnonzero(cells.ravel() > 12).tolist(),
            "residual": float(corrected.mean()), "residual_coverage": _coverage(corrected),
            "histogram": hist, "continuity": continuity, "registered": registered}


class ShotDetector:
    """Find hard cuts and gradual edits without a fixed minimum shot duration.

    Feed sequential frames with :meth:`process_frame`, then call :meth:`finish`.
    Finalisation removes brief exposure flashes and consolidates each dissolve
    into one boundary. ``diagnostics`` explains every accepted/rejected proposal.
    Detection uses <=320-pixel images and retains <=0.85 s of image history.
    """

    def __init__(self, fps=24.0):
        self.fps = max(float(fps), 1.0)
        self._history = deque(maxlen=max(8, round(self.fps * .85) + 2))
        self._metrics = []
        self._hard = []
        self._gradual = []
        self._flashes = set()
        self._appearance = []
        self.diagnostics = []
        self._finished = None

    def process_frame(self, index, frame):
        if self._finished is not None:
            raise RuntimeError("Cannot add frames after shot detection is finalised.")
        if self._history and index != self._history[-1][0] + 1:
            raise ValueError("ShotDetector requires sequential frame numbers.")
        image = _small(frame)
        histogram = _histogram(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        self._appearance.append({"frame": int(index), "mean": float(image.mean()), "std": float(image.std()),
                                 "palette": float(np.sort(histogram)[-3:].sum()),
                                 "texture": float(np.abs(laplacian).mean()),
                                 "focus_cells": cv2.resize(laplacian ** 2, (8, 6), interpolation=cv2.INTER_AREA).ravel().tolist()})
        if self._history:
            previous = self._history[-1][1]
            initial = _difference(previous, image)
            metric = _difference(previous, image, register=True) if (
                initial["coverage"] >= .20 and initial["mean"] >= 9) else initial
            metric["frame"] = int(index)
            self._metrics.append(metric)
            # New camera views change many image cells and cannot be explained
            # by a coherent camera transform. Fine caption motion is local.
            broad = metric["coverage"] >= .35
            residual = metric["residual"]
            covered = metric["residual_coverage"]
            motion = metric["continuity"] >= .30 and residual < metric["mean"] * .5 and covered < .25
            full_cut = (broad and not motion and (
                residual >= 22 and covered >= .35
                or residual >= 15 and covered >= .40 and metric["histogram"] >= .36))
            inset_cut = (not motion and covered >= .125 and metric["span_x"] >= .25
                         and metric["span_y"] >= 2 / 3 and metric["local_strength"] >= 27
                         and residual >= 5.5 and metric["histogram"] >= .065)
            proposed = full_cut or inset_cut
            if proposed:
                self._hard.append({**metric, "kind": "cut", "full_cut": full_cut,
                                   "confidence": min(1.0, residual / 45)})
            self._check_flash(index, image)
            self._check_gradual(index, image)
        self._history.append((int(index), image))
        return []

    def _check_flash(self, index, image):
        # A genuine short insert remains a shot. Only an almost untextured
        # black/white exposure excursion returning to the same view is removed.
        for span in range(1, min(3, len(self._history) - 1) + 1):
            before = self._history[-span - 1][1]
            middle = [item[1] for item in list(self._history)[-span:]]
            extreme = all((float(x.mean()) < 18 or float(x.mean()) > 237)
                          and float(x.std()) < 16 for x in middle)
            if extreme:
                returned = _difference(before, image, register=True)
                if returned["residual"] < 8 and returned["histogram"] < .20:
                    self._flashes.update(range(index - span, index + 1))

    def _check_gradual(self, index, image):
        for seconds in (.25, .5, .75):
            lag = max(4, round(self.fps * seconds))
            if len(self._history) < lag:
                continue
            sequence = [item[1] for item in list(self._history)[-lag:]] + [image]
            # Linearity of the image trajectory is high for a dissolve/fade,
            # low for moving subjects and pans. Small vectors bound CPU cost.
            vectors = [cv2.resize(x, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
                       for x in sequence]
            steps = [float(np.linalg.norm(b - a)) for a, b in zip(vectors, vectors[1:])]
            path = sum(steps)
            distance = float(np.linalg.norm(vectors[-1] - vectors[0]))
            linearity = distance / max(path, 1e-6)
            if linearity < .82 or max(steps, default=0) / max(path, 1e-6) > .45:
                continue
            change = _difference(sequence[0], image, register=True)
            if change["residual"] < 22 or change["residual_coverage"] < .40:
                continue
            # Ignore static padding so a fade's midpoint does not drift toward
            # whichever large comparison window happened to observe it.
            active = [i for i, step in enumerate(steps) if step > max(15, max(steps) * .18)]
            if len(active) < 3:
                continue
            start, end = index - lag + active[0], index - lag + active[-1] + 1
            self._gradual.append({"frame": round((start + end) / 2), "start": start, "end": end,
                                  "kind": "dissolve", "confidence": linearity,
                                  "strength": change["residual"]})

    def finish(self, frame_count=None):
        if self._finished is not None:
            return list(self._finished)
        count = frame_count if frame_count is not None else (self._history[-1][0] + 1 if self._history else 0)
        accepted = []
        for proposal in self._hard:
            item = dict(proposal)
            item["accepted"] = item["frame"] not in self._flashes
            if not item["full_cut"]:
                frame = item["frame"]
                neighbors = [x["mean"] for x in self._metrics[max(0, frame - 4):frame + 2]
                             if x["frame"] != frame]
                baseline = float(np.median(neighbors)) if neighbors else 0
                if item["mean"] < max(2.5, baseline * 3.0):
                    item["accepted"] = False
                    item["reason"] = "smooth partial-frame subject or camera motion"
                cells = item["active_cells"]
                if cells and item["coverage"] < .45 and item["histogram"] < .20:
                    local = lambda i: float(np.mean(np.asarray(self._appearance[i]["focus_cells"])[cells]))
                    earlier = [local(i) for i in range(max(0, frame - 4), frame)]
                    later = [local(i) for i in range(frame + 1, min(len(self._appearance), frame + 4))]
                    reference = max(earlier, default=0)
                    if (reference > 4 and local(frame) < reference * .78
                            and min([local(frame - 1), *later], default=reference) < reference * .78
                            and max(later, default=0) >= reference * .80):
                        item["accepted"] = False
                        item["reason"] = "brief local motion-blur valley returning to the same detail level"
            if not item["accepted"]:
                item.setdefault("reason", "brief exposure flash returning to the same view")
            else:
                accepted.append(item)
            self.diagnostics.append(item)
        groups = []
        for item in sorted(self._gradual, key=lambda x: x["start"]):
            if groups and item["start"] <= groups[-1][-1]["end"] + 2:
                groups[-1].append(item)
            else:
                groups.append([item])
        for group in groups:
            item = dict(max(group, key=lambda x: x["strength"]))
            item["start"] = min(x["start"] for x in group)
            item["end"] = max(x["end"] for x in group)
            span = self._appearance[item["start"]:item["end"] + 1]
            dark = [x for x in span if x["mean"] < 18 and x["std"] < 16]
            if dark:
                item["frame"] = round((dark[0]["frame"] + dark[-1]["frame"]) / 2)
                item["kind"] = "fade"
            # A fast dissolve may also trigger adjacent-frame proposals. It is
            # one edit, not a series of tiny artificial shots.
            nearby = [x for x in accepted if item["start"] <= x["frame"] <= item["end"]]
            if nearby:
                for x in nearby:
                    accepted.remove(x)
                    x["accepted"] = False
                    x["reason"] = "part of a gradual transition"
            item["accepted"] = True
            accepted.append(item)
            self.diagnostics.append(item)
        self._add_composition_transitions(accepted)
        self._merge_wipes(accepted)
        # Black/white leader and tail contain no additional photographed shot.
        for item in list(accepted):
            before, after = self._appearance[:item["frame"]], self._appearance[item["frame"]:]
            empty = lambda rows: bool(rows) and all(
                (x["mean"] < 18 or x["mean"] > 237) and x["std"] < 3 for x in rows)
            if empty(before) or empty(after):
                accepted.remove(item)
                item["accepted"] = False
                item["reason"] = "empty leader or tail"
            elif item["frame"] < self.fps * .2 and before and all(x["texture"] < 1.0 for x in before):
                accepted.remove(item)
                item["accepted"] = False
                item["reason"] = "untextured entrance animation before the first shot"
        # No .4 s refractory period: short editorial inserts deserve a still.
        self._finished = sorted({int(x["frame"]) for x in accepted if 0 < x["frame"] < count})
        return list(self._finished)

    def _add_composition_transitions(self, accepted):
        """A short authored scroll/wipe inside a static canvas ends a view.

        This catches a phone mockup moving from one screen to the next. A
        continuous full-frame camera pan and changing caption do not qualify.
        """
        runs = []
        for metric in self._metrics:
            active = (metric["mean"] >= 4.0 and metric["coverage"] >= .15
                      and metric["raw_span_y"] >= 1 / 3 and metric["raw_span_x"] >= .25)
            if active:
                if runs and metric["frame"] <= runs[-1][-1]["frame"] + 3:
                    runs[-1].append(metric)
                else:
                    runs.append([metric])
        for run in runs:
            start, end = run[0]["frame"], run[-1]["frame"]
            duration = (end - start + 1) / self.fps
            if not (.12 <= duration <= 1.6) or start < self.fps * .5:
                continue
            if np.median([x["coverage"] for x in run]) >= .85:
                continue
            if max(x["raw_span_y"] for x in run) < 2 / 3:
                continue
            if (np.median([x["continuity"] for x in run]) > .90
                    and np.median([x["residual"] / max(x["mean"], .01) for x in run]) < .35):
                # A smoothly registered zoom retains the same photographed
                # view, even when it begins/ends inside a graphic frame.
                continue
            if sum(x["mean"] for x in run) < 40 or max(x["mean"] for x in run) < 12:
                continue
            # Require calm imagery either side: no segmentation of a moving
            # speaker, an ongoing pan, or continuous camera action.
            before = self._metrics[max(0, start - 6):max(0, start - 1)]
            after = self._metrics[end:min(len(self._metrics), end + 5)]
            if len(before) < 3 or len(after) < 3:
                continue
            baseline = max(np.median([x["mean"] for x in before]), np.median([x["mean"] for x in after]))
            if baseline > 6 or max(x["mean"] for x in run) < baseline * 2.5:
                continue
            nearby = [x for x in accepted if start - 2 <= x["frame"] <= end + 2]
            if any(x["residual"] >= 22 and x["coverage"] >= .7 for x in nearby if "residual" in x):
                continue
            for item in nearby:
                accepted.remove(item)
                item["accepted"] = False
                item["reason"] = "part of an authored composition transition"
            item = {"frame": round((start + end) / 2), "start": start, "end": end,
                    "kind": "composition", "confidence": .7, "accepted": True}
            accepted.append(item)
            self.diagnostics.append(item)

    def _merge_wipes(self, accepted):
        groups = []
        for item in sorted(accepted, key=lambda x: x["frame"]):
            if groups and item["frame"] - groups[-1][-1]["frame"] <= max(2, round(self.fps * .14)):
                groups[-1].append(item)
            else:
                groups.append([item])
        for group in groups:
            if len(group) < 2:
                continue
            start, end = group[0]["frame"], group[-1]["frame"]
            middle = self._appearance[start:end + 1]
            endpoints = [self._appearance[max(0, start - 1)], self._appearance[end]]
            # Flat wipe graphics briefly replace photographed content. Dense
            # rapid edits between real images (including one-frame inserts)
            # must remain independent shots.
            flat_wipe = (len(group) >= 3 and max(x["palette"] for x in middle) >= .85
                         and max(x["palette"] for x in middle) >= min(x["palette"] for x in endpoints) + .06)
            surrounding = self._metrics[max(0, start - 3):min(len(self._metrics), end + 2)]
            moving_wipe = (len(group) == 2 and all(x.get("residual", 100) < 22 for x in group)
                           and len(surrounding) >= 5 and min(x["mean"] for x in surrounding) >= 5)
            if not (flat_wipe or moving_wipe):
                continue
            # A real two-frame insert has a settled middle frame. Wipes have
            # sustained image change throughout their interior.
            inside = self._metrics[start:end - 1]
            if inside and min(x["mean"] for x in inside) < 3:
                continue
            for item in group:
                accepted.remove(item)
                item["accepted"] = False
                item["reason"] = "part of an animated wipe"
            item = {"frame": round((start + end) / 2), "start": start, "end": end,
                    "kind": "wipe", "confidence": .8, "accepted": True}
            accepted.append(item)
            self.diagnostics.append(item)
