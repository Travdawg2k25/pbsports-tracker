"""ByteTrack-style multi-object tracking.

Implemented here rather than pulled from the detector library so the ball and the players
can use different association rules: players associate on IoU with a constant-velocity
prediction, the ball associates on centre distance because it moves further between
frames than its own width.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackConfig
from .types import Box, Detection, Track, TrackFrame


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


class _Live:
    __slots__ = ("track_id", "box", "velocity", "age", "hits", "frames", "last_frame")

    def __init__(self, track_id: int, box: Box, frame: int, conf: float):
        self.track_id = track_id
        self.box = np.array(box, dtype=float)
        self.velocity = np.zeros(4)
        self.age = 0
        self.hits = 1
        self.last_frame = frame
        self.frames: list[TrackFrame] = [TrackFrame(frame, box, conf)]

    def predict(self) -> np.ndarray:
        return self.box + self.velocity

    def update(self, box: Box, frame: int, conf: float) -> None:
        new = np.array(box, dtype=float)
        # Light velocity smoothing; a full Kalman filter buys little for a fixed camera.
        self.velocity = 0.5 * self.velocity + 0.5 * (new - self.box)
        self.box = new
        self.age = 0
        self.hits += 1
        self.last_frame = frame
        self.frames.append(TrackFrame(frame, box, conf))


class ByteTracker:
    """Two-stage association: confident detections first, then leftovers."""

    def __init__(self, cfg: TrackConfig, cls: str = "player"):
        self.cfg = cfg
        self.cls = cls
        self.live: list[_Live] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame: int) -> None:
        high = [d for d in detections if d.conf >= self.cfg.high_thresh]
        low = [d for d in detections if self.cfg.low_thresh <= d.conf < self.cfg.high_thresh]

        unmatched = self._associate(high, frame, self.cfg.match_thresh)
        self._associate(low, frame, self.cfg.second_match_thresh, create=False)

        for det in unmatched:
            self.live.append(_Live(self._next_id, det.box, frame, det.conf))
            self._next_id += 1

        for t in list(self.live):
            if t.last_frame != frame:
                t.age += 1
            if t.age > self.cfg.max_age:
                self._retire(t)

    def _associate(
        self, dets: list[Detection], frame: int, max_cost: float, create: bool = True
    ) -> list[Detection]:
        """Hungarian match on IoU distance (1 - IoU); ``max_cost`` is the distance limit."""
        pending = [t for t in self.live if t.last_frame != frame]
        if not dets or not pending:
            return list(dets) if create else []
        pred = np.stack([t.predict() for t in pending])
        boxes = np.array([d.box for d in dets], dtype=float)
        cost = 1.0 - iou_matrix(pred, boxes)
        rows, cols = linear_sum_assignment(cost)
        matched_dets: set[int] = set()
        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] <= max_cost:
                pending[r].update(dets[c].box, frame, dets[c].conf)
                matched_dets.add(c)
        return [d for i, d in enumerate(dets) if i not in matched_dets] if create else []

    def _retire(self, t: _Live) -> None:
        self.live.remove(t)
        if t.hits >= self.cfg.min_hits and len(t.frames) >= self.cfg.min_track_len:
            self.finished.append(Track(track_id=t.track_id, cls=self.cls, frames=t.frames))

    def tracks(self) -> list[Track]:
        for t in list(self.live):
            self._retire(t)
        return sorted(self.finished, key=lambda tr: tr.start)


class BallTracker:
    """Single-target tracker: nearest detection to a linear prediction, with coasting."""

    def __init__(self, max_gap: int = 12, max_jump_px: float = 260.0):
        self.max_gap = max_gap
        self.max_jump_px = max_jump_px
        self.frames: list[TrackFrame] = []
        self._last: np.ndarray | None = None
        self._vel = np.zeros(2)
        self._gap = 0

    def update(self, detections: list[Detection], frame: int) -> None:
        cands = [d for d in detections if d.cls == "ball"]
        if not cands:
            self._gap += 1
            if self._gap > self.max_gap:
                self._last = None
                self._vel = np.zeros(2)
            return
        det = min(cands, key=lambda d: self._distance(d.box))
        if self._last is not None and self._distance(det.box) > self.max_jump_px * (1 + self._gap):
            self._gap += 1
            return
        centre = _centre(det.box)
        if self._last is not None:
            self._vel = 0.6 * self._vel + 0.4 * (centre - self._last)
        self._last = centre
        self._gap = 0
        self.frames.append(TrackFrame(frame, det.box, det.conf))

    def _distance(self, box: Box) -> float:
        if self._last is None:
            return 0.0
        return float(np.linalg.norm(_centre(box) - (self._last + self._vel)))

    def track(self) -> Track:
        return Track(track_id=0, cls="ball", frames=self.frames)


def _centre(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
