"""Camera motion compensation.

A tripod camera still drifts, and a lot of "single camera" game footage is actually a
panning/zooming operator shot. Either way the court calibration is only valid for the
frame it was marked on, so every frame gets a homography back to that reference frame
and all geometry is done in reference pixels.

The scene is rigid apart from the players, so ORB features plus a RANSAC homography
recovers the camera motion. Matching against the reference directly avoids drift; when
the pan is large enough that the reference no longer overlaps, the estimate chains
through the previous frame instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import dump_json, load_json
from .video import iter_frames, probe

log = logging.getLogger("hoopvision")

MIN_INLIERS = 30
MIN_MATCHES = 12
# Feature matching gains nothing from 4K pixels and costs a second a frame at it.
FEATURE_WIDTH = 1280


@dataclass
class Motion:
    """Per-frame homography mapping frame pixels onto the reference frame."""

    reference_frame: int = 0
    homographies: dict[int, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.homographies is None:
            self.homographies = {}

    def matrix(self, frame: int) -> np.ndarray:
        if not self.homographies:
            return np.eye(3)
        H = self.homographies.get(frame)
        if H is not None:
            return H
        nearest = min(self.homographies, key=lambda f: abs(f - frame))
        return self.homographies[nearest]

    def warp(self, x: float, y: float, frame: int) -> tuple[float, float]:
        H = self.matrix(frame)
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)[0][0]
        return float(out[0]), float(out[1])

    def warp_box(self, box, frame: int):
        x1, y1 = self.warp(box[0], box[1], frame)
        x2, y2 = self.warp(box[2], box[3], frame)
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def save(self, path: str | Path) -> None:
        dump_json(
            {
                "reference_frame": self.reference_frame,
                "homographies": {
                    str(f): H.tolist() for f, H in sorted(self.homographies.items())
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Motion:
        raw = load_json(path)
        return cls(
            reference_frame=raw["reference_frame"],
            homographies={
                int(f): np.array(H, dtype=np.float64)
                for f, H in raw["homographies"].items()
            },
        )


def floor_mask(frame: np.ndarray) -> np.ndarray:
    """Mask of the wood playing surface.

    The gym is not one plane: walls, bleachers and the ceiling truss move differently
    from the floor under the same camera rotation, and a homography fitted to them
    shears the court. Restricting features to the floor keeps the estimate on the
    plane the geometry actually uses.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 60, 80), (30, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))


class _Features:
    def __init__(self, max_features: int = 4000, mask_floor: bool = True):
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.mask_floor = mask_floor

    def describe(self, frame: np.ndarray):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = floor_mask(frame) if self.mask_floor else None
        if mask is not None and mask.mean() < 10:
            mask = None
        return self.orb.detectAndCompute(grey, mask)

    def homography(self, src, dst) -> tuple[np.ndarray | None, int]:
        """Homography mapping the ``src`` frame's pixels onto the ``dst`` frame's."""
        (kp_s, des_s), (kp_d, des_d) = src, dst
        if des_s is None or des_d is None or len(kp_s) < MIN_MATCHES or len(kp_d) < MIN_MATCHES:
            return None, 0
        matches = self.matcher.match(des_s, des_d)
        if len(matches) < MIN_MATCHES:
            return None, 0
        p_s = np.float32([kp_s[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p_d = np.float32([kp_d[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(p_s, p_d, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        return H, int(mask.sum())


def _downscale(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    if width <= 0 or frame.shape[1] <= width:
        return frame, 1.0
    s = width / frame.shape[1]
    return cv2.resize(frame, (width, int(round(frame.shape[0] * s)))), s


def rescale(H: np.ndarray, scale: float) -> np.ndarray:
    """A homography found at ``scale`` of full size, expressed in full-size pixels."""
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(S) @ H @ S


def estimate_motion(
    video: str | Path,
    reference_frame: int = 0,
    stride: int = 1,
    end: int | None = None,
    mask_floor: bool = True,
    feature_width: int = FEATURE_WIDTH,
) -> Motion:
    """Estimate every processed frame's homography back to ``reference_frame``.

    Homographies are in full-resolution pixels regardless of ``feature_width``, which
    only controls the resolution features are matched at.
    """
    feat = _Features(mask_floor=mask_floor)
    motion = Motion(reference_frame=reference_frame)
    ref = None
    prev = None
    prev_H = np.eye(3)
    chained = 0
    scale = 1.0
    for idx, frame in iter_frames(video, stride=stride, end=end):
        small, scale = _downscale(frame, feature_width)
        desc = feat.describe(small)
        if ref is None:
            ref = desc
            motion.homographies[idx] = np.eye(3)
            prev, prev_H = desc, np.eye(3)
            continue
        H, inliers = feat.homography(desc, ref)
        if H is None or inliers < MIN_INLIERS:
            # Too little overlap with the reference: step through the previous frame.
            step, step_inliers = feat.homography(desc, prev) if prev else (None, 0)
            if step is not None and step_inliers >= MIN_MATCHES:
                H = prev_H @ step
                chained += 1
            else:
                H = prev_H
        motion.homographies[idx] = H
        prev, prev_H = desc, H
    if scale != 1.0:
        motion.homographies = {
            f: rescale(H, scale) for f, H in motion.homographies.items()
        }
    if chained:
        log.info("motion: %d/%d frames chained from the previous frame",
                 chained, len(motion.homographies))
    return motion


def mosaic(
    video: str | Path,
    motion: Motion,
    stride: int = 10,
    scale: float = 1.0,
    max_side: int = 12000,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Stitch the clip into one image in reference-frame space.

    A tight pan never shows the whole court on one frame, so landmarks get marked on
    the mosaic instead. Returns the image and the ``(dx, dy)`` offset added to
    ``scale``-multiplied reference pixels to place them on it.
    """
    meta = probe(video)
    w, h = meta.width, meta.height
    box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = np.concatenate(
        [cv2.perspectiveTransform(box, H) for H in motion.homographies.values()]
    ).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx, dy = float(-x0 * scale), float(-y0 * scale)
    out_w = int((x1 - x0) * scale) + 1
    out_h = int((y1 - y0) * scale) + 1
    if max(out_w, out_h) > max_side:
        raise ValueError(f"mosaic would be {out_w}x{out_h}; camera motion looks unstable")
    shift = np.array([[scale, 0, dx], [0, scale, dy], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for idx, frame in iter_frames(video, stride=stride):
        warped = cv2.warpPerspective(frame, shift @ motion.matrix(idx), (out_w, out_h))
        empty = (canvas.sum(axis=2) == 0) & (warped.sum(axis=2) > 0)
        canvas[empty] = warped[empty]
    return canvas, (dx, dy)


def displacement(motion: Motion, width: int, height: int) -> float:
    """Largest movement of the frame centre, in pixels — how much the camera moved."""
    cx, cy = width / 2, height / 2
    return max(
        (float(np.hypot(*(np.subtract(motion.warp(cx, cy, f), (cx, cy)))))
         for f in motion.homographies),
        default=0.0,
    )
