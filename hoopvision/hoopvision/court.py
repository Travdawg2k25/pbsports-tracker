"""Court calibration: image pixels <-> court coordinates in feet.

Calibration marks at least four known court landmarks plus a pixel box around each rim
on one reference frame; everything the event logic needs (distance to the hoop, three-
point line, which end of the floor) is derived from the court-space projection of the
player's feet.

One homography covers the whole game only while the camera holds still. Attaching a
:class:`~hoopvision.stabilize.Motion` maps each frame back onto the reference frame
first, so the same calibration survives a camera that pans, zooms or drifts.

Court frame: origin at the left baseline/sideline corner, x along the length, y across
the width, both in feet. Default dimensions are a 94x50 ft full court.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .stabilize import Motion
from .types import Box, dump_json, load_json

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
# Rim centres, 5.25 ft from the baseline on the centre line of the court.
HOOPS_FT: dict[str, tuple[float, float]] = {
    "left": (5.25, 25.0),
    "right": (COURT_LENGTH_FT - 5.25, 25.0),
}

# Landmarks a human can point at unambiguously on a sideline view.
LANDMARKS_FT: dict[str, tuple[float, float]] = {
    "left_baseline_near_sideline": (0.0, 0.0),
    "left_baseline_far_sideline": (0.0, COURT_WIDTH_FT),
    "right_baseline_near_sideline": (COURT_LENGTH_FT, 0.0),
    "right_baseline_far_sideline": (COURT_LENGTH_FT, COURT_WIDTH_FT),
    "halfcourt_near_sideline": (COURT_LENGTH_FT / 2, 0.0),
    "halfcourt_far_sideline": (COURT_LENGTH_FT / 2, COURT_WIDTH_FT),
}


@dataclass
class Calibration:
    """Image->court homography plus the pixel boxes of the two rims."""

    image_points: dict[str, tuple[float, float]] = field(default_factory=dict)
    rim_boxes: dict[str, Box] = field(default_factory=dict)  # "left" / "right"
    court_length_ft: float = COURT_LENGTH_FT
    court_width_ft: float = COURT_WIDTH_FT
    motion: Motion | None = None

    def __post_init__(self) -> None:
        self._H: np.ndarray | None = None

    @property
    def homography(self) -> np.ndarray:
        if self._H is None:
            named = [(k, v) for k, v in self.image_points.items() if k in LANDMARKS_FT]
            if len(named) < 4:
                raise ValueError(
                    f"need at least 4 known landmarks, got {len(named)}: "
                    f"{sorted(self.image_points)}"
                )
            src = np.array([v for _, v in named], dtype=np.float32)
            dst = np.array([LANDMARKS_FT[k] for k, _ in named], dtype=np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                raise ValueError("homography solve failed; check the marked points")
            self._H = H
        return self._H

    def to_reference(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        """Pixel position on the frame the calibration was marked on."""
        if self.motion is None or frame is None:
            return float(x), float(y)
        return self.motion.warp(x, y, frame)

    def to_court(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        rx, ry = self.to_reference(x, y, frame)
        pt = np.array([[[rx, ry]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return float(out[0]), float(out[1])

    def foot_point(self, box: Box, frame: int | None = None) -> tuple[float, float]:
        """Court position of a player box, taken at the bottom centre (their feet)."""
        return self.to_court((box[0] + box[2]) / 2, box[3], frame)

    def court_pixel_box(self) -> Box | None:
        """Bounding pixel box of the marked landmarks — the ball search region."""
        if not self.image_points:
            return None
        xs = [p[0] for p in self.image_points.values()]
        ys = [p[1] for p in self.image_points.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    def nearest_hoop(self, court_xy: tuple[float, float]) -> str:
        return "left" if court_xy[0] < self.court_length_ft / 2 else "right"

    def hoop_distance_ft(self, court_xy: tuple[float, float], hoop: str | None = None) -> float:
        hoop = hoop or self.nearest_hoop(court_xy)
        hx, hy = HOOPS_FT[hoop]
        return float(np.hypot(court_xy[0] - hx, court_xy[1] - hy))

    def save(self, path: str | Path) -> None:
        dump_json(
            {
                "image_points": {k: list(v) for k, v in self.image_points.items()},
                "rim_boxes": {k: list(v) for k, v in self.rim_boxes.items()},
                "court_length_ft": self.court_length_ft,
                "court_width_ft": self.court_width_ft,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        raw = load_json(path)
        return cls(
            image_points={k: tuple(v) for k, v in raw["image_points"].items()},
            rim_boxes={k: tuple(v) for k, v in raw.get("rim_boxes", {}).items()},
            court_length_ft=raw.get("court_length_ft", COURT_LENGTH_FT),
            court_width_ft=raw.get("court_width_ft", COURT_WIDTH_FT),
        )


def shot_value(calib: Calibration, court_xy: tuple[float, float], three_radius: float) -> int:
    """2 or 3 points for a shot taken from ``court_xy``."""
    return 3 if calib.hoop_distance_ft(court_xy) >= three_radius else 2
