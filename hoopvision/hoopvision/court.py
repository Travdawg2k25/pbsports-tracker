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
# High school floors are 84 ft long; the lane and free-throw line are the same.
HS_COURT_LENGTH_FT = 84.0
LANE_WIDTH_FT = 12.0
FT_LINE_FT = 19.0
RIM_FROM_BASELINE_FT = 5.25
# A regulation rim is 18 inches across — the scale reference for the rim-only path.
RIM_DIAMETER_FT = 1.5


def hoops_ft(
    length: float = COURT_LENGTH_FT, width: float = COURT_WIDTH_FT
) -> dict[str, tuple[float, float]]:
    """Rim centres, 5.25 ft off each baseline on the centre line of the court."""
    return {
        "left": (RIM_FROM_BASELINE_FT, width / 2),
        "right": (length - RIM_FROM_BASELINE_FT, width / 2),
    }


def landmarks_ft(
    length: float = COURT_LENGTH_FT,
    width: float = COURT_WIDTH_FT,
    lane_width: float = LANE_WIDTH_FT,
    ft_line: float = FT_LINE_FT,
) -> dict[str, tuple[float, float]]:
    """Court points a human can pick out unambiguously, in feet.

    Baseline and halfcourt corners need a camera that sees the whole floor. The lane
    and free-throw line of one end are enough on their own, which is all a camera
    framed on a single basket ever shows.
    """
    mid = width / 2
    lane_near, lane_far = mid - lane_width / 2, mid + lane_width / 2
    marks: dict[str, tuple[float, float]] = {
        "halfcourt_near_sideline": (length / 2, 0.0),
        "halfcourt_far_sideline": (length / 2, width),
        "centre_circle_near": (length / 2, mid - 6.0),
        "centre_circle_far": (length / 2, mid + 6.0),
    }
    for side, baseline, inward in (("left", 0.0, 1.0), ("right", length, -1.0)):
        marks.update(
            {
                f"{side}_baseline_near_sideline": (baseline, 0.0),
                f"{side}_baseline_far_sideline": (baseline, width),
                f"{side}_lane_baseline_near": (baseline, lane_near),
                f"{side}_lane_baseline_far": (baseline, lane_far),
                f"{side}_ft_line_near": (baseline + inward * ft_line, lane_near),
                f"{side}_ft_line_far": (baseline + inward * ft_line, lane_far),
            }
        )
    return marks


HOOPS_FT: dict[str, tuple[float, float]] = hoops_ft()
LANDMARKS_FT: dict[str, tuple[float, float]] = landmarks_ft()


@dataclass
class Calibration:
    """Image->court homography plus the pixel boxes of the two rims."""

    image_points: dict[str, tuple[float, float]] = field(default_factory=dict)
    rim_boxes: dict[str, Box] = field(default_factory=dict)  # "left" / "right"
    court_length_ft: float = COURT_LENGTH_FT
    court_width_ft: float = COURT_WIDTH_FT
    lane_width_ft: float = LANE_WIDTH_FT
    ft_line_ft: float = FT_LINE_FT
    motion: Motion | None = None

    def __post_init__(self) -> None:
        self._H: np.ndarray | None = None

    @property
    def landmarks(self) -> dict[str, tuple[float, float]]:
        return landmarks_ft(
            self.court_length_ft, self.court_width_ft, self.lane_width_ft, self.ft_line_ft
        )

    @property
    def hoops(self) -> dict[str, tuple[float, float]]:
        return hoops_ft(self.court_length_ft, self.court_width_ft)

    @property
    def has_homography(self) -> bool:
        """True when enough landmarks are marked to solve a real image->court map.

        The rim-only path (auto-detected rim, no court landmarks) has none: makes,
        points and FG% still work off the rim box, but court-space distances fall back
        to an approximate pixel scale — see :meth:`to_court` and :attr:`distance_reliable`.
        """
        marks = self.landmarks
        named = [k for k in self.image_points if k in marks]
        if len(named) < 4:
            return False
        try:
            return self.homography is not None
        except ValueError:
            return False

    @property
    def distance_reliable(self) -> bool:
        """Whether court-space distances (2-vs-3, shot distance) can be trusted.

        Only a solved homography gives real feet. Without it, distance-derived stats
        (three-pointers, shot distance) are best-effort and the box score flags them so
        a human can correct them.
        """
        return self.has_homography

    @property
    def homography(self) -> np.ndarray:
        if self._H is None:
            marks = self.landmarks
            named = [(k, v) for k, v in self.image_points.items() if k in marks]
            if len(named) < 4:
                raise ValueError(
                    f"need at least 4 known landmarks, got {len(named)}: "
                    f"{sorted(self.image_points)}"
                )
            src = np.array([v for _, v in named], dtype=np.float32)
            dst = np.array([marks[k] for k, _ in named], dtype=np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                raise ValueError("homography solve failed; check the marked points")
            self._H = H
        return self._H

    @property
    def _px_per_ft(self) -> float:
        """Rough pixels-per-foot from the rim box (a rim is ~1.5 ft across).

        Used only on the rim-only path to turn pixel gaps into approximate feet so the
        possession/proximity logic keeps working. It ignores perspective, so it is a
        coarse scale, not a real projection.
        """
        widths = [abs(r[2] - r[0]) for r in self.rim_boxes.values() if r]
        if widths:
            return max(1e-3, float(np.mean(widths)) / RIM_DIAMETER_FT)
        # No rim either: assume a mid-range broadcast scale so distances stay finite.
        return 10.0

    def to_reference(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        """Pixel position on the frame the calibration was marked on."""
        if self.motion is None or frame is None:
            return float(x), float(y)
        return self.motion.warp(x, y, frame)

    def to_court(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        """Court-space position in feet.

        With a solved homography this is a true projection. On the rim-only path there is
        no homography, so the reference-frame pixel is scaled by :attr:`_px_per_ft` into
        approximate feet: good enough for "who is closest to the ball", not for real
        distances. :attr:`distance_reliable` says which case you are in.
        """
        rx, ry = self.to_reference(x, y, frame)
        if not self.has_homography:
            scale = self._px_per_ft
            return rx / scale, ry / scale
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
        hx, hy = self.hoops[hoop]
        return float(np.hypot(court_xy[0] - hx, court_xy[1] - hy))

    def save(self, path: str | Path) -> None:
        dump_json(
            {
                "image_points": {k: list(v) for k, v in self.image_points.items()},
                "rim_boxes": {k: list(v) for k, v in self.rim_boxes.items()},
                "court_length_ft": self.court_length_ft,
                "court_width_ft": self.court_width_ft,
                "lane_width_ft": self.lane_width_ft,
                "ft_line_ft": self.ft_line_ft,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        raw = load_json(path)
        return cls(
            image_points={k: tuple(v) for k, v in raw.get("image_points", {}).items()},
            rim_boxes={k: tuple(v) for k, v in raw.get("rim_boxes", {}).items()},
            court_length_ft=raw.get("court_length_ft", COURT_LENGTH_FT),
            court_width_ft=raw.get("court_width_ft", COURT_WIDTH_FT),
            lane_width_ft=raw.get("lane_width_ft", LANE_WIDTH_FT),
            ft_line_ft=raw.get("ft_line_ft", FT_LINE_FT),
        )


def shot_value(calib: Calibration, court_xy: tuple[float, float], three_radius: float) -> int:
    """2 or 3 points for a shot taken from ``court_xy``.

    Without a solved homography the distance is approximate, so we do not guess a three:
    the shot counts as 2 and the box score marks the value best-effort for a human to fix.
    """
    if not calib.distance_reliable:
        return 2
    return 3 if calib.hoop_distance_ft(court_xy) >= three_radius else 2
