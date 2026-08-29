# court_calibration.py — Purple Box Sports | Court Calibration
# ================================================================
# Provides known court geometry for fixed-camera setups.
# When the user taps hoop locations during setup, the AI knows
# exactly where to look for scores, three-pointers, etc.
# ================================================================

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PurpleBox.CourtCalibration")


# ═════════════════════════════════════════════════════════════════════════════
# COURT GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class HoopRegion:
    """Defines a hoop's location and scoring detection zone."""
    center: Tuple[int, int]           # (x, y) pixel coordinates of rim center
    radius: int = 40                  # Scoring detection radius (pixels)
    backboard_top: int = 0            # y-coordinate of backboard top
    net_bottom: int = 0               # y-coordinate below net

    @property
    def x(self) -> int:
        return self.center[0]

    @property
    def y(self) -> int:
        return self.center[1]

    @property
    def scoring_box(self) -> Tuple[int, int, int, int]:
        """Bounding box for score detection (x1, y1, x2, y2)."""
        return (
            self.x - self.radius,
            self.y - self.radius // 2,   # Slightly above rim
            self.x + self.radius,
            self.y + self.radius * 2,    # Below net
        )

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within the scoring zone."""
        dist = math.hypot(x - self.x, y - self.y)
        return dist <= self.radius

    def ball_entering_from_above(
        self,
        ball_pos: Tuple[float, float],
        prev_pos: Optional[Tuple[float, float]],
    ) -> bool:
        """
        Detect if ball is entering the hoop from above.
        Key condition for a made basket.
        """
        if prev_pos is None:
            return False

        bx, by = ball_pos
        px, py = prev_pos

        # Ball must be within horizontal range of hoop
        if abs(bx - self.x) > self.radius:
            return False

        # Ball must be moving downward
        if by <= py:
            return False

        # Ball must cross the rim y-coordinate (from above to at/below)
        if py <= self.y and by >= self.y - 10:
            # Check ball is close enough horizontally
            if abs(bx - self.x) <= self.radius * 0.8:
                return True

        return False

    def to_dict(self) -> Dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "backboard_top": self.backboard_top,
            "net_bottom": self.net_bottom,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HoopRegion":
        return cls(
            center=tuple(data["center"]),
            radius=data.get("radius", 40),
            backboard_top=data.get("backboard_top", 0),
            net_bottom=data.get("net_bottom", 0),
        )


@dataclass
class CourtCalibration:
    """
    Full court calibration from user-provided reference points.

    The user taps:
      - Left hoop center
      - Right hoop center
      - (Optional) Three-point line corners
      - (Optional) Half-court line

    From these points, we derive:
      - Exact scoring zones
      - Three-point line boundary
      - Court orientation and scale
    """
    left_hoop: Optional[HoopRegion] = None
    right_hoop: Optional[HoopRegion] = None

    # Three-point arc reference points (pixel coords)
    three_point_left_corner: Optional[Tuple[int, int]] = None
    three_point_right_corner: Optional[Tuple[int, int]] = None

    # Court boundaries
    court_left: int = 0
    court_right: int = 1280
    court_top: int = 0
    court_bottom: int = 720
    half_court_x: Optional[int] = None

    # Frame dimensions
    frame_width: int = 1280
    frame_height: int = 720

    # Derived
    _is_calibrated: bool = False

    def calibrate(
        self,
        left_hoop_center: Tuple[int, int],
        right_hoop_center: Tuple[int, int],
        frame_width: int = 1280,
        frame_height: int = 720,
        hoop_radius: int = 40,
    ) -> None:
        """
        Set up calibration from user-provided hoop positions.

        Parameters
        ----------
        left_hoop_center : (x, y) of the left hoop rim center
        right_hoop_center : (x, y) of the right hoop rim center
        frame_width, frame_height : video frame dimensions
        hoop_radius : pixel radius for score detection zone
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Scale hoop radius based on court size
        court_width_px = abs(right_hoop_center[0] - left_hoop_center[0])
        if court_width_px > 0:
            # Hoop radius proportional to court width (~3% of court width)
            hoop_radius = max(25, int(court_width_px * 0.035))

        self.left_hoop = HoopRegion(
            center=left_hoop_center,
            radius=hoop_radius,
            backboard_top=left_hoop_center[1] - int(hoop_radius * 1.5),
            net_bottom=left_hoop_center[1] + int(hoop_radius * 2),
        )

        self.right_hoop = HoopRegion(
            center=right_hoop_center,
            radius=hoop_radius,
            backboard_top=right_hoop_center[1] - int(hoop_radius * 1.5),
            net_bottom=right_hoop_center[1] + int(hoop_radius * 2),
        )

        # Derive half-court
        self.half_court_x = (left_hoop_center[0] + right_hoop_center[0]) // 2

        # Derive three-point distance (~23.75 ft = ~37% of court length in pixels)
        three_pt_dist_px = int(court_width_px * 0.28)
        self.three_point_left_corner = (
            left_hoop_center[0] + three_pt_dist_px,
            frame_height // 2,
        )
        self.three_point_right_corner = (
            right_hoop_center[0] - three_pt_dist_px,
            frame_height // 2,
        )

        self._is_calibrated = True

        logger.info(
            "Court calibrated | left_hoop=%s right_hoop=%s | "
            "hoop_radius=%dpx | court_width=%dpx | half_court_x=%d",
            left_hoop_center, right_hoop_center,
            hoop_radius, court_width_px, self.half_court_x,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def get_nearest_hoop(self, x: float, y: float) -> Optional[HoopRegion]:
        """Return the hoop closest to the given position."""
        if not self._is_calibrated:
            return None

        dist_left = math.hypot(x - self.left_hoop.x, y - self.left_hoop.y)
        dist_right = math.hypot(x - self.right_hoop.x, y - self.right_hoop.y)

        if dist_left < dist_right:
            return self.left_hoop
        return self.right_hoop

    def is_three_point(self, shooter_x: float, shooter_y: float, target_hoop: HoopRegion) -> bool:
        """
        Determine if a shooter's position is beyond the three-point line.
        Uses distance from the hoop center.
        """
        if not self._is_calibrated:
            return False

        dist = math.hypot(shooter_x - target_hoop.x, shooter_y - target_hoop.y)
        # Three-point line is roughly 28% of court width from the hoop
        court_width = abs(self.right_hoop.x - self.left_hoop.x)
        three_pt_dist = court_width * 0.28

        return dist >= three_pt_dist

    def get_court_zone(self, x: float, y: float) -> str:
        """Classify a position into a court zone."""
        if not self._is_calibrated:
            return "unknown"

        # Determine which half
        if x < self.half_court_x:
            hoop = self.left_hoop
        else:
            hoop = self.right_hoop

        dist = math.hypot(x - hoop.x, y - hoop.y)
        court_width = abs(self.right_hoop.x - self.left_hoop.x)

        # Paint (~8% of court width from hoop)
        paint_dist = court_width * 0.09
        # Mid-range (between paint and three-point)
        three_pt_dist = court_width * 0.28

        if dist <= paint_dist:
            return "paint"
        elif dist <= three_pt_dist:
            return "mid_range"
        else:
            return "three_point"

    def check_score(
        self,
        ball_pos: Tuple[float, float],
        prev_ball_pos: Optional[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the ball has scored at either hoop.

        Returns a dict with score info if detected, None otherwise.
        """
        if not self._is_calibrated or ball_pos is None:
            return None

        for side, hoop in [("left", self.left_hoop), ("right", self.right_hoop)]:
            if hoop.ball_entering_from_above(ball_pos, prev_ball_pos):
                return {
                    "scored": True,
                    "hoop_side": side,
                    "hoop_center": hoop.center,
                    "ball_pos": ball_pos,
                }

        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save calibration to JSON file."""
        data = {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "left_hoop": self.left_hoop.to_dict() if self.left_hoop else None,
            "right_hoop": self.right_hoop.to_dict() if self.right_hoop else None,
            "half_court_x": self.half_court_x,
            "court_left": self.court_left,
            "court_right": self.court_right,
            "is_calibrated": self._is_calibrated,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "CourtCalibration":
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        cal = cls(
            frame_width=data.get("frame_width", 1280),
            frame_height=data.get("frame_height", 720),
            half_court_x=data.get("half_court_x"),
            court_left=data.get("court_left", 0),
            court_right=data.get("court_right", 1280),
        )

        if data.get("left_hoop"):
            cal.left_hoop = HoopRegion.from_dict(data["left_hoop"])
        if data.get("right_hoop"):
            cal.right_hoop = HoopRegion.from_dict(data["right_hoop"])

        cal._is_calibrated = data.get("is_calibrated", False)
        return cal

    def to_dict(self) -> Dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "left_hoop": self.left_hoop.to_dict() if self.left_hoop else None,
            "right_hoop": self.right_hoop.to_dict() if self.right_hoop else None,
            "half_court_x": self.half_court_x,
            "is_calibrated": self._is_calibrated,
        }


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATED SCORE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
class CalibratedScoreDetector:
    """
    Score detection using known hoop positions from calibration.

    Much more accurate than trajectory-based detection because we know
    exactly where the hoops are and just check if the ball passes through.
    """

    def __init__(self, calibration: CourtCalibration):
        self.cal = calibration
        self._prev_ball_pos: Optional[Tuple[float, float]] = None
        self._cooldown_frames: int = 0
        self._cooldown_max: int = 30  # Frames to wait between score detections
        self._ball_in_hoop_frames: Dict[str, int] = {"left": 0, "right": 0}
        self._ball_near_hoop_threshold: int = 3  # Frames ball must be in zone

    def update(
        self,
        ball_pos: Optional[Tuple[float, float]],
        frame_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a score occurred this frame.

        Returns score info dict or None.
        """
        if not self.cal.is_calibrated:
            self._prev_ball_pos = ball_pos
            return None

        # Cooldown between scores
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            self._prev_ball_pos = ball_pos
            return None

        if ball_pos is None:
            self._prev_ball_pos = None
            return None

        bx, by = ball_pos

        # Check each hoop
        for side, hoop in [("left", self.cal.left_hoop), ("right", self.cal.right_hoop)]:
            # Method 1: Ball entering from above (trajectory-based)
            if self._prev_ball_pos is not None:
                if hoop.ball_entering_from_above(ball_pos, self._prev_ball_pos):
                    self._cooldown_frames = self._cooldown_max
                    self._prev_ball_pos = ball_pos
                    logger.info(
                        "SCORE DETECTED (trajectory) at %s hoop | ball=%s | frame=%d",
                        side, ball_pos, frame_id,
                    )
                    return {
                        "scored": True,
                        "method": "trajectory",
                        "hoop_side": side,
                        "hoop_center": hoop.center,
                        "ball_pos": ball_pos,
                        "frame_id": frame_id,
                    }

            # Method 2: Ball dwelling in scoring zone (backup)
            if hoop.contains_point(int(bx), int(by)):
                self._ball_in_hoop_frames[side] += 1
                if self._ball_in_hoop_frames[side] >= self._ball_near_hoop_threshold:
                    # Ball has been in the hoop zone for several frames
                    # Check it entered from above (y was decreasing then increasing)
                    if self._prev_ball_pos and self._prev_ball_pos[1] < by:
                        self._cooldown_frames = self._cooldown_max
                        self._ball_in_hoop_frames[side] = 0
                        self._prev_ball_pos = ball_pos
                        logger.info(
                            "SCORE DETECTED (dwell) at %s hoop | ball=%s | frame=%d",
                            side, ball_pos, frame_id,
                        )
                        return {
                            "scored": True,
                            "method": "dwell",
                            "hoop_side": side,
                            "hoop_center": hoop.center,
                            "ball_pos": ball_pos,
                            "frame_id": frame_id,
                        }
            else:
                self._ball_in_hoop_frames[side] = 0

        self._prev_ball_pos = ball_pos
        return None

    def reset(self) -> None:
        self._prev_ball_pos = None
        self._cooldown_frames = 0
        self._ball_in_hoop_frames = {"left": 0, "right": 0}


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-CALIBRATION (from first N frames of video)
# ═════════════════════════════════════════════════════════════════════════════
class AutoCalibrator:
    """
    Attempts to auto-detect hoop positions from the first N frames
    by looking for consistent rim detections from the YOLO model.

    Falls back to manual calibration if rim isn't consistently detected.
    """

    def __init__(self, required_detections: int = 20, frames_to_scan: int = 300):
        self.required_detections = required_detections
        self.frames_to_scan = frames_to_scan
        self._rim_positions: List[Tuple[float, float]] = []

    def add_rim_detection(self, x: float, y: float) -> None:
        """Add a rim detection from a frame."""
        self._rim_positions.append((x, y))

    def attempt_calibration(self, frame_width: int, frame_height: int) -> Optional[CourtCalibration]:
        """
        Try to determine hoop positions from collected rim detections.
        Uses clustering to find the two hoop locations.
        """
        if len(self._rim_positions) < self.required_detections:
            logger.warning(
                "Not enough rim detections for auto-calibration (%d/%d)",
                len(self._rim_positions), self.required_detections,
            )
            return None

        positions = np.array(self._rim_positions)

        # Simple clustering: split by x-coordinate (left half vs right half)
        mid_x = frame_width / 2
        left_rims = positions[positions[:, 0] < mid_x]
        right_rims = positions[positions[:, 0] >= mid_x]

        if len(left_rims) < 5 or len(right_rims) < 5:
            # Maybe only one hoop visible — use that
            if len(left_rims) >= 10:
                left_center = (int(np.median(left_rims[:, 0])), int(np.median(left_rims[:, 1])))
                # Estimate right hoop as mirror
                right_center = (frame_width - left_center[0], left_center[1])
            elif len(right_rims) >= 10:
                right_center = (int(np.median(right_rims[:, 0])), int(np.median(right_rims[:, 1])))
                left_center = (frame_width - right_center[0], right_center[1])
            else:
                logger.warning("Cannot determine hoop positions from detections")
                return None
        else:
            left_center = (int(np.median(left_rims[:, 0])), int(np.median(left_rims[:, 1])))
            right_center = (int(np.median(right_rims[:, 0])), int(np.median(right_rims[:, 1])))

        cal = CourtCalibration()
        cal.calibrate(left_center, right_center, frame_width, frame_height)

        logger.info(
            "Auto-calibration successful! left=%s right=%s (%d rim detections used)",
            left_center, right_center, len(self._rim_positions),
        )

        return cal

    def reset(self) -> None:
        self._rim_positions.clear()


# ═════════════════════════════════════════════════════════════════════════════
# CLI TOOL — Manual calibration from first frame
# ═════════════════════════════════════════════════════════════════════════════
def calibrate_from_video(video_path: str, output_path: str = "calibration.json") -> CourtCalibration:
    """
    Open video, show first frame, let user click hoop positions.
    Saves calibration to JSON.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")

    h, w = frame.shape[:2]
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 3)
            cv2.imshow("Calibrate — Click LEFT hoop, then RIGHT hoop", frame)
            if len(clicks) == 2:
                cv2.destroyAllWindows()

    cv2.imshow("Calibrate — Click LEFT hoop, then RIGHT hoop", frame)
    cv2.setMouseCallback("Calibrate — Click LEFT hoop, then RIGHT hoop", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicks) < 2:
        raise RuntimeError("Need 2 clicks (left hoop, right hoop)")

    cal = CourtCalibration()
    cal.calibrate(clicks[0], clicks[1], w, h)
    cal.save(output_path)

    print(f"Calibration saved: {output_path}")
    print(f"  Left hoop: {clicks[0]}")
    print(f"  Right hoop: {clicks[1]}")
    print(f"  Frame: {w}x{h}")

    return cal


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python court_calibration.py <video_path> [output.json]")
        sys.exit(1)

    video = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "calibration.json"
    calibrate_from_video(video, output)
