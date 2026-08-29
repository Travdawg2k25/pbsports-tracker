# =============================================================================
# utils.py — Purple Box Sports
# Shared utility functions used across all modules
# =============================================================================

from __future__ import annotations

import colorsys
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import (
    Any, Deque, Dict, List, Optional, Sequence, Tuple, Union
)

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(
    level:    str  = "INFO",
    log_file: Optional[str] = None,
) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)


# =============================================================================
# BOUNDING BOX HELPERS
# =============================================================================

BBox = Tuple[float, float, float, float]   # x1, y1, x2, y2


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_distance(a: BBox, b: BBox) -> float:
    """Euclidean distance between bbox centers."""
    ca, cb = bbox_center(a), bbox_center(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def point_in_bbox(
    point:  Tuple[float, float],
    box:    BBox,
    margin: float = 0.0,
) -> bool:
    x, y = point
    return (
        box[0] - margin <= x <= box[2] + margin
        and box[1] - margin <= y <= box[3] + margin
    )


def expand_bbox(
    box:    BBox,
    pad:    int,
    w_max:  int = 9999,
    h_max:  int = 9999,
) -> BBox:
    return (
        max(0,     box[0] - pad),
        max(0,     box[1] - pad),
        min(w_max, box[2] + pad),
        min(h_max, box[3] + pad),
    )


def xyxy_to_xywh(box: BBox) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(box: Tuple[float, float, float, float]) -> BBox:
    x, y, w, h = box
    return x, y, x + w, y + h


def clip_bbox(box: BBox, w: int, h: int) -> BBox:
    return (
        max(0, min(box[0], w)),
        max(0, min(box[1], h)),
        max(0, min(box[2], w)),
        max(0, min(box[3], h)),
    )


def crop_bbox(frame: np.ndarray, box: BBox, pad: int = 0) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(box, pad, w, h)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


# =============================================================================
# DISTANCE / GEOMETRY
# =============================================================================

def euclidean(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def normalize_point(
    pt: Tuple[float, float],
    w:  int,
    h:  int,
) -> Tuple[float, float]:
    return (pt[0] / max(1, w), pt[1] / max(1, h))


def angle_between(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Angle in degrees from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def velocity(
    prev: Tuple[float, float],
    curr: Tuple[float, float],
    dt:   float = 1.0,
) -> Tuple[float, float]:
    dt = max(1e-6, dt)
    return ((curr[0] - prev[0]) / dt, (curr[1] - prev[1]) / dt)


def speed(
    prev: Tuple[float, float],
    curr: Tuple[float, float],
    dt:   float = 1.0,
) -> float:
    vx, vy = velocity(prev, curr, dt)
    return math.hypot(vx, vy)


# =============================================================================
# COLOR HELPERS
# =============================================================================

def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color from track ID."""
    hue = (track_id * 37) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def team_color(team_id: Optional[int]) -> Tuple[int, int, int]:
    palette = {
        0: (255, 100,  50),
        1: ( 50, 100, 255),
    }
    return palette.get(team_id, (200, 200, 200))


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_bbox(
    frame:      np.ndarray,
    box:        BBox,
    color:      Tuple[int, int, int] = (0, 255, 0),
    thickness:  int   = 2,
    label:      str   = "",
    font_scale: float = 0.55,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Prevent top clipping if label is rendered near upper frame edge
        if y1 - th - 6 < 0:
            bg_y1, bg_y2 = y1, y1 + th + 6
            text_y = y1 + th + 2
        else:
            bg_y1, bg_y2 = y1 - th - 6, y1
            text_y = y1 - 4

        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )


def draw_trail(
    frame:   np.ndarray,
    history: Sequence[Tuple[float, float]],
    color:   Tuple[int, int, int] = (0, 200, 255),
    max_len: int = 30,
) -> None:
    pts = list(history)[-max_len:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]),     int(pts[i][1]))
        cv2.line(frame, p1, p2, c, 2, cv2.LINE_AA)


def draw_text_box(
    frame:      np.ndarray,
    text:       str,
    origin:     Tuple[int, int],
    color:      Tuple[int, int, int] = (30, 30, 30),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.55,
    alpha:      float = 0.70,
) -> None:
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    x, y = origin
    x2, y1_box, y2_box = x + tw + 8, y - th - 6, y + 4
    h, w = frame.shape[:2]
    
    # Clip ROI boundaries to frame dimensions
    x1_c, x2_c = max(0, x), min(w, x2)
    y1_c, y2_c = max(0, y1_box), min(h, y2_box)

    if x2_c > x1_c and y2_c > y1_c:
        roi = frame[y1_c:y2_c, x1_c:x2_c]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2_c - x1_c, y2_c - y1_c), color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    cv2.putText(
        frame, text, (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
    )


def draw_scoreboard(
    frame:      np.ndarray,
    score:      Dict[int, int],
    team_names: Dict[int, str],
    frame_id:   int,
    fps:        float = 30.0,
) -> None:
    h, w = frame.shape[:2]
    board_w, board_h = 340, 56
    bx = (w - board_w) // 2
    by = 10
    
    # Local ROI blend to avoid copying full frame
    bx1, bx2 = max(0, bx), min(w, bx + board_w)
    by1, by2 = max(0, by), min(h, by + board_h)
    
    if bx2 > bx1 and by2 > by1:
        roi = frame[by1:by2, bx1:bx2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (bx2 - bx1, by2 - by1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, roi, 0.25, 0, roi)

    elapsed = int(frame_id / max(1.0, fps))
    mins, secs = divmod(elapsed, 60)
    time_str = f"{mins:02d}:{secs:02d}"

    teams = list(score.items())
    if len(teams) >= 2:
        t0, s0 = teams[0]
        t1, s1 = teams[1]
        n0 = team_names.get(t0, f"T{t0}")
        n1 = team_names.get(t1, f"T{t1}")
        line = f"{n0}  {s0}  —  {s1}  {n1}    {time_str}"
    else:
        line = time_str

    cv2.putText(
        frame, line,
        (bx + 8, by + 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


# =============================================================================
# FRAME / VIDEO HELPERS
# =============================================================================

def resize_frame(
    frame:  np.ndarray,
    width:  int,
    height: int,
) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def is_frame_blurry(frame: np.ndarray, threshold: float = 80.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


# =============================================================================
# MATH / SIGNAL HELPERS
# =============================================================================

def moving_average(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    result = []
    dq: Deque[float] = deque(maxlen=window)
    for v in values:
        dq.append(v)
        result.append(sum(dq) / len(dq))
    return result


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


# =============================================================================
# TIMING
# =============================================================================

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 1e-6 else 0.0

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 1e-6 else 0.0


# =============================================================================
# FILE HELPERS
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_filename(base: str, ext: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"


__all__ = [
    "setup_logging",
    "BBox",
    "bbox_area", "bbox_center", "bbox_iou", "bbox_distance",
    "point_in_bbox", "expand_bbox", "xyxy_to_xywh", "xywh_to_xyxy",
    "clip_bbox", "crop_bbox",
    "euclidean", "normalize_point", "angle_between", "velocity", "speed",
    "id_to_color", "team_color",
    "draw_bbox", "draw_trail", "draw_text_box",
    "draw_scoreboard",
    "resize_frame", "frame_to_rgb", "is_frame_blurry",
    "moving_average", "clamp", "safe_divide",
    "FPSCounter",
    "ensure_dir", "timestamp_filename",
]