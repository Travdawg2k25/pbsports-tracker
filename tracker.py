# =============================================================================
# tracker.py  —  Purple Box Sports
# Multi-object tracking for basketball analytics
# =============================================================================

from __future__ import annotations

import time
import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, Any, Deque, Iterator
)

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MAX_LOST          = 30      # frames before track is DEAD
DEFAULT_MIN_HITS          = 3       # frames before TENTATIVE → CONFIRMED
DEFAULT_IOU_THRESHOLD     = 0.3     # IoU match threshold
DEFAULT_DISTANCE_THRESH   = 150.0   # pixel distance match threshold
DEFAULT_PROXIMITY_THRESH  = 60.0    # possession proximity (px)
DEFAULT_NMS_IOU           = 0.45    # NMS suppression threshold
KALMAN_PROCESS_NOISE      = 1e-2
KALMAN_MEASURE_NOISE      = 1e-1
KALMAN_ERROR_COV          = 1.0
MAX_TRAJECTORY_LEN        = 90      # frames of trajectory history
BALL_TRAJECTORY_LEN       = 60
RIM_STABILITY_FRAMES      = 10

# =============================================================================
# ENUMS
# =============================================================================

class TrackState(Enum):
    TENTATIVE = auto()   # newly created, not yet confirmed
    CONFIRMED = auto()   # seen enough times to trust
    LOST      = auto()   # missed for a few frames
    DEAD      = auto()   # exceeded max_lost, remove


class ObjectClass(Enum):
    PLAYER   = "player"
    BALL     = "ball"
    RIM      = "rim"
    REFEREE  = "referee"
    COACH    = "coach"
    UNKNOWN  = "unknown"


class EventType(Enum):
    SCORE    = "score"
    REBOUND  = "rebound"
    STEAL    = "steal"
    ASSIST   = "assist"
    BLOCK    = "block"
    TURNOVER = "turnover"
    FOUL     = "foul"


# =============================================================================
# DETECTION  (raw output from YOLO / any detector)
# =============================================================================

@dataclass
class Detection:
    """
    Single detection from a model inference pass.

    bbox   : (x1, y1, x2, y2) in pixels
    conf   : confidence score  [0, 1]
    class_id   : integer class index
    class_name : string label
    frame_id   : source frame number
    extra      : arbitrary metadata (jersey crop, embedding …)
    """
    bbox:       Tuple[float, float, float, float]
    conf:       float
    class_id:   int
    class_name: str
    frame_id:   int  = 0
    extra:      Dict = field(default_factory=dict)

    # ── derived geometry ────────────────────────────────────────────────────

    @property
    def x1(self) -> float: return self.bbox[0]

    @property
    def y1(self) -> float: return self.bbox[1]

    @property
    def x2(self) -> float: return self.bbox[2]

    @property
    def y2(self) -> float: return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0,
                (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """(top-left-x, top-left-y, width, height)"""
        return (self.x1, self.y1, self.width, self.height)

    def scale(self, sx: float, sy: float) -> "Detection":
        """Return a new Detection scaled by (sx, sy)."""
        return Detection(
            bbox=(self.x1 * sx, self.y1 * sy,
                  self.x2 * sx, self.y2 * sy),
            conf=self.conf,
            class_id=self.class_id,
            class_name=self.class_name,
            frame_id=self.frame_id,
            extra=self.extra.copy(),
        )

    def iou(self, other: "Detection") -> float:
        return _compute_iou(self.bbox, other.bbox)

    def __repr__(self) -> str:
        cx, cy = self.center
        return (f"Detection({self.class_name} "
                f"conf={self.conf:.2f} "
                f"center=({cx:.0f},{cy:.0f}))")


# =============================================================================
# KALMAN FILTER  (constant-velocity, 4-state: x,y,vx,vy)
# =============================================================================

class _KalmanFilter:
    """
    Lightweight 2-D constant-velocity Kalman filter.
    State  : [x, y, vx, vy]
    Measure: [x, y]
    """

    def __init__(self, x: float, y: float):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)

        # State transition
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.Q = np.eye(4, dtype=np.float64) * KALMAN_PROCESS_NOISE
        self.R = np.eye(2, dtype=np.float64) * KALMAN_MEASURE_NOISE
        self.P = np.eye(4, dtype=np.float64) * KALMAN_ERROR_COV

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, mx: float, my: float) -> Tuple[float, float]:
        z = np.array([[mx], [my]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


# =============================================================================
# TRACK
# =============================================================================

class Track:
    """
    A single tracked object with full state history.
    """

    _next_id: int = 1

    @classmethod
    def _new_id(cls) -> int:
        tid = cls._next_id
        cls._next_id += 1
        return tid

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 1

    def __init__(
        self,
        detection:   Detection,
        max_lost:    int = DEFAULT_MAX_LOST,
        min_hits:    int = DEFAULT_MIN_HITS,
        track_id:    Optional[int] = None,
    ):
        self.track_id:    int        = track_id or Track._new_id()
        self.class_name:  str        = detection.class_name
        self.class_id:    int        = detection.class_id
        self.state:       TrackState = TrackState.TENTATIVE

        # Geometry
        self.bbox:        Tuple[float, float, float, float] = detection.bbox
        self.conf:        float = detection.conf

        # Kalman filter initialised at detection center
        cx, cy = detection.center
        self._kf = _KalmanFilter(cx, cy)

        # Counters
        self.hits:        int = 1
        self.age:         int = 1
        self.frames_lost: int = 0
        self.max_lost:    int = max_lost
        self.min_hits:    int = min_hits

        # Metadata
        self.jersey_number: Optional[str] = detection.extra.get("jersey")
        self.team_id:       Optional[int] = detection.extra.get("team_id")
        self.extra:         Dict          = detection.extra.copy()

        # History
        self.trajectory: Deque[Tuple[float, float]] = deque(
            maxlen=MAX_TRAJECTORY_LEN
        )
        self.trajectory.append(detection.center)
        self.bbox_history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=MAX_TRAJECTORY_LEN
        )
        self.bbox_history.append(detection.bbox)

        # Frame reference
        self.last_frame_id: int = detection.frame_id

    # ── State transitions ────────────────────────────────────────────────────

    def mark_hit(self, detection: Detection) -> None:
        """Call when a detection is matched to this track."""
        cx, cy = detection.center
        self._kf.predict()
        self._kf.update(cx, cy)

        self.bbox      = detection.bbox
        self.conf      = detection.conf
        self.hits     += 1
        self.age      += 1
        self.frames_lost = 0
        self.last_frame_id = detection.frame_id

        self.trajectory.append(self._kf.position)
        self.bbox_history.append(detection.bbox)

        if (self.state == TrackState.TENTATIVE
                and self.hits >= self.min_hits):
            self.state = TrackState.CONFIRMED

        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        # Update metadata if provided
        if "jersey" in detection.extra:
            self.jersey_number = detection.extra["jersey"]
        if "team_id" in detection.extra:
            self.team_id = detection.extra["team_id"]

    def mark_missed(self) -> None:
        """Call when no detection matched this track."""
        self._kf.predict()
        self.frames_lost += 1
        self.age         += 1

        predicted = self._kf.position
        self.trajectory.append(predicted)

        if self.frames_lost >= self.max_lost:
            self.state = TrackState.DEAD
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    # ── Geometry helpers ─────────────────────────────────────────────────────

    @property
    def center(self) -> Tuple[float, float]:
        return self._kf.position

    @property
    def predicted_center(self) -> Tuple[float, float]:
        return self._kf.position

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._kf.velocity

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        cx, cy = self.center
        return math.hypot(cx - point[0], cy - point[1])

    def distance_to_track(self, other: "Track") -> float:
        return self.distance_to_point(other.center)

    def iou_with_detection(self, det: Detection) -> float:
        return _compute_iou(self.bbox, det.bbox)

    def iou_with_track(self, other: "Track") -> float:
        return _compute_iou(self.bbox, other.bbox)

    # ── Motion helpers ───────────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    @property
    def is_moving(self) -> bool:
        return self.speed > 2.0

    def predicted_position_in(self, n_frames: int) -> Tuple[float, float]:
        cx, cy   = self.center
        vx, vy   = self.velocity
        return (cx + vx * n_frames, cy + vy * n_frames)

    def recent_displacement(self, n: int = 5) -> Tuple[float, float]:
        traj = list(self.trajectory)
        if len(traj) < 2:
            return (0.0, 0.0)
        n = min(n, len(traj) - 1)
        x0, y0 = traj[-n - 1]
        x1, y1 = traj[-1]
        return (x1 - x0, y1 - y0)

    # ── Status helpers ───────────────────────────────────────────────────────

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_dead(self) -> bool:
        return self.state == TrackState.DEAD

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def to_dict(self) -> Dict:
        cx, cy = self.center
        return {
            "track_id":     self.track_id,
            "class_name":   self.class_name,
            "state":        self.state.name,
            "center":       (round(cx, 1), round(cy, 1)),
            "bbox":         tuple(round(v, 1) for v in self.bbox),
            "conf":         round(self.conf, 3),
            "hits":         self.hits,
            "age":          self.age,
            "frames_lost":  self.frames_lost,
            "jersey":       self.jersey_number,
            "team_id":      self.team_id,
            "speed":        round(self.speed, 2),
        }

    def __repr__(self) -> str:
        cx, cy = self.center
        return (f"Track(id={self.track_id} "
                f"cls={self.class_name} "
                f"state={self.state.name} "
                f"center=({cx:.0f},{cy:.0f}) "
                f"hits={self.hits})")
                # =============================================================================
# MATCHING UTILITIES
# =============================================================================

def _compute_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _compute_iou_matrix(
    tracks:     List[Track],
    detections: List[Detection],
) -> np.ndarray:
    """Returns (n_tracks × n_detections) IoU matrix."""
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            mat[i, j] = _compute_iou(t.bbox, d.bbox)
    return mat


def _compute_distance_matrix(
    tracks:     List[Track],
    detections: List[Detection],
) -> np.ndarray:
    """Returns (n_tracks × n_detections) centroid-distance matrix."""
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            mat[i, j] = t.distance_to_point(d.center)
    return mat


def _hungarian_match(
    cost_matrix: np.ndarray,
    threshold:   float,
    maximize:    bool = False,
) -> List[Tuple[int, int]]:
    """
    Run Hungarian algorithm on cost_matrix.
    Returns list of (track_idx, det_idx) pairs passing threshold.
    """
    if cost_matrix.size == 0:
        return []

    if SCIPY_AVAILABLE:
        if maximize:
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = _greedy_assignment(cost_matrix, maximize)

    matches = []
    for r, c in zip(row_ind, col_ind):
        val = cost_matrix[r, c]
        if maximize:
            if val >= threshold:
                matches.append((r, c))
        else:
            if val <= threshold:
                matches.append((r, c))
    return matches


def _greedy_assignment(
    cost_matrix: np.ndarray,
    maximize:    bool = False,
) -> Tuple[List[int], List[int]]:
    """Fallback greedy assignment when scipy is unavailable."""
    matrix = cost_matrix.copy()
    if maximize:
        matrix = -matrix

    row_ind, col_ind = [], []
    used_rows, used_cols = set(), set()

    flat = np.argsort(matrix.ravel())
    for idx in flat:
        r, c = divmod(int(idx), matrix.shape[1])
        if r not in used_rows and c not in used_cols:
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)

    return row_ind, col_ind


def _distance_match(
    tracks:     List[Track],
    detections: List[Detection],
    threshold:  float = DEFAULT_DISTANCE_THRESH,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match tracks to detections by centroid distance.
    Returns (matches, unmatched_track_indices, unmatched_det_indices).
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    dist_mat = _compute_distance_matrix(tracks, detections)
    pairs    = _hungarian_match(dist_mat, threshold, maximize=False)

    matched_t = {p[0] for p in pairs}
    matched_d = {p[1] for p in pairs}

    unmatched_t = [i for i in range(len(tracks))     if i not in matched_t]
    unmatched_d = [j for j in range(len(detections)) if j not in matched_d]

    return pairs, unmatched_t, unmatched_d


def nms_detections(
    detections: List[Detection],
    iou_thresh: float = DEFAULT_NMS_IOU,
) -> List[Detection]:
    """
    Non-maximum suppression on a list of Detection objects.
    Keeps highest-confidence box when IoU overlap exceeds threshold.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: List[Detection] = []

    for det in detections:
        suppress = False
        for k in kept:
            if _compute_iou(det.bbox, k.bbox) > iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


# =============================================================================
# MULTI-OBJECT TRACKER  (IoU + distance cascade)
# =============================================================================

class MultiObjectTracker:
    """
    General-purpose multi-object tracker using:
      1. IoU-based Hungarian matching (primary)
      2. Distance-based matching (secondary, for lost tracks)
      3. Kalman-filter prediction between frames
    """

    def __init__(
        self,
        max_lost: int = DEFAULT_MAX_LOST,
        min_hits: int = DEFAULT_MIN_HITS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        distance_threshold: float = DEFAULT_DISTANCE_THRESH,
        classes_to_track: Optional[List[str]] = None,
        # Backward-compatibility / pipeline parameter aliases:
        max_disappeared: Optional[int] = None,
        max_distance: Optional[float] = None,
        **kwargs,
    ):
        # Resolve aliases if passed from pipeline.py
        if max_disappeared is not None:
            max_lost = max_disappeared
        if max_distance is not None:
            distance_threshold = max_distance

        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.classes_to_track = classes_to_track  # None = track everything

        self._tracks: Dict[int, Track] = {}
        self.frame_id: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Feed new detections; returns all CONFIRMED (+ TENTATIVE) tracks.
        """
        self.frame_id += 1

        # Filter by class if requested
        if self.classes_to_track:
            detections = [
                d for d in detections
                if d.class_name in self.classes_to_track
            ]

        detections = nms_detections(detections)

        active_tracks = [
            t for t in self._tracks.values()
            if t.state != TrackState.DEAD
        ]

        # ── Stage 1: IoU match confirmed tracks ─────────────────────────────
        confirmed = [t for t in active_tracks if t.is_confirmed]
        iou_mat   = _compute_iou_matrix(confirmed, detections)
        iou_pairs = _hungarian_match(
            iou_mat, self.iou_threshold, maximize=True
        )

        matched_t_idx = {p[0] for p in iou_pairs}
        matched_d_idx = {p[1] for p in iou_pairs}

        for ti, di in iou_pairs:
            confirmed[ti].mark_hit(detections[di])

        # ── Stage 2: Distance match remaining tracks ─────────────────────────
        unmatched_tracks = (
            [t for i, t in enumerate(confirmed) if i not in matched_t_idx]
            + [t for t in active_tracks if t.is_tentative or t.is_lost]
        )
        unmatched_dets = [
            d for j, d in enumerate(detections) if j not in matched_d_idx
        ]

        dist_pairs, still_unmatched_t, still_unmatched_d = _distance_match(
            unmatched_tracks, unmatched_dets, self.distance_threshold
        )

        matched_d_stage2 = {p[1] for p in dist_pairs}
        for ti, di in dist_pairs:
            unmatched_tracks[ti].mark_hit(unmatched_dets[di])

        # ── Stage 3: Mark missed ─────────────────────────────────────────────
        for ti in still_unmatched_t:
            unmatched_tracks[ti].mark_missed()

        # ── Stage 4: Create new tracks ───────────────────────────────────────
        new_det_indices = [
            j for j in still_unmatched_d
            if j < len(unmatched_dets)
        ]
        for di in new_det_indices:
            det = unmatched_dets[di]
            t   = Track(
                det,
                max_lost=self.max_lost,
                min_hits=self.min_hits,
            )
            self._tracks[t.track_id] = t

        # ── Stage 5: Remove dead tracks ──────────────────────────────────────
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if not t.is_dead
        }

        return self.active_tracks

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> List[Track]:
        return [
            t for t in self._tracks.values()
            if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
        ]

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [
            t for t in self._tracks.values()
            if t.state == TrackState.CONFIRMED
        ]

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def get_tracks_by_class(self, class_name: str) -> List[Track]:
        return [
            t for t in self.active_tracks
            if t.class_name == class_name
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self.frame_id = 0
        Track.reset_id_counter()


# Alias
PlayerTracker = MultiObjectTracker


# =============================================================================
# BALL TRACKER
# =============================================================================

class BallTracker:
    """
    Specialised single-object tracker for the basketball.
    Maintains arc history, shot detection, and possession context.
    """

    def __init__(
        self,
        max_lost:           int   = 15,
        trajectory_len:     int   = BALL_TRAJECTORY_LEN,
        speed_threshold:    float = 8.0,
        arc_min_height:     float = 30.0,
    ):
        self.max_lost        = max_lost
        self.trajectory_len  = trajectory_len
        self.speed_threshold = speed_threshold
        self.arc_min_height  = arc_min_height

        self._track:      Optional[Track] = None
        self._kf:         Optional[_KalmanFilter] = None
        self._lost_count: int = 0

        # Trajectory-gating params. Because the basketball is a small, fast
        # object its detection confidence is naturally low, so the detector is
        # run at a low conf and MANY false-positive boxes appear. We reject
        # those by preferring the candidate that best agrees with the Kalman
        # prediction (smooth motion) rather than the highest-confidence box.
        self.gate_radius:      float = 160.0   # max plausible px move / frame
        self.high_conf_accept: float = 0.35    # a strong box overrides gating

        self.trajectory: Deque[Tuple[float, float]] = deque(
            maxlen=trajectory_len
        )
        self.speed_history: Deque[float] = deque(maxlen=30)
        self.frame_id: int = 0

    # ── Update ───────────────────────────────────────────────────────────────

    def _select_candidate(
        self,
        ball_dets: List[Detection],
    ) -> Optional[Detection]:
        """
        Pick the most plausible ball among many (noisy) candidates.

        - No prediction yet: take highest confidence.
        - With a prediction: score each candidate by distance to the predicted
          position, blended with confidence. Candidates outside the gate
          radius are rejected unless their confidence is high enough to be
          trusted on its own (handles teleports after occlusion).
        """
        if not ball_dets:
            return None

        # No history → trust confidence
        if self._kf is None or len(self.trajectory) < 2:
            return max(ball_dets, key=lambda d: d.conf)

        px, py = self._kf.position
        vx, vy = self._kf.velocity
        # Predict where the ball should be this frame
        exp_x, exp_y = px + vx, py + vy

        best_det   = None
        best_score = -1.0
        fallback   = None  # highest-conf box outside the gate
        for d in ball_dets:
            cx, cy = d.center
            dist   = math.hypot(cx - exp_x, cy - exp_y)
            if dist <= self.gate_radius:
                # Lower distance and higher conf are both good.
                prox  = 1.0 - (dist / self.gate_radius)   # 0..1
                score = 0.7 * prox + 0.3 * d.conf
                if score > best_score:
                    best_score = score
                    best_det   = d
            else:
                if fallback is None or d.conf > fallback.conf:
                    fallback = d

        if best_det is not None:
            return best_det
        # Nothing inside the gate — only accept a far box if it's confident
        if fallback is not None and fallback.conf >= self.high_conf_accept:
            return fallback
        return None

    def update(
        self,
        detections: List[Detection],
    ) -> Optional[Tuple[float, float]]:
        """
        Feed ball detections (filtered to class 'ball').
        Returns current ball position or None.
        """
        self.frame_id += 1
        ball_dets = [d for d in detections if d.class_name == "ball"]

        best = self._select_candidate(ball_dets)

        if best is not None:
            cx, cy = best.center
            self._lost_count = 0

            if self._kf is None:
                self._kf = _KalmanFilter(cx, cy)
            else:
                self._kf.predict()
                self._kf.update(cx, cy)

            pos = self._kf.position
            self.trajectory.append(pos)

            # Speed
            if len(self.trajectory) >= 2:
                prev = self.trajectory[-2]
                spd  = math.hypot(
                    pos[0] - prev[0], pos[1] - prev[1]
                )
                self.speed_history.append(spd)

            if self._track is None:
                det_obj = Detection(
                    bbox=best.bbox,
                    conf=best.conf,
                    class_id=best.class_id,
                    class_name="ball",
                    frame_id=self.frame_id,
                )
                self._track = Track(
                    det_obj,
                    max_lost=self.max_lost,
                    min_hits=1,
                )
            else:
                self._track.mark_hit(best)

            return pos

        else:
            self._lost_count += 1
            if self._kf is not None:
                predicted = self._kf.predict()
                self.trajectory.append(predicted)
                if self._lost_count > self.max_lost:
                    self._kf    = None
                    self._track = None
                return predicted
            return None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        if self._kf is not None:
            return self._kf.position
        return None

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        if self._kf is not None:
            return self._kf.velocity
        return None

    @property
    def speed(self) -> float:
        if self.speed_history:
            return float(np.mean(list(self.speed_history)[-5:]))
        return 0.0

    @property
    def is_tracked(self) -> bool:
        return self._kf is not None and self._lost_count <= self.max_lost

    @property
    def is_in_flight(self) -> bool:
        return self.is_tracked and self.speed > self.speed_threshold

    # ── Arc / Shot detection ─────────────────────────────────────────────────

    def detect_arc(self) -> bool:
        """
        Returns True if trajectory shows upward-then-downward arc
        consistent with a shot attempt.
        """
        traj = list(self.trajectory)
        if len(traj) < 10:
            return False

        ys = [p[1] for p in traj[-20:]]
        if len(ys) < 6:
            return False

        mid  = len(ys) // 2
        top  = min(ys)
        base = max(ys)

        # In image coords y increases downward, so arc peak = min y
        peak_idx = ys.index(top)
        arc_height = base - top

        return (
            arc_height >= self.arc_min_height
            and 1 <= peak_idx <= len(ys) - 2
        )

    def get_trajectory_array(self) -> np.ndarray:
        """Returns trajectory as (N, 2) float32 array."""
        return np.array(list(self.trajectory), dtype=np.float32)

    def reset(self) -> None:
        self._track      = None
        self._kf         = None
        self._lost_count = 0
        self.trajectory.clear()
        self.speed_history.clear()
        self.frame_id = 0
       # =============================================================================
# RIM TRACKER
# =============================================================================

class RimTracker:
    """
    Tracks the basketball rim — typically a fixed object.
    Averages detections over time for a stable position estimate.
    Provides scoring-zone geometry.
    """

    def __init__(
        self,
        stability_frames: int   = RIM_STABILITY_FRAMES,
        score_zone_scale: float = 1.8,
    ):
        self.stability_frames = stability_frames
        self.score_zone_scale = score_zone_scale

        self._detections: Deque[Detection] = deque(maxlen=stability_frames)
        self._stable_bbox: Optional[Tuple[float, float, float, float]] = None
        self.frame_id: int = 0

    def update(self, detections: List[Detection]) -> Optional[Tuple]:
        """
        Feed rim detections (filtered to class 'rim').
        Returns stable rim bbox or None.
        """
        self.frame_id += 1
        rim_dets = [d for d in detections if d.class_name == "rim"]

        if rim_dets:
            best = max(rim_dets, key=lambda d: d.conf)
            self._detections.append(best)

        if len(self._detections) >= max(1, self.stability_frames // 2):
            xs1 = [d.x1 for d in self._detections]
            ys1 = [d.y1 for d in self._detections]
            xs2 = [d.x2 for d in self._detections]
            ys2 = [d.y2 for d in self._detections]
            self._stable_bbox = (
                float(np.median(xs1)),
                float(np.median(ys1)),
                float(np.median(xs2)),
                float(np.median(ys2)),
            )

        return self._stable_bbox

    # ── Geometry ─────────────────────────────────────────────────────────────

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self._stable_bbox

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self._stable_bbox is None:
            return None
        x1, y1, x2, y2 = self._stable_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> Optional[float]:
        if self._stable_bbox is None:
            return None
        return self._stable_bbox[2] - self._stable_bbox[0]

    @property
    def height(self) -> Optional[float]:
        if self._stable_bbox is None:
            return None
        return self._stable_bbox[3] - self._stable_bbox[1]

    @property
    def is_stable(self) -> bool:
        return self._stable_bbox is not None

    def scoring_zone(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns an expanded bbox representing the scoring zone
        above and around the rim.
        """
        if self._stable_bbox is None:
            return None

        x1, y1, x2, y2 = self._stable_bbox
        cx = (x1 + x2) / 2.0
        w  = (x2 - x1) * self.score_zone_scale
        h  = (y2 - y1) * self.score_zone_scale * 2.0

        return (
            cx - w / 2.0,
            y1 - h,          # extend above the rim
            cx + w / 2.0,
            y2,
        )

    def ball_near_rim(
        self,
        ball_pos: Tuple[float, float],
        scale:    float = 2.5,
    ) -> bool:
        """True if ball center is within scale * rim_width of rim center."""
        if self.center is None or self.width is None:
            return False
        dist = math.hypot(
            ball_pos[0] - self.center[0],
            ball_pos[1] - self.center[1],
        )
        return dist < self.width * scale

    def ball_above_rim(self, ball_pos: Tuple[float, float]) -> bool:
        """True if ball y is above (less than) rim top y."""
        if self._stable_bbox is None:
            return False
        return ball_pos[1] < self._stable_bbox[1]

    def ball_through_rim(
        self,
        ball_pos:  Tuple[float, float],
        prev_pos:  Optional[Tuple[float, float]] = None,
    ) -> bool:
        """
        Heuristic: ball passed through rim plane from above.
        Uses current + previous position to detect downward crossing.
        """
        if self._stable_bbox is None:
            return False

        x1, y1, x2, y2 = self._stable_bbox
        rim_y  = (y1 + y2) / 2.0
        rim_x1 = x1
        rim_x2 = x2

        bx, by = ball_pos
        in_x   = rim_x1 <= bx <= rim_x2

        if prev_pos is not None:
            _, prev_y = prev_pos
            crossed   = prev_y < rim_y <= by
            return in_x and crossed

        return in_x and abs(by - rim_y) < (y2 - y1)

    def reset(self) -> None:
        self._detections.clear()
        self._stable_bbox = None
        self.frame_id     = 0


# =============================================================================
# VISUALIZER
# =============================================================================

class Visualizer:
    """
    Renders tracks, trajectories, zones, and overlays onto frames.
    All drawing uses integer coordinates for OpenCV compatibility.
    """

    # Colour palette  (BGR)
    COLOURS = {
        "player":      (0,   200,   0),
        "ball":        (0,   165, 255),
        "rim":         (0,     0, 255),
        "referee":     (255, 255,   0),
        "coach":       (200, 200, 200),
        "team_0":      (255,  80,  80),
        "team_1":      (80,   80, 255),
        "trajectory":  (0,   255, 255),
        "zone":        (180,   0, 180),
        "possession":  (0,   255,   0),
        "scoring":     (0,   255, 128),
        "text_bg":     (0,     0,   0),
        "white":       (255, 255, 255),
        "confirmed":   (0,   255,   0),
        "tentative":   (0,   200, 200),
        "lost":        (100, 100, 100),
    }

    def __init__(
        self,
        show_trajectories: bool = True,
        show_ids:          bool = True,
        show_conf:         bool = False,
        show_velocity:     bool = False,
        trajectory_len:    int  = 30,
        thickness:         int  = 2,
        font_scale:        float = 0.55,
    ):
        self.show_trajectories = show_trajectories
        self.show_ids          = show_ids
        self.show_conf         = show_conf
        self.show_velocity     = show_velocity
        self.trajectory_len    = trajectory_len
        self.thickness         = thickness
        self.font_scale        = font_scale
        self.font              = cv2.FONT_HERSHEY_SIMPLEX

    # ── Colour helpers ───────────────────────────────────────────────────────

    def _track_colour(self, track: Track) -> Tuple[int, int, int]:
        if track.class_name == "player":
            if track.team_id == 0:
                return self.COLOURS["team_0"]
            elif track.team_id == 1:
                return self.COLOURS["team_1"]
        return self.COLOURS.get(track.class_name, self.COLOURS["white"])

    def _state_colour(self, track: Track) -> Tuple[int, int, int]:
        if track.is_confirmed:
            return self.COLOURS["confirmed"]
        if track.is_tentative:
            return self.COLOURS["tentative"]
        return self.COLOURS["lost"]

    # ── Single track ─────────────────────────────────────────────────────────

    def draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        possessor_id: Optional[int] = None,
    ) -> None:
        """Draw bounding box, label, and optional velocity arrow."""
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        colour = self._track_colour(track)

        # Highlight possessor
        if possessor_id is not None and track.track_id == possessor_id:
            colour = self.COLOURS["possession"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.thickness)

        # Label
        if self.show_ids:
            parts = [f"#{track.track_id}"]
            if track.jersey_number:
                parts.append(f"J{track.jersey_number}")
            if self.show_conf:
                parts.append(f"{track.conf:.2f}")
            label = " ".join(parts)
            self._draw_label(frame, label, x1, y1, colour)

        # Velocity arrow
        if self.show_velocity and track.is_moving:
            cx, cy   = int(track.center[0]), int(track.center[1])
            vx, vy   = track.velocity
            tip_x    = int(cx + vx * 4)
            tip_y    = int(cy + vy * 4)
            cv2.arrowedLine(
                frame,
                (cx, cy),
                (tip_x, tip_y),
                colour,
                self.thickness,
                tipLength=0.3,
            )

    def draw_trajectory(
        self,
        frame:  np.ndarray,
        track:  Track,
        colour: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw trajectory polyline for a track."""
        if not self.show_trajectories:
            return

        traj   = list(track.trajectory)[-self.trajectory_len:]
        colour = colour or self.COLOURS["trajectory"]

        for i in range(1, len(traj)):
            p1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
            p2 = (int(traj[i][0]),     int(traj[i][1]))
            alpha = i / len(traj)
            c = tuple(int(v * alpha) for v in colour)
            cv2.line(frame, p1, p2, c, 1)

    def draw_ball(
        self,
        frame:    np.ndarray,
        position: Tuple[float, float],
        radius:   int = 12,
        in_flight: bool = False,
    ) -> None:
        """Draw ball as circle with optional flight indicator."""
        cx, cy = int(position[0]), int(position[1])
        colour = self.COLOURS["ball"]

        cv2.circle(frame, (cx, cy), radius, colour, -1)
        cv2.circle(frame, (cx, cy), radius + 2, colour, 1)

        if in_flight:
            cv2.circle(frame, (cx, cy), radius + 6,
                       self.COLOURS["scoring"], 1)

    def draw_ball_trajectory(
        self,
        frame:    np.ndarray,
        tracker:  BallTracker,
    ) -> None:
        """Draw ball arc trajectory."""
        traj = list(tracker.trajectory)[-self.trajectory_len:]
        if len(traj) < 2:
            return

        for i in range(1, len(traj)):
            p1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
            p2 = (int(traj[i][0]),     int(traj[i][1]))
            alpha = i / len(traj)
            r = int(255 * alpha)
            g = int(165 * (1 - alpha))
            cv2.line(frame, p1, p2, (0, g, r), 2)

    def draw_rim(
        self,
        frame:  np.ndarray,
        tracker: RimTracker,
    ) -> None:
        """Draw rim bounding box and scoring zone."""
        if not tracker.is_stable:
            return

        x1, y1, x2, y2 = (int(v) for v in tracker.bbox)
        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            self.COLOURS["rim"],
            self.thickness,
        )

        sz = tracker.scoring_zone()
        if sz:
            sx1, sy1, sx2, sy2 = (int(v) for v in sz)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (sx1, sy1), (sx2, sy2),
                self.COLOURS["scoring"],
                -1,
            )
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(
                frame,
                (sx1, sy1), (sx2, sy2),
                self.COLOURS["scoring"],
                1,
            )

    # ── Labels ───────────────────────────────────────────────────────────────

    def _draw_label(
        self,
        frame:  np.ndarray,
        text:   str,
        x:      int,
        y:      int,
        colour: Tuple[int, int, int],
    ) -> None:
        (tw, th), _ = cv2.getTextSize(
            text, self.font, self.font_scale, 1
        )
        ty = max(y - 4, th + 4)
        cv2.rectangle(
            frame,
            (x, ty - th - 4),
            (x + tw + 4, ty),
            self.COLOURS["text_bg"],
            -1,
        )
        cv2.putText(
            frame, text,
            (x + 2, ty - 2),
            self.font, self.font_scale,
            colour, 1, cv2.LINE_AA,
        )

    def draw_score_event(
        self,
        frame:  np.ndarray,
        text:   str = "SCORE!",
        colour: Tuple[int, int, int] = (0, 255, 128),
    ) -> None:
        """Flash a score event banner."""
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, self.font, 2.0, 3)
        tx = (w - tw) // 2
        ty = h // 3
        cv2.putText(
            frame, text,
            (tx, ty),
            self.font, 2.0,
            colour, 3, cv2.LINE_AA,
        )

    # ── Zones ────────────────────────────────────────────────────────────────

    def draw_zone(
        self,
        frame:      np.ndarray,
        zone_bbox:  Tuple[float, float, float, float],
        label:      str = "",
        colour:     Tuple[int, int, int] = (180, 0, 180),
        alpha:      float = 0.12,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in zone_bbox)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        if label:
            self._draw_label(frame, label, x1, y1, colour)

    def draw_zones(
        self,
        frame:  np.ndarray,
        zones:  Dict[str, Tuple[float, float, float, float]],
    ) -> None:
        for name, bbox in zones.items():
            self.draw_zone(frame, bbox, label=name)

    # ── Full scene ───────────────────────────────────────────────────────────

    def draw_all(
        self,
        frame:           np.ndarray,
        tracks:          List[Track],
        ball_tracker:    Optional[BallTracker]  = None,
        rim_tracker:     Optional[RimTracker]   = None,
        possessor_id:    Optional[int]          = None,
        zones:           Optional[Dict]         = None,
        score_flash:     bool                   = False,
        fps:             Optional[float]        = None,
        frame_id:        Optional[int]          = None,
    ) -> np.ndarray:
        """
        One-call render: draws everything onto frame in-place.
        Returns the annotated frame.
        """
        # Zones
        if zones:
            self.draw_zones(frame, zones)

        # Rim
        if rim_tracker is not None:
            self.draw_rim(frame, rim_tracker)

        # Ball trajectory
        if ball_tracker is not None:
            self.draw_ball_trajectory(frame, ball_tracker)

        # Tracks
        for track in tracks:
            self.draw_trajectory(frame, track)
            self.draw_track(frame, track, possessor_id)

        # Ball
        if ball_tracker is not None and ball_tracker.position is not None:
            self.draw_ball(
                frame,
                ball_tracker.position,
                in_flight=ball_tracker.is_in_flight,
            )

        # Score flash
        if score_flash:
            self.draw_score_event(frame)

        # HUD
        self._draw_hud(frame, fps=fps, frame_id=frame_id)

        return frame

    def _draw_hud(
        self,
        frame:    np.ndarray,
        fps:      Optional[float] = None,
        frame_id: Optional[int]   = None,
    ) -> None:
        lines = []
        if fps      is not None: lines.append(f"FPS: {fps:.1f}")
        if frame_id is not None: lines.append(f"Frame: {frame_id}")

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (10, 24 + i * 22),
                self.font, 0.6,
                self.COLOURS["white"], 1, cv2.LINE_AA,
            )
            # =============================================================================
# FPS COUNTER
# =============================================================================

class FPSCounter:
    """Rolling-window FPS measurement."""

    def __init__(self, window: int = 30):
        self.window     = window
        self._times:    Deque[float] = deque(maxlen=window)
        self._last_tick: Optional[float] = None

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._times.append(now - self._last_tick)
        self._last_tick = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    def reset(self) -> None:
        self._times.clear()
        self._last_tick = None


# =============================================================================
# ZONE DETECTOR
# =============================================================================

class ZoneDetector:
    """
    Maps positions to named court zones.
    Zones are defined as (x1, y1, x2, y2) rectangles in pixel space.
    """

    def __init__(
        self,
        frame_width:  int = 1280,
        frame_height: int = 720,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.zones: Dict[str, Tuple[float, float, float, float]] = {}
        self._setup_default_zones()

    def _setup_default_zones(self) -> None:
        w, h = self.frame_width, self.frame_height
        self.zones = {
            "paint_left":    (0,         h * 0.4, w * 0.2,  h),
            "paint_right":   (w * 0.8,   h * 0.4, w,        h),
            "three_point":   (w * 0.1,   0,       w * 0.9,  h * 0.85),
            "mid_range":     (w * 0.2,   h * 0.2, w * 0.8,  h * 0.75),
            "backcourt":     (0,         0,       w,        h * 0.15),
            "full_court":    (0,         0,       w,        h),
        }

    def add_zone(
        self,
        name: str,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        self.zones[name] = bbox

    def get_zone(
        self,
        position: Tuple[float, float],
    ) -> Optional[str]:
        """Returns name of innermost zone containing position."""
        x, y = position
        matched = []
        for name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                matched.append((area, name))

        if not matched:
            return None
        matched.sort(key=lambda t: t[0])
        return matched[0][1]

    def point_in_zone(
        self,
        position: Tuple[float, float],
        zone_name: str,
    ) -> bool:
        if zone_name not in self.zones:
            return False
        x, y = position
        x1, y1, x2, y2 = self.zones[zone_name]
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_all_zones_for_point(
        self,
        position: Tuple[float, float],
    ) -> List[str]:
        return [
            name for name, (x1, y1, x2, y2) in self.zones.items()
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2
        ]

    def resize(self, new_width: int, new_height: int) -> None:
        sx = new_width  / self.frame_width
        sy = new_height / self.frame_height
        self.zones = {
            name: (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            for name, (x1, y1, x2, y2) in self.zones.items()
        }
        self.frame_width  = new_width
        self.frame_height = new_height


# =============================================================================
# TEAM CLASSIFIER
# =============================================================================

class TeamClassifier:
    """
    Classifies player tracks into teams.
    Default: left half = team 0, right half = team 1.
    Extend with colour histogram or deep embeddings for production.
    """

    def __init__(
        self,
        n_teams:    int   = 2,
        side_split: float = 0.5,
    ):
        self.n_teams    = n_teams
        self.side_split = side_split

    def classify(
        self,
        track:       Track,
        frame:       Optional[np.ndarray] = None,
        frame_width: int = 1280,
    ) -> int:
        """Returns team_id (0 or 1)."""
        cx, _ = track.center
        return 0 if cx < frame_width * self.side_split else 1

    def classify_all(
        self,
        tracks:      List[Track],
        frame:       Optional[np.ndarray] = None,
        frame_width: int = 1280,
    ) -> None:
        """Classify all player tracks in-place."""
        for track in tracks:
            if track.class_name == "player":
                track.team_id = self.classify(track, frame, frame_width)

    def colour_histogram_classify(
        self,
        track: Track,
        frame: np.ndarray,
        reference_hists: Optional[List[np.ndarray]] = None,
    ) -> int:
        """
        Classify by comparing jersey colour histogram to reference histograms.
        reference_hists: list of [hist_team0, hist_team1]
        Falls back to side_split if references not provided.
        """
        if reference_hists is None or frame is None:
            return self.classify(track, frame, frame.shape[1])

        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self.classify(track)

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        scores = [
            cv2.compareHist(hist, ref, cv2.HISTCMP_CORREL)
            for ref in reference_hists
        ]
        return int(np.argmax(scores))


# =============================================================================
# POSSESSION TRACKER
# =============================================================================

class PossessionTracker:
    """
    Estimates which player has possession of the ball
    based on proximity of ball to player tracks.
    """

    def __init__(
        self,
        proximity_threshold: float = DEFAULT_PROXIMITY_THRESH,
    ):
        self.proximity_threshold   = proximity_threshold
        self.current_possessor_id: Optional[int] = None
        self.possession_history:   List[Dict]    = []
        self.frame_count:          int           = 0

    def update(
        self,
        ball_position:  Optional[Tuple[float, float]],
        player_tracks:  List[Track],
        frame_id:       int = 0,
    ) -> Optional[int]:
        """Returns track_id of player with possession, or None."""
        self.frame_count += 1

        if ball_position is None or not player_tracks:
            return self.current_possessor_id

        nearest = min(
            player_tracks,
            key=lambda t: t.distance_to_point(ball_position),
        )
        dist = nearest.distance_to_point(ball_position)

        if dist <= self.proximity_threshold:
            new_id = nearest.track_id
            if new_id != self.current_possessor_id:
                self.possession_history.append({
                    "frame_id": frame_id,
                    "track_id": new_id,
                    "jersey":   nearest.jersey_number,
                    "team_id":  nearest.team_id,
                    "distance": round(dist, 2),
                })
                self.current_possessor_id = new_id
        else:
            self.current_possessor_id = None

        return self.current_possessor_id

    def get_possession_summary(self) -> Dict:
        """Returns possession change counts by track_id."""
        counts: Dict[int, int] = defaultdict(int)
        for entry in self.possession_history:
            counts[entry["track_id"]] += 1
        return dict(counts)

    def get_team_possession(self) -> Dict[int, int]:
        """Returns possession change counts by team_id."""
        counts: Dict[int, int] = defaultdict(int)
        for entry in self.possession_history:
            if entry["team_id"] is not None:
                counts[entry["team_id"]] += 1
        return dict(counts)

    def reset(self) -> None:
        self.current_possessor_id = None
        self.possession_history.clear()
        self.frame_count = 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TrackState",
    "ObjectClass",
    "EventType",
    # Data classes
    "Detection",
    "Track",
    # Trackers
    "MultiObjectTracker",
    "PlayerTracker",
    "BallTracker",
    "RimTracker",
    # Utilities
    "Visualizer",
    "FPSCounter",
    "ZoneDetector",
    "TeamClassifier",
    "PossessionTracker",
    # Functions
    "_compute_iou",
    "_compute_iou_matrix",
    "_compute_distance_matrix",
    "_hungarian_match",
    "_distance_match",
    "nms_detections",
    # Internal
    "_KalmanFilter",
]
