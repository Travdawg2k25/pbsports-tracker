"""
dead_ball_detector.py
─────────────────────
Detects "dead ball" periods in basketball video — times when the ball
is out of play (out of bounds, free throw reset, timeout, etc.).

Dead ball periods are excluded from highlight reels and stat tracking.

Classes
───────
DeadBallState       — Enum for current ball state
DeadBallEvent       — Dataclass representing a detected dead ball segment
DeadBallDetector    — Main detector class
DeadBallFilter      — Post-processor to merge/filter dead ball segments
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Enums & Dataclasses
# ──────────────────────────────────────────────

class DeadBallState(Enum):
    LIVE        = auto()   # Ball is in active play
    DEAD        = auto()   # Ball is out of play
    UNCERTAIN   = auto()   # Transitioning / not enough data


@dataclass
class DeadBallEvent:
    """Represents a single dead ball segment."""
    start_frame:    int
    end_frame:      int
    start_time_sec: float
    end_time_sec:   float
    reason:         str = "unknown"
    confidence:     float = 0.0

    @property
    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time_sec - self.start_time_sec)

    def __repr__(self) -> str:
        return (
            f"DeadBallEvent(frames={self.start_frame}-{self.end_frame}, "
            f"duration={self.duration_sec:.2f}s, reason='{self.reason}', "
            f"conf={self.confidence:.2f})"
        )


@dataclass
class _BallHistory:
    """Internal rolling history of ball detections."""
    positions:      deque = field(default_factory=lambda: deque(maxlen=90))
    confidences:    deque = field(default_factory=lambda: deque(maxlen=90))
    detected_flags: deque = field(default_factory=lambda: deque(maxlen=90))
    velocities:     deque = field(default_factory=lambda: deque(maxlen=90))


# ──────────────────────────────────────────────
# Main Detector
# ──────────────────────────────────────────────

class DeadBallDetector:
    """
    Detects dead ball states from frame-by-frame ball tracking data.

    Parameters
    ──────────
    fps                     : Video frames per second
    missing_frames_threshold: Consecutive frames without ball → dead ball
    stationary_threshold_px : Max pixel movement to consider ball stationary
    stationary_frames       : Frames ball must be stationary → dead ball
    out_of_bounds_margin    : Pixel margin outside frame edge → out of bounds
    min_dead_duration_sec   : Minimum dead ball duration to record
    confidence_threshold    : Minimum detection confidence to trust position
    """

    def __init__(
        self,
        fps:                        float = 30.0,
        missing_frames_threshold:   int   = 15,
        stationary_threshold_px:    float = 8.0,
        stationary_frames:          int   = 45,
        out_of_bounds_margin:       int   = 20,
        min_dead_duration_sec:      float = 0.5,
        confidence_threshold:       float = 0.35,
    ):
        self.fps                        = max(fps, 1.0)
        self.missing_frames_threshold   = missing_frames_threshold
        self.stationary_threshold_px    = stationary_threshold_px
        self.stationary_frames          = stationary_frames
        self.out_of_bounds_margin       = out_of_bounds_margin
        self.min_dead_duration_sec      = min_dead_duration_sec
        self.confidence_threshold       = confidence_threshold

        # State
        self._history           = _BallHistory()
        self._state             = DeadBallState.LIVE
        self._dead_start_frame: Optional[int]   = None
        self._dead_start_time:  Optional[float] = None
        self._dead_reason:      str             = "unknown"
        self._frame_index:      int             = 0
        self._missing_count:    int             = 0
        self._stationary_count: int             = 0

        # Output
        self.dead_ball_events:  List[DeadBallEvent] = []
        self._last_position:    Optional[Tuple[float, float]] = None

    # ── Public API ─────────────────────────────

    def reset(self) -> None:
        """Reset detector for a new video."""
        self._history           = _BallHistory()
        self._state             = DeadBallState.LIVE
        self._dead_start_frame  = None
        self._dead_start_time   = None
        self._dead_reason       = "unknown"
        self._frame_index       = 0
        self._missing_count     = 0
        self._stationary_count  = 0
        self.dead_ball_events   = []
        self._last_position     = None

    def update(
        self,
        ball_detected:  bool,
        ball_position:  Optional[Tuple[float, float]] = None,
        ball_confidence: float = 0.0,
        frame_shape:    Optional[Tuple[int, int]] = None,
    ) -> DeadBallState:
        """
        Call once per frame with current ball detection result.

        Parameters
        ──────────
        ball_detected   : Whether the ball was detected this frame
        ball_position   : (cx, cy) center of ball in pixels, or None
        ball_confidence : Detection confidence score
        frame_shape     : (height, width) of the frame, for OOB detection

        Returns
        ───────
        Current DeadBallState
        """
        current_time = self._frame_index / self.fps

        # Record history
        self._history.detected_flags.append(ball_detected)
        self._history.confidences.append(ball_confidence)

        if ball_detected and ball_position is not None:
            self._history.positions.append(ball_position)

            # Compute velocity
            if self._last_position is not None:
                dx = ball_position[0] - self._last_position[0]
                dy = ball_position[1] - self._last_position[1]
                velocity = float(np.sqrt(dx * dx + dy * dy))
            else:
                velocity = 0.0

            self._history.velocities.append(velocity)
            self._last_position = ball_position
        else:
            if self._last_position is not None:
                self._history.positions.append(self._last_position)
            self._history.velocities.append(0.0)

        # ── Run detection logic ─────────────────
        dead, reason, confidence = self._check_dead_ball(
            ball_detected, ball_position, ball_confidence, frame_shape
        )

        # ── State machine ───────────────────────
        if dead:
            if self._state == DeadBallState.LIVE:
                self._state             = DeadBallState.DEAD
                self._dead_start_frame  = self._frame_index
                self._dead_start_time   = current_time
                self._dead_reason       = reason
        else:
            if self._state == DeadBallState.DEAD:
                # Ball came back — close the event
                duration = current_time - (self._dead_start_time or current_time)
                if duration >= self.min_dead_duration_sec:
                    evt = DeadBallEvent(
                        start_frame     = self._dead_start_frame or self._frame_index,
                        end_frame       = self._frame_index,
                        start_time_sec  = self._dead_start_time or current_time,
                        end_time_sec    = current_time,
                        reason          = self._dead_reason,
                        confidence      = confidence,
                    )
                    self.dead_ball_events.append(evt)
                self._state             = DeadBallState.LIVE
                self._dead_start_frame  = None
                self._dead_start_time   = None

        self._frame_index += 1
        return self._state

    def finalize(self) -> List[DeadBallEvent]:
        """
        Call at end of video to close any open dead ball segment.
        Returns all detected dead ball events.
        """
        if self._state == DeadBallState.DEAD and self._dead_start_frame is not None:
            current_time = self._frame_index / self.fps
            duration = current_time - (self._dead_start_time or current_time)
            if duration >= self.min_dead_duration_sec:
                evt = DeadBallEvent(
                    start_frame     = self._dead_start_frame,
                    end_frame       = self._frame_index,
                    start_time_sec  = self._dead_start_time or 0.0,
                    end_time_sec    = current_time,
                    reason          = self._dead_reason,
                    confidence      = 0.5,
                )
                self.dead_ball_events.append(evt)

        return self.dead_ball_events

    @property
    def current_state(self) -> DeadBallState:
        return self._state

    @property
    def is_dead_ball(self) -> bool:
        return self._state == DeadBallState.DEAD

    @property
    def current_frame(self) -> int:
        return self._frame_index

    # ── Internal Logic ──────────────────────────

    def _check_dead_ball(
        self,
        ball_detected:  bool,
        ball_position:  Optional[Tuple[float, float]],
        ball_confidence: float,
        frame_shape:    Optional[Tuple[int, int]],
    ) -> Tuple[bool, str, float]:
        """
        Returns (is_dead, reason, confidence).
        Checks multiple dead ball conditions in priority order.
        """

        # 1. Ball missing for too many consecutive frames
        if not ball_detected:
            self._missing_count += 1
        else:
            self._missing_count = 0

        if self._missing_count >= self.missing_frames_threshold:
            return True, "ball_missing", min(1.0, self._missing_count / (self.missing_frames_threshold * 2))

        # 2. Ball out of bounds
        if ball_detected and ball_position is not None and frame_shape is not None:
            h, w = frame_shape[0], frame_shape[1]
            cx, cy = ball_position
            margin = self.out_of_bounds_margin
            if cx < -margin or cx > w + margin or cy < -margin or cy > h + margin:
                return True, "out_of_bounds", 0.9

        # 3. Ball stationary for too long (e.g. free throw reset, timeout)
        if ball_detected and ball_confidence >= self.confidence_threshold:
            recent_velocities = list(self._history.velocities)[-self.stationary_frames:]
            if len(recent_velocities) >= self.stationary_frames:
                avg_velocity = float(np.mean(recent_velocities))
                if avg_velocity < self.stationary_threshold_px:
                    self._stationary_count += 1
                    if self._stationary_count >= self.stationary_frames:
                        conf = min(1.0, self._stationary_count / (self.stationary_frames * 2))
                        return True, "ball_stationary", conf
                else:
                    self._stationary_count = 0
            else:
                self._stationary_count = 0
        else:
            self._stationary_count = 0

        # 4. Low confidence detections only (uncertain / occluded)
        recent_conf = list(self._history.confidences)[-10:]
        if len(recent_conf) >= 10:
            avg_conf = float(np.mean(recent_conf))
            if avg_conf < self.confidence_threshold * 0.5:
                return True, "low_confidence", 0.6

        return False, "live", 0.0

    def get_live_frame_mask(self, total_frames: int) -> np.ndarray:
        """
        Returns a boolean numpy array of shape (total_frames,).
        True  → frame is LIVE (include in highlights / stats)
        False → frame is DEAD (exclude)
        """
        mask = np.ones(total_frames, dtype=bool)
        for evt in self.dead_ball_events:
            s = max(0, evt.start_frame)
            e = min(total_frames, evt.end_frame)
            mask[s:e] = False
        return mask

    def get_dead_time_ranges(self) -> List[Tuple[float, float]]:
        """Returns list of (start_sec, end_sec) for all dead ball periods."""
        return [(e.start_time_sec, e.end_time_sec) for e in self.dead_ball_events]

    def summary(self) -> dict:
        """Returns a summary dictionary of dead ball statistics."""
        total_dead_frames = sum(e.duration_frames for e in self.dead_ball_events)
        total_dead_sec    = sum(e.duration_sec    for e in self.dead_ball_events)
        reasons: dict     = {}
        for e in self.dead_ball_events:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1

        return {
            "total_dead_ball_events":   len(self.dead_ball_events),
            "total_dead_frames":        total_dead_frames,
            "total_dead_seconds":       round(total_dead_sec, 2),
            "dead_ball_reasons":        reasons,
            "events":                   [
                {
                    "start_frame":  e.start_frame,
                    "end_frame":    e.end_frame,
                    "duration_sec": round(e.duration_sec, 2),
                    "reason":       e.reason,
                    "confidence":   round(e.confidence, 3),
                }
                for e in self.dead_ball_events
            ],
        }


# ──────────────────────────────────────────────
# Post-Processor
# ──────────────────────────────────────────────

class DeadBallFilter:
    """
    Post-processes a list of DeadBallEvents to:
    - Merge events that are close together
    - Remove events shorter than a minimum duration
    - Pad events slightly for cleaner cuts
    """

    def __init__(
        self,
        merge_gap_sec:      float = 1.0,
        min_duration_sec:   float = 0.3,
        pad_sec:            float = 0.1,
        fps:                float = 30.0,
    ):
        self.merge_gap_sec    = merge_gap_sec
        self.min_duration_sec = min_duration_sec
        self.pad_sec          = pad_sec
        self.fps              = fps

    def filter(self, events: List[DeadBallEvent]) -> List[DeadBallEvent]:
        """Apply merge, filter, and pad operations."""
        if not events:
            return []

        # Sort by start time
        sorted_events = sorted(events, key=lambda e: e.start_time_sec)

        # Merge nearby events
        merged = self._merge(sorted_events)

        # Remove short events
        filtered = [e for e in merged if e.duration_sec >= self.min_duration_sec]

        # Pad events
        padded = self._pad(filtered)

        return padded

    def _merge(self, events: List[DeadBallEvent]) -> List[DeadBallEvent]:
        if not events:
            return []

        result = [events[0]]
        for evt in events[1:]:
            last = result[-1]
            gap = evt.start_time_sec - last.end_time_sec
            if gap <= self.merge_gap_sec:
                # Merge into last
                merged = DeadBallEvent(
                    start_frame     = last.start_frame,
                    end_frame       = evt.end_frame,
                    start_time_sec  = last.start_time_sec,
                    end_time_sec    = evt.end_time_sec,
                    reason          = last.reason,
                    confidence      = max(last.confidence, evt.confidence),
                )
                result[-1] = merged
            else:
                result.append(evt)

        return result

    def _pad(self, events: List[DeadBallEvent]) -> List[DeadBallEvent]:
        pad_frames = int(self.pad_sec * self.fps)
        padded = []
        for e in events:
            padded.append(DeadBallEvent(
                start_frame     = max(0, e.start_frame - pad_frames),
                end_frame       = e.end_frame + pad_frames,
                start_time_sec  = max(0.0, e.start_time_sec - self.pad_sec),
                end_time_sec    = e.end_time_sec + self.pad_sec,
                reason          = e.reason,
                confidence      = e.confidence,
            ))
        return padded


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────

def build_dead_ball_detector(fps: float = 30.0) -> DeadBallDetector:
    """
    Returns a DeadBallDetector with sensible defaults for basketball.
    Adjust thresholds here for your specific camera / court setup.
    """
    return DeadBallDetector(
        fps                         = fps,
        missing_frames_threshold    = int(fps * 0.75),   # ~0.75 sec missing
        stationary_threshold_px     = 6.0,
        stationary_frames           = int(fps * 2.0),    # 2 sec stationary
        out_of_bounds_margin        = 30,
        min_dead_duration_sec       = 0.5,
        confidence_threshold        = 0.35,
    )


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DeadBallDetector...")

    detector = build_dead_ball_detector(fps=30.0)

    # Simulate 300 frames: live → dead (missing) → live → dead (stationary)
    for i in range(300):
        if 50 <= i < 80:
            # Ball missing for 30 frames → dead
            state = detector.update(False, None, 0.0, (720, 1280))
        elif 150 <= i < 220:
            # Ball stationary for 70 frames → dead
            state = detector.update(True, (640.0, 360.0), 0.8, (720, 1280))
        else:
            # Ball moving normally
            cx = 300.0 + i * 2.0
            cy = 300.0 + np.sin(i * 0.1) * 50.0
            state = detector.update(True, (cx, cy), 0.85, (720, 1280))

    events = detector.finalize()
    print(f"\nDetected {len(events)} dead ball event(s):")
    for e in events:
        print(f"  {e}")

    print("\nSummary:")
    import json
    print(json.dumps(detector.summary(), indent=2))

    filt = DeadBallFilter(fps=30.0)
    filtered = filt.filter(events)
    print(f"\nAfter filtering: {len(filtered)} event(s)")
    for e in filtered:
        print(f"  {e}")

    print("\n✅ DeadBallDetector self-test complete.")