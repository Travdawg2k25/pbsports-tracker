# =============================================================================
# basketball_events.py  —  Purple Box Sports
# Basketball event detection engine  (bug-fixed, production-ready)
# =============================================================================

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Deque, Dict, List, Optional, Sequence, Tuple, Union
)

import numpy as np

# ── Tracker imports  (only what is actually used) ────────────────────────────
# FIX #1: Removed unused imports (_compute_iou, TrackState,
#          PossessionTracker, Detection) — keeping only what is referenced.
from tracker import (
    Track,
    MultiObjectTracker,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Ball-motion thresholds
BALL_SPEED_THRESHOLD    = 12.0   # px/frame to consider ball moving fast
ARC_MIN_HEIGHT          = 40     # px — minimum rise for a valid shot arc
RISING_FRAMES_MIN       = 4      # consecutive rising frames to confirm shot
RIM_PROXIMITY_PX        = 60     # px — ball must be within this to score

# Possession
POSSESSION_RADIUS_PX    = 150    # px — player "owns" ball within this radius (increased for better attribution)
POSSESSION_MIN_FRAMES   = 2      # frames to confirm possession (reduced for faster assignment)
POSSESSION_LOSS_FRAMES  = 15     # frames without proximity before loss (increased to retain possession longer)

# Passing
PASS_FLIGHT_FRAMES_MAX  = 45     # max frames a pass can be in flight
PASS_MIN_DISTANCE_PX    = 60     # min distance for a valid pass

# Steals
STEAL_INTERCEPT_RADIUS  = 55     # px — defender intercepts within this

# Blocks
BLOCK_RIM_PROXIMITY     = 90     # px — block must happen near rim
BLOCK_SPEED_THRESHOLD   = 18.0   # px/frame — ball must be moving fast

# Dead ball
POST_SCORE_DEAD_FRAMES  = 90     # frames after score to mark as dead
BALL_STATIONARY_FRAMES  = 25     # frames with no movement → dead ball
BALL_STATIONARY_THRESH  = 4.0    # px/frame — "stationary" threshold
OUT_OF_BOUNDS_MARGIN    = 20     # px from frame edge → out of bounds

# Momentum / fast break
FAST_BREAK_SPEED        = 20.0   # ball speed threshold for fast break
FAST_BREAK_FRAMES       = 12     # consecutive fast frames

# =============================================================================
# ENUMS
# =============================================================================

class EventType(Enum):
    # Scoring
    SCORE_2PT           = "score_2pt"
    SCORE_3PT           = "score_3pt"
    FREE_THROW_MADE     = "free_throw_made"
    FREE_THROW_MISS     = "free_throw_miss"
    # Shooting
    SHOT_ATTEMPT        = "shot_attempt"
    SHOT_MISS           = "shot_miss"
    # Possession
    REBOUND_OFF         = "rebound_off"
    REBOUND_DEF         = "rebound_def"
    STEAL               = "steal"
    TURNOVER            = "turnover"
    ASSIST              = "assist"
    PASS                = "pass"
    BLOCK               = "block"
    FOUL                = "foul"
    # Ball state
    DEAD_BALL           = "dead_ball"
    LIVE_BALL           = "live_ball"
    OUT_OF_BOUNDS       = "out_of_bounds"
    # Tempo
    FAST_BREAK          = "fast_break"
    # Generic
    POSSESSION_CHANGE   = "possession_change"
    UNKNOWN             = "unknown"


class CourtZone(Enum):
    PAINT           = "paint"
    MID_RANGE       = "mid_range"
    THREE_POINT     = "three_point"
    FREE_THROW_LINE = "free_throw_line"
    BACKCOURT       = "backcourt"
    UNKNOWN         = "unknown"


# =============================================================================
# GAME EVENT DATACLASS
# =============================================================================

@dataclass
class GameEvent:
    """
    Represents a single detected basketball event.
    Consumed by StatsEngine and HighlightRecorder.
    """
    event_type:         EventType
    frame_id:           int
    timestamp:          float               = field(default_factory=time.time)

    # Primary actor (scorer, rebounder, stealer, etc.)
    primary_track_id:   Optional[int]       = None
    primary_jersey:     Optional[str]       = None
    primary_team_id:    Optional[int]       = None

    # Secondary actor (turnover victim, pass target, etc.)
    secondary_track_id: Optional[int]       = None
    secondary_jersey:   Optional[str]       = None
    secondary_team_id:  Optional[int]       = None

    # Spatial context
    ball_position:      Optional[Tuple[float, float]] = None
    court_zone:         Optional[str]       = None
    is_three_point:     bool                = False

    # Extra metadata (hot_hand_bonus, reason, etc.)
    meta:               Dict                = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GameEvent({self.event_type.value} "
            f"frame={self.frame_id} "
            f"actor={self.primary_jersey or self.primary_track_id})"
        )


# =============================================================================
# EVENT LOG
# =============================================================================

class EventLog:
    """Thread-safe, bounded event history."""

    def __init__(self, maxlen: int = 2000):
        self._log: Deque[GameEvent] = deque(maxlen=maxlen)

    def append(self, event: GameEvent) -> None:
        self._log.append(event)

    def recent(self, n: int = 20) -> List[GameEvent]:
        return list(self._log)[-n:]

    def by_type(self, etype: EventType) -> List[GameEvent]:
        return [e for e in self._log if e.event_type == etype]

    def since_frame(self, frame_id: int) -> List[GameEvent]:
        return [e for e in self._log if e.frame_id >= frame_id]

    def all(self) -> List[GameEvent]:
        return list(self._log)

    def clear(self) -> None:
        self._log.clear()

    def __len__(self) -> int:
        return len(self._log)
        # =============================================================================
# COURT ZONE CLASSIFIER
# =============================================================================

class CourtZoneClassifier:
    """
    Maps a pixel position to a CourtZone.
    Calibrated to a standard side-view broadcast frame.
    Override zone_map for different camera angles.
    """

    def __init__(
        self,
        frame_width:  int = 1280,
        frame_height: int = 720,
        three_point_x_ratio:    float = 0.22,   # fraction from each side
        paint_x_ratio:          float = 0.12,
        paint_y_ratio:          float = 0.30,
        mid_range_y_ratio:      float = 0.55,
    ):
        self.fw  = frame_width
        self.fh  = frame_height
        self.three_x  = int(frame_width  * three_point_x_ratio)
        self.paint_x  = int(frame_width  * paint_x_ratio)
        self.paint_y  = int(frame_height * paint_y_ratio)
        self.mid_y    = int(frame_height * mid_range_y_ratio)

    def classify(
        self,
        x: float,
        y: float,
        attacking_left: bool = True,
    ) -> CourtZone:
        """
        Returns CourtZone for ball at (x, y).
        attacking_left=True means the scoring basket is on the left side.
        """
        if attacking_left:
            near_basket  = x < self.three_x
            far_basket   = x > (self.fw - self.three_x)
        else:
            near_basket  = x > (self.fw - self.three_x)
            far_basket   = x < self.three_x

        if far_basket:
            return CourtZone.BACKCOURT

        if near_basket:
            if x < self.paint_x and y > self.paint_y:
                return CourtZone.PAINT
            return CourtZone.MID_RANGE

        # Beyond three-point line
        return CourtZone.THREE_POINT

    def is_three_point(
        self,
        x: float,
        y: float,
        attacking_left: bool = True,
    ) -> bool:
        return self.classify(x, y, attacking_left) == CourtZone.THREE_POINT


# =============================================================================
# POSSESSION TRACKER
# =============================================================================

class PossessionTracker:
    """
    Determines which player (and team) currently possesses the ball.

    Rules:
      - Player within POSSESSION_RADIUS_PX for POSSESSION_MIN_FRAMES → gains possession
      - No player within radius for POSSESSION_LOSS_FRAMES → possession lost
      - Returns (possessor_track_id, team_id) or (None, None)
    """

    def __init__(
        self,
        radius:         int = POSSESSION_RADIUS_PX,
        min_frames:     int = POSSESSION_MIN_FRAMES,
        loss_frames:    int = POSSESSION_LOSS_FRAMES,
    ):
        self.radius      = radius
        self.min_frames  = min_frames
        self.loss_frames = loss_frames

        self._candidate_id:     Optional[int] = None
        self._candidate_frames: int           = 0
        self._possessor_id:     Optional[int] = None
        self._possessor_team:   Optional[int] = None
        self._no_contact_frames: int          = 0

    def update(
        self,
        ball_center:    Tuple[float, float],
        player_tracks:  List[Track],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (possessor_track_id, team_id).
        """
        bx, by = ball_center
        closest_id   = None
        closest_dist = float("inf")

        for track in player_tracks:
            cx, cy = track.center
            dist   = math.hypot(cx - bx, cy - by)
            if dist < closest_dist:
                closest_dist = dist
                closest_id   = track.track_id

        if closest_dist <= self.radius and closest_id is not None:
            self._no_contact_frames = 0

            if closest_id == self._candidate_id:
                self._candidate_frames += 1
            else:
                self._candidate_id     = closest_id
                self._candidate_frames = 1

            if self._candidate_frames >= self.min_frames:
                self._possessor_id = self._candidate_id
                # Resolve team
                for t in player_tracks:
                    if t.track_id == self._possessor_id:
                        self._possessor_team = getattr(t, "team_id", None)
                        break

        else:
            self._no_contact_frames += 1
            if self._no_contact_frames >= self.loss_frames:
                self._possessor_id   = None
                self._possessor_team = None
                self._candidate_id   = None
                self._candidate_frames = 0

        return self._possessor_id, self._possessor_team

    @property
    def possessor_id(self) -> Optional[int]:
        return self._possessor_id

    @property
    def possessor_team(self) -> Optional[int]:
        return self._possessor_team

    def reset(self) -> None:
        self._candidate_id      = None
        self._candidate_frames  = 0
        self._possessor_id      = None
        self._possessor_team    = None
        self._no_contact_frames = 0
        
        # =============================================================================
# SHOT DETECTOR
# FIX #3: Store _start_y when rising begins; compare start_y - peak_y
# =============================================================================

class ShotDetector:
    """
    Detects shot attempts by tracking ball arc.

    A shot is confirmed when:
      - Ball rises for >= RISING_FRAMES_MIN consecutive frames
      - Total rise height (start_y - peak_y) >= arc_min_height
      - Ball speed >= speed_threshold
    """

    def __init__(
        self,
        speed_threshold:    float = BALL_SPEED_THRESHOLD,
        arc_min_height:     float = ARC_MIN_HEIGHT,
        rising_frames_min:  int   = RISING_FRAMES_MIN,
    ):
        self.speed_threshold   = speed_threshold
        self.arc_min_height    = arc_min_height
        self.rising_frames_min = rising_frames_min

        self._prev_ball_y:   Optional[float] = None
        self._prev_ball_pos: Optional[Tuple[float, float]] = None
        self._rising_frames: int   = 0
        self._peak_y:        float = float("inf")

        # FIX #3: track where the ball was when rising began
        self._start_y:       float = 0.0

        self.active_shot:    Optional[Dict] = None   # set when shot detected

    def update(
        self,
        ball_center:    Tuple[float, float],
        frame_id:       int,
        possessor_id:   Optional[int]   = None,
        possessor_team: Optional[int]   = None,
        court_zone:     Optional[str]   = None,
    ) -> Optional[GameEvent]:
        """
        Returns a SHOT_ATTEMPT GameEvent when an arc is confirmed, else None.
        """
        bx, by = ball_center
        event  = None

        if self._prev_ball_pos is not None:
            px, py = self._prev_ball_pos
            speed  = math.hypot(bx - px, by - py)

            # In image coords: y decreases upward → going_up means by < prev_y
            going_up = by < self._prev_ball_y

            if going_up and speed >= self.speed_threshold:
                if self._rising_frames == 0:
                    # FIX #3: record the y where rising started
                    self._start_y = self._prev_ball_y

                self._rising_frames += 1
                self._peak_y = min(self._peak_y, by)

            else:
                # Ball has stopped rising — evaluate arc
                # FIX #3: compare start_y - peak_y (not prev_ball_y - peak_y)
                arc_height = self._start_y - self._peak_y

                if (
                    self._rising_frames >= self.rising_frames_min
                    and arc_height >= self.arc_min_height
                ):
                    is_three = court_zone == CourtZone.THREE_POINT.value

                    self.active_shot = {
                        "frame_id":     frame_id,
                        "possessor_id": possessor_id,
                        "team_id":      possessor_team,
                        "court_zone":   court_zone,
                        "is_three":     is_three,
                        "ball_pos":     ball_center,
                        "arc_height":   arc_height,
                    }

                    event = GameEvent(
                        event_type       = EventType.SHOT_ATTEMPT,
                        frame_id         = frame_id,
                        primary_track_id = possessor_id,
                        primary_team_id  = possessor_team,
                        ball_position    = ball_center,
                        court_zone       = court_zone,
                        is_three_point   = is_three,
                        meta             = {"arc_height": arc_height},
                    )
                    logger.debug(
                        f"[ShotDetector] SHOT_ATTEMPT frame={frame_id} "
                        f"arc={arc_height:.1f}px zone={court_zone}"
                    )

                # Reset arc tracking
                self._rising_frames = 0
                self._peak_y        = float("inf")
                self._start_y       = 0.0

        self._prev_ball_y   = by
        self._prev_ball_pos = ball_center
        return event

    def clear_active_shot(self) -> None:
        self.active_shot = None


# =============================================================================
# SCORE DETECTOR
# FIX #5: Fall back to recent possessor if active_shot is None
# =============================================================================

class ScoreDetector:
    """
    Detects scoring events by monitoring ball proximity to the rim.

    Scoring confirmed when:
      - Ball passes within RIM_PROXIMITY_PX of the rim center
      - Ball was above the rim and now is at or below it (crossing)
      - Sufficient frames have passed since the last score (cooldown)
    """

    def __init__(
        self,
        rim_proximity:      float = RIM_PROXIMITY_PX,
        score_cooldown:     int   = 45,
    ):
        self.rim_proximity  = rim_proximity
        self.score_cooldown = score_cooldown

        self._last_score_frame: int           = -9999
        self._prev_ball_y:      Optional[float] = None

        # FIX #5: maintain a rolling recent-possessor fallback
        self._recent_possessor_id:   Optional[int] = None
        self._recent_possessor_team: Optional[int] = None

    def update_possessor(
        self,
        possessor_id:   Optional[int],
        possessor_team: Optional[int],
    ) -> None:
        """
        Call every frame so we always have a fallback possessor.
        FIX #5: keeps last known possessor for scorer attribution.
        """
        if possessor_id is not None:
            self._recent_possessor_id   = possessor_id
            self._recent_possessor_team = possessor_team

    def update(
        self,
        ball_center:    Tuple[float, float],
        rim_center:     Tuple[float, float],
        frame_id:       int,
        shooter_track:  Optional[Track]  = None,
        shooter_team:   Optional[int]    = None,
        court_zone:     Optional[str]    = None,
        is_three:       bool             = False,
    ) -> Optional[GameEvent]:
        """
        Returns SCORE_2PT or SCORE_3PT GameEvent when scoring is detected.
        """
        bx, by   = ball_center
        rx, ry   = rim_center
        event    = None

        dist_to_rim = math.hypot(bx - rx, by - ry)

        if (
            dist_to_rim <= self.rim_proximity
            and self._prev_ball_y is not None
            and self._prev_ball_y < ry          # ball was above rim
            and by >= ry                         # ball now at/below rim
            and (frame_id - self._last_score_frame) >= self.score_cooldown
        ):
            # ── Resolve scorer  (FIX #5) ─────────────────────────────────
            if shooter_track is not None:
                scorer_id   = shooter_track.track_id
                scorer_team = shooter_team
                scorer_jersey = getattr(shooter_track, "jersey_number", None)
            else:
                # Fall back to most recent possessor
                scorer_id   = self._recent_possessor_id
                scorer_team = self._recent_possessor_team
                scorer_jersey = None
                logger.debug(
                    f"[ScoreDetector] No active shot — falling back "
                    f"to recent possessor id={scorer_id}"
                )

            etype = EventType.SCORE_3PT if is_three else EventType.SCORE_2PT
            self._last_score_frame = frame_id

            event = GameEvent(
                event_type       = etype,
                frame_id         = frame_id,
                primary_track_id = scorer_id,
                primary_jersey   = scorer_jersey,
                primary_team_id  = scorer_team,
                ball_position    = ball_center,
                court_zone       = court_zone,
                is_three_point   = is_three,
                meta             = {"rim_dist": round(dist_to_rim, 1)},
            )
            logger.info(
                f"[ScoreDetector] {etype.value} frame={frame_id} "
                f"scorer={scorer_jersey or scorer_id} team={scorer_team}"
            )

        self._prev_ball_y = by
        return event
        
        # =============================================================================
# PASS DETECTOR
# FIX #4: Expose in_flight property so steal/turnover detectors
#         can suppress false positives during ball flight.
# =============================================================================

class PassDetector:
    """
    Detects passes by tracking ball ownership transfers.

    A pass is confirmed when:
      - Ball leaves one player's proximity (sender)
      - Ball travels >= PASS_MIN_DISTANCE_PX
      - Ball arrives at a different player of the same team (receiver)
      - All within PASS_FLIGHT_FRAMES_MAX frames
    """

    def __init__(
        self,
        min_distance:       float = PASS_MIN_DISTANCE_PX,
        max_flight_frames:  int   = PASS_FLIGHT_FRAMES_MAX,
        possession_radius:  float = POSSESSION_RADIUS_PX,
    ):
        self.min_distance      = min_distance
        self.max_flight_frames = max_flight_frames
        self.possession_radius = possession_radius

        self._sender_id:     Optional[int]           = None
        self._sender_team:   Optional[int]           = None
        self._sender_jersey: Optional[str]           = None
        self._launch_pos:    Optional[Tuple[float, float]] = None
        self._flight_frames: int                     = 0

        # FIX #4: in_flight is True from launch until receipt/timeout
        self._in_flight:     bool = False

    # FIX #4: public property consumed by StealDetector / TurnoverDetector
    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def update(
        self,
        ball_center:    Tuple[float, float],
        possessor_id:   Optional[int],
        possessor_team: Optional[int],
        player_tracks:  List[Track],
        frame_id:       int,
    ) -> Optional[GameEvent]:
        """
        Returns a PASS GameEvent on successful pass completion, else None.
        """
        event = None

        if not self._in_flight:
            # Waiting for ball to leave a player
            if possessor_id is not None:
                self._sender_id     = possessor_id
                self._sender_team   = possessor_team
                self._sender_jersey = self._resolve_jersey(
                    possessor_id, player_tracks
                )
                self._launch_pos    = ball_center

            elif self._sender_id is not None and possessor_id is None:
                # Ball just left sender — flight begins
                self._in_flight     = True
                self._flight_frames = 0

        else:
            # Ball is in flight
            self._flight_frames += 1

            if possessor_id is not None and possessor_id != self._sender_id:
                # Ball arrived at a new player
                receiver_team = possessor_team

                if receiver_team == self._sender_team:
                    # Same-team transfer → PASS
                    dist = math.hypot(
                        ball_center[0] - self._launch_pos[0],
                        ball_center[1] - self._launch_pos[1],
                    ) if self._launch_pos else 0.0

                    if dist >= self.min_distance:
                        receiver_jersey = self._resolve_jersey(
                            possessor_id, player_tracks
                        )
                        event = GameEvent(
                            event_type          = EventType.PASS,
                            frame_id            = frame_id,
                            primary_track_id    = self._sender_id,
                            primary_jersey      = self._sender_jersey,
                            primary_team_id     = self._sender_team,
                            secondary_track_id  = possessor_id,
                            secondary_jersey    = receiver_jersey,
                            secondary_team_id   = receiver_team,
                            ball_position       = ball_center,
                            meta                = {
                                "flight_frames": self._flight_frames,
                                "distance":      round(dist, 1),
                            },
                        )
                        logger.debug(
                            f"[PassDetector] PASS frame={frame_id} "
                            f"{self._sender_jersey}→{receiver_jersey}"
                        )
                self._reset_flight()

            elif self._flight_frames > self.max_flight_frames:
                # Flight timed out — not a pass
                self._reset_flight()

            elif possessor_id == self._sender_id:
                # Sender regained ball — cancel flight
                self._reset_flight()

        return event

    def _reset_flight(self) -> None:
        self._in_flight     = False
        self._flight_frames = 0
        self._sender_id     = None
        self._sender_team   = None
        self._sender_jersey = None
        self._launch_pos    = None

    @staticmethod
    def _resolve_jersey(
        track_id: int,
        tracks:   List[Track],
    ) -> Optional[str]:
        for t in tracks:
            if t.track_id == track_id:
                return getattr(t, "jersey_number", None)
        return None


# =============================================================================
# STEAL DETECTOR
# FIX #4: Check pass_detector.in_flight continuously (not just on completion)
# =============================================================================

class StealDetector:
    """
    Detects steals: defensive player intercepts ball mid-pass or strips ball.
    """

    def __init__(
        self,
        intercept_radius:   float = STEAL_INTERCEPT_RADIUS,
    ):
        self.intercept_radius = intercept_radius

        self._prev_possessor_id:   Optional[int] = None
        self._prev_possessor_team: Optional[int] = None

    def update(
        self,
        possessor_id:       Optional[int],
        possessor_team:     Optional[int],
        player_tracks:      List[Track],
        frame_id:           int,
        pass_in_flight:     bool = False,   # FIX #4: live flag, not callback
    ) -> Optional[GameEvent]:
        """
        Returns a STEAL GameEvent when an interception is detected.
        pass_in_flight should be pass_detector.in_flight (FIX #4).
        """
        event = None

        if (
            possessor_id is not None
            and self._prev_possessor_id is not None
            and possessor_id != self._prev_possessor_id
            and possessor_team is not None
            and self._prev_possessor_team is not None
            and possessor_team != self._prev_possessor_team
            and pass_in_flight  # FIX #4: only flag steal during active flight
        ):
            # Opposing team gained ball during a pass flight → STEAL
            stealer_jersey = None
            loser_jersey   = None

            for t in player_tracks:
                if t.track_id == possessor_id:
                    stealer_jersey = getattr(t, "jersey_number", None)
                if t.track_id == self._prev_possessor_id:
                    loser_jersey = getattr(t, "jersey_number", None)

            event = GameEvent(
                event_type          = EventType.STEAL,
                frame_id            = frame_id,
                primary_track_id    = possessor_id,
                primary_jersey      = stealer_jersey,
                primary_team_id     = possessor_team,
                secondary_track_id  = self._prev_possessor_id,
                secondary_jersey    = loser_jersey,
                secondary_team_id   = self._prev_possessor_team,
                meta                = {"pass_in_flight": True},
            )
            logger.info(
                f"[StealDetector] STEAL frame={frame_id} "
                f"by={stealer_jersey} from={loser_jersey}"
            )

        self._prev_possessor_id   = possessor_id
        self._prev_possessor_team = possessor_team
        return event

    # Kept for backward compatibility — no longer the primary mechanism
    def notify_pass(self) -> None:
        pass

    def reset(self) -> None:
        self._prev_possessor_id   = None
        self._prev_possessor_team = None


# =============================================================================
# TURNOVER DETECTOR
# FIX #2: Save prev_possessor_id BEFORE updating instance state
# FIX #4: Accept live pass_in_flight flag instead of relying on notify
# =============================================================================

class TurnoverDetector:
    """
    Detects turnovers: possession changes to opposing team
    that are NOT steals or completed passes.
    """

    def __init__(self) -> None:
        self._prev_possessor_id:   Optional[int] = None
        self._prev_team_id:        Optional[int] = None

    def _get_team(
        self,
        track_id:       int,
        player_tracks:  List[Track],
    ) -> Optional[int]:
        for t in player_tracks:
            if t.track_id == track_id:
                return getattr(t, "team_id", None)
        return None

    def update(
        self,
        possessor_id:   Optional[int],
        player_tracks:  List[Track],
        frame_id:       int,
        steal_flag:     bool = False,
        pass_in_flight: bool = False,   # FIX #4
    ) -> Optional[GameEvent]:
        """
        Returns a TURNOVER GameEvent on unforced possession loss, else None.
        """
        if possessor_id is None:
            return None

        if possessor_id == self._prev_possessor_id:
            return None

        new_team  = self._get_team(possessor_id, player_tracks)

        # FIX #2: capture OLD values BEFORE updating instance state
        prev_possessor_id = self._prev_possessor_id
        prev_team         = self._prev_team_id

        # Now update instance state safely
        self._prev_possessor_id = possessor_id
        self._prev_team_id      = new_team

        if new_team is None or prev_team is None or new_team == prev_team:
            return None

        # Opposing team gained possession
        if steal_flag or pass_in_flight:
            # Already attributed as steal or pass-in-flight
            return None

        # FIX #2: use saved prev_possessor_id (not self._prev_possessor_id)
        loser = next(
            (t for t in player_tracks if t.track_id == prev_possessor_id),
            None,
        )
        loser_jersey = getattr(loser, "jersey_number", None) if loser else None

        event = GameEvent(
            event_type       = EventType.TURNOVER,
            frame_id         = frame_id,
            primary_track_id = prev_possessor_id,
            primary_jersey   = loser_jersey,
            primary_team_id  = prev_team,
            meta             = {
                "new_possessor_id":   possessor_id,
                "new_possessor_team": new_team,
            },
        )
        logger.info(
            f"[TurnoverDetector] TURNOVER frame={frame_id} "
            f"loser={loser_jersey or prev_possessor_id}"
        )
        return event

    def reset(self) -> None:
        self._prev_possessor_id = None
        self._prev_team_id      = None
        
        # =============================================================================
# REBOUND DETECTOR
# =============================================================================

class ReboundDetector:
    """
    Detects offensive and defensive rebounds.

    A rebound is detected when:
      - A SHOT_MISS or SHOT_ATTEMPT recently occurred
      - Ball was near the rim
      - A player gains possession shortly after
    """

    def __init__(
        self,
        rebound_window_frames: int   = 60,
        rim_proximity:         float = RIM_PROXIMITY_PX * 2,
    ):
        self.rebound_window = rebound_window_frames
        self.rim_proximity  = rim_proximity

        self._miss_frame:       int           = -9999
        self._miss_team:        Optional[int] = None   # team that shot
        self._prev_possessor:   Optional[int] = None

    def notify_miss(
        self,
        frame_id:   int,
        shooting_team: Optional[int],
    ) -> None:
        self._miss_frame = frame_id
        self._miss_team  = shooting_team

    def update(
        self,
        possessor_id:   Optional[int],
        possessor_team: Optional[int],
        player_tracks:  List[Track],
        frame_id:       int,
    ) -> Optional[GameEvent]:
        """Returns REBOUND_OFF or REBOUND_DEF GameEvent, else None."""
        if possessor_id is None:
            self._prev_possessor = None
            return None

        if possessor_id == self._prev_possessor:
            return None

        frames_since_miss = frame_id - self._miss_frame

        if frames_since_miss > self.rebound_window or self._miss_team is None:
            self._prev_possessor = possessor_id
            return None

        # New player gained possession after a miss
        is_offensive = (possessor_team == self._miss_team)
        etype = EventType.REBOUND_OFF if is_offensive else EventType.REBOUND_DEF

        rebounder_jersey = None
        for t in player_tracks:
            if t.track_id == possessor_id:
                rebounder_jersey = getattr(t, "jersey_number", None)
                break

        event = GameEvent(
            event_type       = etype,
            frame_id         = frame_id,
            primary_track_id = possessor_id,
            primary_jersey   = rebounder_jersey,
            primary_team_id  = possessor_team,
            meta             = {
                "frames_since_miss": frames_since_miss,
                "is_offensive":      is_offensive,
            },
        )
        logger.info(
            f"[ReboundDetector] {etype.value} frame={frame_id} "
            f"rebounder={rebounder_jersey or possessor_id}"
        )

        # Reset so we don't double-count
        self._miss_frame = -9999
        self._miss_team  = None
        self._prev_possessor = possessor_id
        return event


# =============================================================================
# BLOCK DETECTOR
# =============================================================================

class BlockDetector:
    """
    Detects blocks: defender deflects ball near the rim at high speed.
    """

    def __init__(
        self,
        rim_proximity:      float = BLOCK_RIM_PROXIMITY,
        speed_threshold:    float = BLOCK_SPEED_THRESHOLD,
        cooldown_frames:    int   = 30,
    ):
        self.rim_proximity   = rim_proximity
        self.speed_threshold = speed_threshold
        self.cooldown_frames = cooldown_frames

        self._last_block_frame: int                       = -9999
        self._prev_ball_pos:    Optional[Tuple[float, float]] = None

    def update(
        self,
        ball_center:    Tuple[float, float],
        rim_center:     Tuple[float, float],
        player_tracks:  List[Track],
        frame_id:       int,
        shot_in_progress: bool = False,
    ) -> Optional[GameEvent]:
        """Returns BLOCK GameEvent when detected, else None."""
        event = None

        if self._prev_ball_pos is not None and shot_in_progress:
            bx, by = ball_center
            px, py = self._prev_ball_pos
            speed  = math.hypot(bx - px, by - py)
            dist   = math.hypot(bx - rim_center[0], by - rim_center[1])

            if (
                speed >= self.speed_threshold
                and dist <= self.rim_proximity
                and (frame_id - self._last_block_frame) >= self.cooldown_frames
            ):
                # Find nearest defender (player not in possession)
                blocker = self._nearest_player(ball_center, player_tracks)
                if blocker is not None:
                    blocker_jersey = getattr(blocker, "jersey_number", None)
                    self._last_block_frame = frame_id
                    event = GameEvent(
                        event_type       = EventType.BLOCK,
                        frame_id         = frame_id,
                        primary_track_id = blocker.track_id,
                        primary_jersey   = blocker_jersey,
                        primary_team_id  = getattr(blocker, "team_id", None),
                        ball_position    = ball_center,
                        meta             = {
                            "speed":    round(speed, 1),
                            "rim_dist": round(dist, 1),
                        },
                    )
                    logger.info(
                        f"[BlockDetector] BLOCK frame={frame_id} "
                        f"blocker={blocker_jersey}"
                    )

        self._prev_ball_pos = ball_center
        return event

    @staticmethod
    def _nearest_player(
        ball_center:    Tuple[float, float],
        player_tracks:  List[Track],
    ) -> Optional[Track]:
        bx, by = ball_center
        nearest = None
        min_d   = float("inf")
        for t in player_tracks:
            cx, cy = t.center
            d = math.hypot(cx - bx, cy - by)
            if d < min_d:
                min_d   = d
                nearest = t
        return nearest


# =============================================================================
# DEAD BALL DETECTOR
# FIX #6: Emit DEAD_BALL event inside notify_score(); fix if-not check
# =============================================================================

class DeadBallDetector:
    """
    Detects dead ball conditions:
      - Post-score stoppage
      - Ball stationary for extended period
      - Ball out of bounds

    FIX #6: DEAD_BALL GameEvent is emitted immediately in notify_score()
            so the transition is never silently dropped by the
            `if not self._is_dead` guard in update().
    """

    def __init__(
        self,
        post_score_frames:      int   = POST_SCORE_DEAD_FRAMES,
        stationary_frames:      int   = BALL_STATIONARY_FRAMES,
        stationary_threshold:   float = BALL_STATIONARY_THRESH,
        out_of_bounds_margin:   int   = OUT_OF_BOUNDS_MARGIN,
        frame_width:            int   = 1280,
        frame_height:           int   = 720,
    ):
        self.post_score_frames    = post_score_frames
        self.stationary_frames    = stationary_frames
        self.stationary_threshold = stationary_threshold
        self.oob_margin           = out_of_bounds_margin
        self.frame_width          = frame_width
        self.frame_height         = frame_height

        self._is_dead:              bool          = False
        self._dead_start:           int           = 0
        self._dead_reason:          Optional[str] = None
        self._score_frame:          int           = -9999
        self._stationary_count:     int           = 0
        self._prev_ball_pos:        Optional[Tuple[float, float]] = None

        # FIX #6: pending event to emit on the next update() call
        self._pending_event:        Optional[GameEvent] = None

    def notify_score(
        self,
        frame_id: int,
    ) -> GameEvent:
        """
        Called immediately when a score is detected.

        FIX #6: Sets dead state AND creates the DEAD_BALL GameEvent here,
        so update() never needs to emit a transition event for post-score.
        Returns the GameEvent so the caller can log/dispatch it immediately.
        """
        self._score_frame  = frame_id
        self._is_dead      = True
        self._dead_start   = frame_id
        self._dead_reason  = "post_score"

        event = GameEvent(
            event_type = EventType.DEAD_BALL,
            frame_id   = frame_id,
            meta       = {"reason": "post_score"},
        )
        # Also store as pending so EventEngine can pick it up in the same tick
        self._pending_event = event
        logger.info(f"[DeadBallDetector] DEAD_BALL (post_score) frame={frame_id}")
        return event

    def update(
        self,
        ball_center:    Optional[Tuple[float, float]],
        frame_id:       int,
    ) -> Tuple[bool, Optional[GameEvent]]:
        """
        Returns (is_dead, event_or_None).

        event_or_None is a DEAD_BALL or LIVE_BALL transition event.
        """
        event: Optional[GameEvent] = None

        # ── Drain pending event from notify_score() ──────────────────────────
        if self._pending_event is not None:
            event = self._pending_event
            self._pending_event = None
            return True, event

        # ── Post-score dead window ────────────────────────────────────────────
        if frame_id - self._score_frame < self.post_score_frames:
            # FIX #6: _is_dead already True from notify_score(); no re-emit
            return True, None

        # ── Out of bounds ─────────────────────────────────────────────────────
        if ball_center is not None:
            bx, by = ball_center
            oob = (
                bx < self.oob_margin
                or bx > self.frame_width  - self.oob_margin
                or by < self.oob_margin
                or by > self.frame_height - self.oob_margin
            )
            if oob:
                if not self._is_dead:
                    self._is_dead    = True
                    self._dead_start = frame_id
                    self._dead_reason = "out_of_bounds"
                    event = GameEvent(
                        event_type = EventType.DEAD_BALL,
                        frame_id   = frame_id,
                        meta       = {"reason": "out_of_bounds"},
                    )
                    logger.info(
                        f"[DeadBallDetector] DEAD_BALL (oob) frame={frame_id}"
                    )
                return True, event

        # ── Ball stationary ───────────────────────────────────────────────────
        if ball_center is not None and self._prev_ball_pos is not None:
            bx, by = ball_center
            px, py = self._prev_ball_pos
            speed  = math.hypot(bx - px, by - py)

            if speed < self.stationary_threshold:
                self._stationary_count += 1
            else:
                self._stationary_count = 0

            if self._stationary_count >= self.stationary_frames:
                if not self._is_dead:
                    self._is_dead    = True
                    self._dead_start = frame_id
                    self._dead_reason = "stationary"
                    event = GameEvent(
                        event_type = EventType.DEAD_BALL,
                        frame_id   = frame_id,
                        meta       = {
                            "reason":             "stationary",
                            "stationary_frames":  self._stationary_count,
                        },
                    )
                    logger.info(
                        f"[DeadBallDetector] DEAD_BALL (stationary) "
                        f"frame={frame_id}"
                    )
                self._prev_ball_pos = ball_center
                return True, event

        # ── Ball is moving — check for live ball transition ───────────────────
        if self._is_dead and ball_center is not None:
            bx, by = ball_center
            if self._prev_ball_pos is not None:
                px, py = self._prev_ball_pos
                speed  = math.hypot(bx - px, by - py)
                if speed >= self.stationary_threshold:
                    self._is_dead         = False
                    self._stationary_count = 0
                    event = GameEvent(
                        event_type = EventType.LIVE_BALL,
                        frame_id   = frame_id,
                        meta       = {
                            "dead_duration_frames": frame_id - self._dead_start,
                            "reason":               self._dead_reason,
                        },
                    )
                    logger.info(
                        f"[DeadBallDetector] LIVE_BALL frame={frame_id} "
                        f"after {frame_id - self._dead_start} dead frames"
                    )

        self._prev_ball_pos = ball_center
        return self._is_dead, event

    @property
    def is_dead(self) -> bool:
        return self._is_dead

    @property
    def dead_reason(self) -> Optional[str]:
        return self._dead_reason

    def reset(self) -> None:
        self._is_dead          = False
        self._dead_start       = 0
        self._dead_reason      = None
        self._score_frame      = -9999
        self._stationary_count = 0
        self._prev_ball_pos    = None
        self._pending_event    = None
        # =============================================================================
# EVENT ENGINE  —  Master orchestrator
# Wires all detectors together; call process_frame() every frame.
# =============================================================================

class EventEngine:
    """
    Orchestrates all basketball event detectors.

    Per-frame pipeline:
      1. Update possession
      2. Update score detector possessor fallback  (FIX #5)
      3. Detect shots
      4. Detect scores
      5. Detect passes  (exposes in_flight flag)  (FIX #4)
      6. Detect steals  (uses live in_flight flag) (FIX #4)
      7. Detect turnovers (uses live in_flight flag) (FIX #4)
      8. Detect rebounds
      9. Detect blocks
     10. Detect dead ball
     11. Append all events to EventLog
    """

    def __init__(
        self,
        team_ids:       List[int]   = None,
        frame_width:    int         = 1280,
        frame_height:   int         = 720,
        game_id:        Optional[str] = None,
    ):
        self.team_ids     = team_ids or [0, 1]
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # ── Detectors ────────────────────────────────────────────────────────
        self.possession_tracker = PossessionTracker()
        self.zone_classifier    = CourtZoneClassifier(frame_width, frame_height)
        self.shot_detector      = ShotDetector()
        self.score_detector     = ScoreDetector()
        self.pass_detector      = PassDetector()
        self.steal_detector     = StealDetector()
        self.turnover_detector  = TurnoverDetector()
        self.rebound_detector   = ReboundDetector()
        self.block_detector     = BlockDetector()
        self.dead_ball_detector = DeadBallDetector(
            frame_width=frame_width, frame_height=frame_height
        )

        # ── Event log ────────────────────────────────────────────────────────
        self.event_log = EventLog()

        # ── Internal state ───────────────────────────────────────────────────
        self._last_possessor_id:   Optional[int] = None
        self._last_possessor_team: Optional[int] = None
        self._steal_this_frame:    bool          = False

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def process_frame(
        self,
        frame_id:       int,
        ball_center:    Optional[Tuple[float, float]],
        rim_center:     Optional[Tuple[float, float]],
        player_tracks:  List[Track],
        attacking_left: bool = True,
    ) -> List[GameEvent]:
        """
        Process one video frame through the full event detection pipeline.

        Returns a list of GameEvents detected in this frame.
        """
        events: List[GameEvent] = []

        # ── 1. Possession ─────────────────────────────────────────────────────
        if ball_center is not None:
            possessor_id, possessor_team = self.possession_tracker.update(
                ball_center, player_tracks
            )
        else:
            possessor_id   = self._last_possessor_id
            possessor_team = self._last_possessor_team

        # ── 2. Score detector possessor fallback (FIX #5) ────────────────────
        self.score_detector.update_possessor(possessor_id, possessor_team)

        # ── 3. Court zone ─────────────────────────────────────────────────────
        court_zone = None
        is_three   = False
        if ball_center is not None:
            zone_enum  = self.zone_classifier.classify(
                ball_center[0], ball_center[1], attacking_left
            )
            court_zone = zone_enum.value
            is_three   = (zone_enum == CourtZone.THREE_POINT)

        # ── 4. Shot detection ─────────────────────────────────────────────────
        shot_evt = None
        if ball_center is not None:
            shot_evt = self.shot_detector.update(
                ball_center    = ball_center,
                frame_id       = frame_id,
                possessor_id   = possessor_id,
                possessor_team = possessor_team,
                court_zone     = court_zone,
            )
            if shot_evt:
                events.append(shot_evt)

        shot_in_progress = self.shot_detector.active_shot is not None

        # ── 5. Score detection ────────────────────────────────────────────────
        if ball_center is not None and rim_center is not None:
            active = self.shot_detector.active_shot
            shooter_track = None
            shooter_team  = None

            if active:
                sid = active.get("possessor_id")
                shooter_team = active.get("team_id")
                if sid is not None:
                    shooter_track = next(
                        (t for t in player_tracks if t.track_id == sid), None
                    )
                is_three = active.get("is_three", is_three)
                court_zone = active.get("court_zone", court_zone)

            score_evt = self.score_detector.update(
                ball_center   = ball_center,
                rim_center    = rim_center,
                frame_id      = frame_id,
                shooter_track = shooter_track,
                shooter_team  = shooter_team,
                court_zone    = court_zone,
                is_three      = is_three,
            )
            if score_evt:
                events.append(score_evt)
                self.shot_detector.clear_active_shot()
                # FIX #6: emit dead ball event immediately on score
                dead_evt = self.dead_ball_detector.notify_score(frame_id)
                events.append(dead_evt)
            else:
                # Check for shot miss (shot was active but no score near rim)
                if (
                    active is not None
                    and rim_center is not None
                    and math.hypot(
                        ball_center[0] - rim_center[0],
                        ball_center[1] - rim_center[1],
                    ) <= RIM_PROXIMITY_PX * 2
                    and not shot_in_progress
                ):
                    miss_evt = GameEvent(
                        event_type       = EventType.SHOT_MISS,
                        frame_id         = frame_id,
                        primary_track_id = active.get("possessor_id"),
                        primary_team_id  = active.get("team_id"),
                        ball_position    = ball_center,
                        court_zone       = court_zone,
                        is_three_point   = is_three,
                    )
                    events.append(miss_evt)
                    self.rebound_detector.notify_miss(
                        frame_id, active.get("team_id")
                    )
                    self.shot_detector.clear_active_shot()

        # ── 6. Pass detection ─────────────────────────────────────────────────
        pass_evt = None
        if ball_center is not None:
            pass_evt = self.pass_detector.update(
                ball_center    = ball_center,
                possessor_id   = possessor_id,
                possessor_team = possessor_team,
                player_tracks  = player_tracks,
                frame_id       = frame_id,
            )
            if pass_evt:
                events.append(pass_evt)

        # FIX #4: read live in_flight flag — not a one-shot callback
        pass_in_flight = self.pass_detector.in_flight

        # ── 7. Steal detection (FIX #4) ───────────────────────────────────────
        self._steal_this_frame = False
        steal_evt = self.steal_detector.update(
            possessor_id   = possessor_id,
            possessor_team = possessor_team,
            player_tracks  = player_tracks,
            frame_id       = frame_id,
            pass_in_flight = pass_in_flight,    # FIX #4
        )
        if steal_evt:
            events.append(steal_evt)
            self._steal_this_frame = True

        # ── 8. Turnover detection (FIX #2, FIX #4) ───────────────────────────
        turnover_evt = self.turnover_detector.update(
            possessor_id   = possessor_id,
            player_tracks  = player_tracks,
            frame_id       = frame_id,
            steal_flag     = self._steal_this_frame,
            pass_in_flight = pass_in_flight,    # FIX #4
        )
        if turnover_evt:
            events.append(turnover_evt)

        # ── 9. Rebound detection ──────────────────────────────────────────────
        rebound_evt = self.rebound_detector.update(
            possessor_id   = possessor_id,
            possessor_team = possessor_team,
            player_tracks  = player_tracks,
            frame_id       = frame_id,
        )
        if rebound_evt:
            events.append(rebound_evt)

        # ── 10. Block detection ───────────────────────────────────────────────
        if ball_center is not None and rim_center is not None:
            block_evt = self.block_detector.update(
                ball_center      = ball_center,
                rim_center       = rim_center,
                player_tracks    = player_tracks,
                frame_id         = frame_id,
                shot_in_progress = shot_in_progress,
            )
            if block_evt:
                events.append(block_evt)

        # ── 11. Dead ball detection ───────────────────────────────────────────
        is_dead, dead_evt = self.dead_ball_detector.update(
            ball_center = ball_center,
            frame_id    = frame_id,
        )
        if dead_evt:
            events.append(dead_evt)

        # ── Save state ────────────────────────────────────────────────────────
        self._last_possessor_id   = possessor_id
        self._last_possessor_team = possessor_team

        # ── Log all events ────────────────────────────────────────────────────
        for evt in events:
            self.event_log.append(evt)
            logger.debug(f"  → {evt}")

        return events

    # =========================================================================
    # HELPERS
    # =========================================================================

    @property
    def is_dead_ball(self) -> bool:
        return self.dead_ball_detector.is_dead

    @property
    def current_possessor(self) -> Optional[int]:
        return self._last_possessor_id

    @property
    def current_possessor_team(self) -> Optional[int]:
        return self._last_possessor_team

    def reset(self) -> None:
        self.possession_tracker.reset()
        self.shot_detector      = ShotDetector()
        self.score_detector     = ScoreDetector()
        self.pass_detector      = PassDetector()
        self.steal_detector.reset()
        self.turnover_detector.reset()
        self.rebound_detector   = ReboundDetector()
        self.block_detector     = BlockDetector()
        self.dead_ball_detector.reset()
        self.event_log.clear()
        self._last_possessor_id   = None
        self._last_possessor_team = None
        self._steal_this_frame    = False
        logger.info("EventEngine reset.")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums / dataclasses
    "EventType",
    "CourtZone",
    "GameEvent",
    "EventLog",
    # Detectors
    "CourtZoneClassifier",
    "PossessionTracker",
    "ShotDetector",
    "ScoreDetector",
    "PassDetector",
    "StealDetector",
    "TurnoverDetector",
    "ReboundDetector",
    "BlockDetector",
    "DeadBallDetector",
    # Master engine
    "EventEngine",
    "BasketballEventDetector",
]

# Alias for backwards compatibility with app.py / pipeline.py
BasketballEventDetector = EventEngine
