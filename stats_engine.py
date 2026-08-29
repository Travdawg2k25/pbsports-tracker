# =============================================================================
# stats_engine.py  —  Purple Box Sports
# Real-time basketball statistics engine
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Deque, Dict, List, Optional,
    Set, Tuple, Union
)

import numpy as np

from basketball_events import (
    EventType,
    GameEvent,
    EventLog,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Rolling window sizes
MOMENTUM_WINDOW_EVENTS  = 10    # events to consider for momentum
STREAK_WINDOW_FRAMES    = 300   # frames for scoring streak detection
HOT_HAND_MIN_MAKES      = 3     # consecutive makes for "hot hand"
HOT_HAND_WINDOW         = 5     # attempts in which makes must occur

# Rating weights (Player Efficiency Rating approximation)
PER_WEIGHTS = {
    "points":       1.00,
    "rebounds":     1.20,
    "assists":      1.50,
    "steals":       2.00,
    "blocks":       2.00,
    "turnovers":   -1.50,
    "fouls":       -0.50,
    "fg_attempts": -0.44,
}

# Highlight score weights
HIGHLIGHT_WEIGHTS = {
    EventType.SCORE_3PT:    10,
    EventType.SCORE_2PT:     6,
    EventType.BLOCK:         8,
    EventType.STEAL:         7,
    EventType.ASSIST:        5,
    EventType.REBOUND_OFF:   4,
    EventType.REBOUND_DEF:   3,
    EventType.FAST_BREAK:    6,
    EventType.TURNOVER:     -2,
    EventType.FOUL:         -1,
}

# Minimum attempts for percentage display
MIN_ATTEMPTS_FOR_PCT = 1

# =============================================================================
# ENUMS
# =============================================================================

class StatCategory(Enum):
    SCORING     = "scoring"
    SHOOTING    = "shooting"
    REBOUNDING  = "rebounding"
    PLAYMAKING  = "playmaking"
    DEFENSE     = "defense"
    EFFICIENCY  = "efficiency"
    MOMENTUM    = "momentum"
    HIGHLIGHTS  = "highlights"


class TeamSide(Enum):
    HOME = 0
    AWAY = 1


# =============================================================================
# SHOOTING LINE DATACLASS
# =============================================================================

@dataclass
class ShootingLine:
    """
    Tracks shot attempts and makes for a specific zone or distance.
    """
    zone:       str
    attempts:   int = 0
    makes:      int = 0

    @property
    def pct(self) -> float:
        if self.attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.makes / self.attempts, 3)

    @property
    def pct_display(self) -> str:
        return f"{self.pct * 100:.1f}%"

    def record_attempt(self, made: bool) -> None:
        self.attempts += 1
        if made:
            self.makes += 1

    def to_dict(self) -> Dict:
        return {
            "zone":     self.zone,
            "attempts": self.attempts,
            "makes":    self.makes,
            "pct":      self.pct,
        }


# =============================================================================
# PLAYER STATS DATACLASS
# =============================================================================

@dataclass
class PlayerStats:
    """
    Complete per-player statistics container.
    Updated in real-time as events are processed.
    """
    track_id:       int
    jersey_number:  Optional[str]  = None
    team_id:        Optional[int]  = None
    name:           Optional[str]  = None

    # ── Scoring ──────────────────────────────────────────────────────────────
    points:             int = 0
    fg_makes:           int = 0
    fg_attempts:        int = 0
    fg3_makes:          int = 0
    fg3_attempts:       int = 0
    ft_makes:           int = 0
    ft_attempts:        int = 0

    # ── Rebounding ───────────────────────────────────────────────────────────
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0

    # ── Playmaking ───────────────────────────────────────────────────────────
    assists:            int = 0
    turnovers:          int = 0
    passes:             int = 0
    pass_attempts:      int = 0

    # ── Defense ──────────────────────────────────────────────────────────────
    steals:             int = 0
    blocks:             int = 0
    fouls:              int = 0

    # ── Streaks / Hot hand ───────────────────────────────────────────────────
    consecutive_makes:  int = 0
    consecutive_misses: int = 0
    is_hot:             bool = False

    # ── Highlight score ──────────────────────────────────────────────────────
    highlight_score:    float = 0.0

    # ── Zone shooting ────────────────────────────────────────────────────────
    zone_shooting:      Dict[str, ShootingLine] = field(default_factory=dict)

    # ── Recent attempts (for hot hand) ───────────────────────────────────────
    recent_attempts:    Deque[bool] = field(
        default_factory=lambda: deque(maxlen=HOT_HAND_WINDOW)
    )

    # ── Frame tracking ───────────────────────────────────────────────────────
    first_seen_frame:   int   = 0
    last_seen_frame:    int   = 0
    frames_on_court:    int   = 0

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at:         float = field(default_factory=time.time)
    updated_at:         float = field(default_factory=time.time)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def total_rebounds(self) -> int:
        return self.offensive_rebounds + self.defensive_rebounds

    @property
    def fg_pct(self) -> float:
        if self.fg_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.fg_makes / self.fg_attempts, 3)

    @property
    def fg3_pct(self) -> float:
        if self.fg3_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.fg3_makes / self.fg3_attempts, 3)

    @property
    def ft_pct(self) -> float:
        if self.ft_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.ft_makes / self.ft_attempts, 3)

    @property
    def effective_fg_pct(self) -> float:
        """eFG% = (FGM + 0.5 * 3PM) / FGA"""
        if self.fg_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(
            (self.fg_makes + 0.5 * self.fg3_makes) / self.fg_attempts, 3
        )

    @property
    def true_shooting_pct(self) -> float:
        """TS% = PTS / (2 * (FGA + 0.44 * FTA))"""
        denom = 2 * (self.fg_attempts + 0.44 * self.ft_attempts)
        if denom < 0.001:
            return 0.0
        return round(self.points / denom, 3)

    @property
    def assist_to_turnover(self) -> float:
        if self.turnovers == 0:
            return float(self.assists)
        return round(self.assists / self.turnovers, 2)

    @property
    def per_rating(self) -> float:
        """Simplified Player Efficiency Rating."""
        return round(
            self.points          * PER_WEIGHTS["points"]
            + self.total_rebounds * PER_WEIGHTS["rebounds"]
            + self.assists        * PER_WEIGHTS["assists"]
            + self.steals         * PER_WEIGHTS["steals"]
            + self.blocks         * PER_WEIGHTS["blocks"]
            + self.turnovers      * PER_WEIGHTS["turnovers"]
            + self.fouls          * PER_WEIGHTS["fouls"]
            + self.fg_attempts    * PER_WEIGHTS["fg_attempts"],
            2,
        )

    def _update_hot_hand(self, made: bool) -> None:
        self.recent_attempts.append(made)
        if made:
            self.consecutive_makes  += 1
            self.consecutive_misses  = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_makes   = 0

        makes_in_window = sum(self.recent_attempts)
        self.is_hot = (
            len(self.recent_attempts) >= HOT_HAND_WINDOW
            and makes_in_window >= HOT_HAND_MIN_MAKES
        )

    def record_fg_attempt(
        self,
        made:       bool,
        is_three:   bool  = False,
        zone:       Optional[str] = None,
    ) -> None:
        self.fg_attempts += 1
        if is_three:
            self.fg3_attempts += 1
        if made:
            self.fg_makes += 1
            if is_three:
                self.fg3_makes += 1
            self.points += 3 if is_three else 2
        self._update_hot_hand(made)

        if zone:
            if zone not in self.zone_shooting:
                self.zone_shooting[zone] = ShootingLine(zone)
            self.zone_shooting[zone].record_attempt(made)

    def record_ft_attempt(self, made: bool) -> None:
        self.ft_attempts += 1
        if made:
            self.ft_makes += 1
            self.points   += 1

    def add_highlight_score(self, delta: float) -> None:
        self.highlight_score = round(self.highlight_score + delta, 2)

    def to_dict(self) -> Dict:
        return {
            "track_id":           self.track_id,
            "jersey_number":      self.jersey_number,
            "team_id":            self.team_id,
            "name":               self.name,
            # Scoring
            "points":             self.points,
            "fg_makes":           self.fg_makes,
            "fg_attempts":        self.fg_attempts,
            "fg_pct":             self.fg_pct,
            "fg3_makes":          self.fg3_makes,
            "fg3_attempts":       self.fg3_attempts,
            "fg3_pct":            self.fg3_pct,
            "ft_makes":           self.ft_makes,
            "ft_attempts":        self.ft_attempts,
            "ft_pct":             self.ft_pct,
            "efg_pct":            self.effective_fg_pct,
            "ts_pct":             self.true_shooting_pct,
            # Rebounding
            "offensive_rebounds": self.offensive_rebounds,
            "defensive_rebounds": self.defensive_rebounds,
            "total_rebounds":     self.total_rebounds,
            # Playmaking
            "assists":            self.assists,
            "turnovers":          self.turnovers,
            "ast_to":             self.assist_to_turnover,
            "passes":             self.passes,
            # Defense
            "steals":             self.steals,
            "blocks":             self.blocks,
            "fouls":              self.fouls,
            # Streaks
            "consecutive_makes":  self.consecutive_makes,
            "is_hot":             self.is_hot,
            # Efficiency
            "per_rating":         self.per_rating,
            "highlight_score":    self.highlight_score,
            # Zone shooting
            "zone_shooting":      {
                k: v.to_dict()
                for k, v in self.zone_shooting.items()
            },
        }

    def box_score_row(self) -> Dict:
        """Compact box score row for display."""
        return {
            "#":    self.jersey_number or str(self.track_id),
            "PTS":  self.points,
            "REB":  self.total_rebounds,
            "AST":  self.assists,
            "STL":  self.steals,
            "BLK":  self.blocks,
            "TO":   self.turnovers,
            "FG":   f"{self.fg_makes}/{self.fg_attempts}",
            "3P":   f"{self.fg3_makes}/{self.fg3_attempts}",
            "FT":   f"{self.ft_makes}/{self.ft_attempts}",
            "PER":  self.per_rating,
        }
        # =============================================================================
# TEAM STATS
# =============================================================================

@dataclass
class TeamStats:
    """
    Aggregated team-level statistics.
    Auto-computed from player stats when refresh() is called.
    """
    team_id:    int
    team_name:  Optional[str] = None
    side:       TeamSide      = TeamSide.HOME

    # ── Scoring ──────────────────────────────────────────────────────────────
    points:             int   = 0
    fg_makes:           int   = 0
    fg_attempts:        int   = 0
    fg3_makes:          int   = 0
    fg3_attempts:       int   = 0
    ft_makes:           int   = 0
    ft_attempts:        int   = 0

    # ── Rebounding ───────────────────────────────────────────────────────────
    offensive_rebounds: int   = 0
    defensive_rebounds: int   = 0

    # ── Playmaking ───────────────────────────────────────────────────────────
    assists:            int   = 0
    turnovers:          int   = 0
    passes:             int   = 0

    # ── Defense ──────────────────────────────────────────────────────────────
    steals:             int   = 0
    blocks:             int   = 0
    fouls:              int   = 0

    # ── Momentum ─────────────────────────────────────────────────────────────
    current_run:        int   = 0   # unanswered points
    largest_lead:       int   = 0
    lead_changes:       int   = 0

    # ── Scoring runs ─────────────────────────────────────────────────────────
    scoring_runs:       List[Dict] = field(default_factory=list)

    # ── Highlight score ──────────────────────────────────────────────────────
    highlight_score:    float = 0.0

    # ── Computed ─────────────────────────────────────────────────────────────

    @property
    def total_rebounds(self) -> int:
        return self.offensive_rebounds + self.defensive_rebounds

    @property
    def fg_pct(self) -> float:
        if self.fg_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.fg_makes / self.fg_attempts, 3)

    @property
    def fg3_pct(self) -> float:
        if self.fg3_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.fg3_makes / self.fg3_attempts, 3)

    @property
    def ft_pct(self) -> float:
        if self.ft_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(self.ft_makes / self.ft_attempts, 3)

    @property
    def effective_fg_pct(self) -> float:
        if self.fg_attempts < MIN_ATTEMPTS_FOR_PCT:
            return 0.0
        return round(
            (self.fg_makes + 0.5 * self.fg3_makes) / self.fg_attempts, 3
        )

    @property
    def assist_to_turnover(self) -> float:
        if self.turnovers == 0:
            return float(self.assists)
        return round(self.assists / self.turnovers, 2)

    @property
    def points_per_possession(self) -> float:
        possessions = max(
            1,
            self.fg_attempts
            - self.offensive_rebounds
            + self.turnovers
            + 0.44 * self.ft_attempts,
        )
        return round(self.points / possessions, 3)

    def refresh_from_players(
        self,
        players: List[PlayerStats],
    ) -> None:
        """Recompute all team totals from player list."""
        team_players = [p for p in players if p.team_id == self.team_id]

        self.points             = sum(p.points             for p in team_players)
        self.fg_makes           = sum(p.fg_makes           for p in team_players)
        self.fg_attempts        = sum(p.fg_attempts        for p in team_players)
        self.fg3_makes          = sum(p.fg3_makes          for p in team_players)
        self.fg3_attempts       = sum(p.fg3_attempts       for p in team_players)
        self.ft_makes           = sum(p.ft_makes           for p in team_players)
        self.ft_attempts        = sum(p.ft_attempts        for p in team_players)
        self.offensive_rebounds = sum(p.offensive_rebounds for p in team_players)
        self.defensive_rebounds = sum(p.defensive_rebounds for p in team_players)
        self.assists            = sum(p.assists            for p in team_players)
        self.turnovers          = sum(p.turnovers          for p in team_players)
        self.passes             = sum(p.passes             for p in team_players)
        self.steals             = sum(p.steals             for p in team_players)
        self.blocks             = sum(p.blocks             for p in team_players)
        self.fouls              = sum(p.fouls              for p in team_players)
        self.highlight_score    = sum(p.highlight_score    for p in team_players)

    def to_dict(self) -> Dict:
        return {
            "team_id":            self.team_id,
            "team_name":          self.team_name,
            "points":             self.points,
            "fg_makes":           self.fg_makes,
            "fg_attempts":        self.fg_attempts,
            "fg_pct":             self.fg_pct,
            "fg3_makes":          self.fg3_makes,
            "fg3_attempts":       self.fg3_attempts,
            "fg3_pct":            self.fg3_pct,
            "ft_makes":           self.ft_makes,
            "ft_attempts":        self.ft_attempts,
            "ft_pct":             self.ft_pct,
            "efg_pct":            self.effective_fg_pct,
            "total_rebounds":     self.total_rebounds,
            "offensive_rebounds": self.offensive_rebounds,
            "defensive_rebounds": self.defensive_rebounds,
            "assists":            self.assists,
            "turnovers":          self.turnovers,
            "ast_to":             self.assist_to_turnover,
            "steals":             self.steals,
            "blocks":             self.blocks,
            "fouls":              self.fouls,
            "current_run":        self.current_run,
            "largest_lead":       self.largest_lead,
            "lead_changes":       self.lead_changes,
            "ppp":                self.points_per_possession,
            "highlight_score":    self.highlight_score,
        }


# =============================================================================
# GAME STATS
# =============================================================================

@dataclass
class GameStats:
    """
    Top-level game statistics container.
    Holds team stats, player stats, and game-level metadata.
    """
    game_id:        str             = field(
        default_factory=lambda: f"game_{int(time.time())}"
    )
    start_time:     float           = field(default_factory=time.time)
    end_time:       Optional[float] = None
    total_frames:   int             = 0
    live_frames:    int             = 0
    dead_frames:    int             = 0

    # Teams
    teams:          Dict[int, TeamStats]   = field(default_factory=dict)

    # Players
    players:        Dict[int, PlayerStats] = field(default_factory=dict)

    # Score history  [(frame_id, team_id, new_score)]
    score_history:  List[Tuple[int, int, int]] = field(default_factory=list)

    # Event counts
    event_counts:   Dict[str, int]  = field(default_factory=lambda: defaultdict(int))

    # Momentum tracking
    momentum_log:   List[Dict]      = field(default_factory=list)

    # Highlight moments
    highlight_moments: List[Dict]   = field(default_factory=list)

    @property
    def score(self) -> Dict[int, int]:
        return {tid: t.points for tid, t in self.teams.items()}

    @property
    def live_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.live_frames / self.total_frames, 3)

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    def get_player(self, track_id: int) -> Optional[PlayerStats]:
        return self.players.get(track_id)

    def get_team(self, team_id: int) -> Optional[TeamStats]:
        return self.teams.get(team_id)

    def to_dict(self) -> Dict:
        return {
            "game_id":         self.game_id,
            "duration_sec":    self.duration_seconds,
            "total_frames":    self.total_frames,
            "live_frames":     self.live_frames,
            "dead_frames":     self.dead_frames,
            "live_ratio":      self.live_ratio,
            "score":           self.score,
            "teams":           {
                str(k): v.to_dict() for k, v in self.teams.items()
            },
            "players":         {
                str(k): v.to_dict() for k, v in self.players.items()
            },
            "event_counts":    dict(self.event_counts),
            "highlight_moments": self.highlight_moments,
        }
        # =============================================================================
# MOMENTUM TRACKER
# =============================================================================

class MomentumTracker:
    """
    Tracks game momentum for each team.

    Momentum is a rolling score based on recent events.
    Positive events increase momentum; negative decrease it.
    """

    _EVENT_MOMENTUM: Dict[EventType, float] = {
        EventType.SCORE_3PT:    +3.0,
        EventType.SCORE_2PT:    +2.0,
        EventType.FREE_THROW_MADE: +1.0,
        EventType.REBOUND_OFF:  +1.5,
        EventType.REBOUND_DEF:  +1.0,
        EventType.STEAL:        +2.5,
        EventType.BLOCK:        +2.5,
        EventType.ASSIST:       +1.0,
        EventType.FAST_BREAK:   +2.0,
        EventType.TURNOVER:     -1.5,
        EventType.FOUL:         -0.5,
        EventType.SHOT_MISS:    -0.5,
    }

    def __init__(
        self,
        team_ids:       List[int],
        window_events:  int = MOMENTUM_WINDOW_EVENTS,
        decay:          float = 0.92,
    ):
        self.team_ids      = team_ids
        self.window_events = window_events
        self.decay         = decay

        self._scores: Dict[int, float] = {t: 0.0 for t in team_ids}
        self._history: Dict[int, Deque[float]] = {
            t: deque(maxlen=window_events) for t in team_ids
        }
        self._run_points: Dict[int, int] = {t: 0 for t in team_ids}
        self._last_scorer: Optional[int] = None

    def update(self, event: GameEvent) -> Dict[int, float]:
        """
        Feed one event; returns current momentum dict {team_id: score}.
        """
        team_id = event.primary_team_id
        delta   = self._EVENT_MOMENTUM.get(event.event_type, 0.0)

        if team_id is not None and delta != 0.0:
            # Apply decay to all teams
            for tid in self.team_ids:
                self._scores[tid] *= self.decay

            self._scores[team_id] = round(
                self._scores.get(team_id, 0.0) + delta, 3
            )
            self._history[team_id].append(delta)

        return dict(self._scores)

    def update_run(self, scoring_team_id: int, points: int) -> None:
        """Track unanswered scoring runs."""
        if scoring_team_id == self._last_scorer or self._last_scorer is None:
            self._run_points[scoring_team_id] = (
                self._run_points.get(scoring_team_id, 0) + points
            )
        else:
            # Run broken — reset opposing team's run
            for tid in self.team_ids:
                if tid != scoring_team_id:
                    self._run_points[tid] = 0
            self._run_points[scoring_team_id] = (
                self._run_points.get(scoring_team_id, 0) + points
            )
        self._last_scorer = scoring_team_id

    def get_run(self, team_id: int) -> int:
        return self._run_points.get(team_id, 0)

    def get_momentum(self, team_id: int) -> float:
        return round(self._scores.get(team_id, 0.0), 3)

    def leading_team(self) -> Optional[int]:
        if not self._scores:
            return None
        return max(self._scores, key=self._scores.get)

    def reset(self) -> None:
        self._scores    = {t: 0.0 for t in self.team_ids}
        self._run_points = {t: 0   for t in self.team_ids}
        self._last_scorer = None


# =============================================================================
# HIGHLIGHT SCORER
# =============================================================================

class HighlightScorer:
    """
    Assigns highlight scores to events and identifies top moments.
    Used to automatically flag best clips for highlight reels.
    """

    def __init__(
        self,
        weights:        Dict[EventType, float] = None,
        combo_bonus:    float = 2.0,
        combo_window:   int   = 5,
    ):
        self.weights      = weights or HIGHLIGHT_WEIGHTS
        self.combo_bonus  = combo_bonus
        self.combo_window = combo_window

        self._recent_events: Deque[GameEvent] = deque(maxlen=combo_window)
        self._top_moments:   List[Dict]       = []

    def score_event(self, event: GameEvent) -> float:
        """
        Returns highlight score for a single event.
        Includes combo bonus if multiple high-value events occurred recently.
        """
        base = self.weights.get(event.event_type, 0.0)
        if base <= 0:
            return base

        # Combo bonus: multiple positive events in window
        recent_positive = sum(
            1 for e in self._recent_events
            if self.weights.get(e.event_type, 0.0) > 0
        )
        combo = self.combo_bonus * max(0, recent_positive - 1)

        # Hot hand bonus
        if event.event_type in (EventType.SCORE_2PT, EventType.SCORE_3PT):
            # Will be set by stats engine if player is hot
            hot_bonus = event.meta.get("hot_hand_bonus", 0.0)
        else:
            hot_bonus = 0.0

        total = base + combo + hot_bonus
        self._recent_events.append(event)
        return round(total, 2)

    def record_moment(
        self,
        event:  GameEvent,
        score:  float,
        frame_id: int,
    ) -> None:
        """Store high-value moment for highlight reel generation."""
        if score >= 5.0:
            self._top_moments.append({
                "frame_id":    frame_id,
                "event_type":  event.event_type.value,
                "score":       score,
                "track_id":    event.primary_track_id,
                "jersey":      event.primary_jersey,
                "team_id":     event.primary_team_id,
                "ball_pos":    event.ball_position,
            })
            # Keep only top 50 moments
            self._top_moments.sort(key=lambda x: x["score"], reverse=True)
            self._top_moments = self._top_moments[:50]

    @property
    def top_moments(self) -> List[Dict]:
        return list(self._top_moments)

    def reset(self) -> None:
        self._recent_events.clear()
        self._top_moments.clear()


# =============================================================================
# SHOT CHART BUILDER
# =============================================================================

class ShotChartBuilder:
    """
    Builds a shot chart mapping shot locations to makes/misses.
    Stores normalized (0–1) court coordinates.
    """

    def __init__(
        self,
        frame_width:  int = 1280,
        frame_height: int = 720,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self._shots: List[Dict] = []

    def record_shot(
        self,
        track_id:   int,
        team_id:    Optional[int],
        jersey:     Optional[str],
        position:   Tuple[float, float],
        made:       bool,
        is_three:   bool,
        zone:       Optional[str],
        frame_id:   int,
    ) -> None:
        nx = round(position[0] / self.frame_width,  4)
        ny = round(position[1] / self.frame_height, 4)
        self._shots.append({
            "track_id":  track_id,
            "team_id":   team_id,
            "jersey":    jersey,
            "x":         nx,
            "y":         ny,
            "made":      made,
            "is_three":  is_three,
            "zone":      zone,
            "frame_id":  frame_id,
        })

    def get_shots(
        self,
        track_id: Optional[int] = None,
        team_id:  Optional[int] = None,
    ) -> List[Dict]:
        shots = self._shots
        if track_id is not None:
            shots = [s for s in shots if s["track_id"] == track_id]
        if team_id is not None:
            shots = [s for s in shots if s["team_id"] == team_id]
        return shots

    def zone_summary(
        self,
        track_id: Optional[int] = None,
        team_id:  Optional[int] = None,
    ) -> Dict[str, Dict]:
        shots = self.get_shots(track_id, team_id)
        summary: Dict[str, Dict] = defaultdict(
            lambda: {"attempts": 0, "makes": 0, "pct": 0.0}
        )
        for s in shots:
            z = s["zone"] or "unknown"
            summary[z]["attempts"] += 1
            if s["made"]:
                summary[z]["makes"] += 1
        for z, d in summary.items():
            d["pct"] = (
                round(d["makes"] / d["attempts"], 3)
                if d["attempts"] > 0 else 0.0
            )
        return dict(summary)

    def to_list(self) -> List[Dict]:
        return list(self._shots)

    def reset(self) -> None:
        self._shots.clear()
        # =============================================================================
# STATS ENGINE  —  Master orchestrator
# =============================================================================

class StatsEngine:
    """
    Central statistics engine for Purple Box Sports.

    Consumes GameEvent objects and maintains:
      - Per-player statistics (PlayerStats)
      - Per-team statistics (TeamStats)
      - Game-level statistics (GameStats)
      - Momentum tracking
      - Highlight scoring
      - Shot chart data

    Usage:
        engine = StatsEngine(team_ids=[0, 1])

        # Each frame:
        events = event_engine.process_frame(...)
        for event in events:
            engine.process_event(event, frame_id)

        # Get snapshot:
        snapshot = engine.snapshot()
        box_score = engine.box_score()
    """

    def __init__(
        self,
        team_ids:       List[int]           = None,
        team_names:     Dict[int, str]      = None,
        frame_width:    int                 = 1280,
        frame_height:   int                 = 720,
        game_id:        Optional[str]       = None,
        callbacks:      Optional[Dict[str, List[Callable]]] = None,
    ):
        self.team_ids   = team_ids or [0, 1]
        self.team_names = team_names or {}
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # ── Game stats container ─────────────────────────────────────────────
        self.game_stats = GameStats(
            game_id=game_id or f"game_{int(time.time())}"
        )

        # ── Initialize team stats ────────────────────────────────────────────
        for i, tid in enumerate(self.team_ids):
            self.game_stats.teams[tid] = TeamStats(
                team_id   = tid,
                team_name = self.team_names.get(tid),
                side      = TeamSide.HOME if i == 0 else TeamSide.AWAY,
            )

        # ── Sub-systems ──────────────────────────────────────────────────────
        self.momentum_tracker  = MomentumTracker(team_ids=self.team_ids)
        self.highlight_scorer  = HighlightScorer()
        self.shot_chart        = ShotChartBuilder(frame_width, frame_height)

        # ── Callbacks ────────────────────────────────────────────────────────
        self._callbacks: Dict[str, List[Callable]] = callbacks or {}

        # ── Internal state ───────────────────────────────────────────────────
        self._current_frame:    int           = 0
        self._last_lead_team:   Optional[int] = None
        self._pending_shot:     Optional[GameEvent] = None   # last SHOT_ATTEMPT

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def process_event(
        self,
        event:    GameEvent,
        frame_id: int,
    ) -> None:
        """
        Process a single GameEvent and update all statistics.
        Call once per event, in frame order.
        """
        self._current_frame = frame_id
        self.game_stats.event_counts[event.event_type.value] += 1

        # ── Route to handler ─────────────────────────────────────────────────
        handler = self._HANDLERS.get(event.event_type)
        if handler:
            handler(self, event, frame_id)

        # ── Momentum ─────────────────────────────────────────────────────────
        momentum = self.momentum_tracker.update(event)

        # ── Highlight score ──────────────────────────────────────────────────
        hl_score = self.highlight_scorer.score_event(event)
        if hl_score > 0 and event.primary_track_id is not None:
            player = self._get_or_create_player(
                event.primary_track_id,
                event.primary_jersey,
                event.primary_team_id,
            )
            player.add_highlight_score(hl_score)
            self.highlight_scorer.record_moment(event, hl_score, frame_id)

        # ── Refresh team totals ──────────────────────────────────────────────
        self._refresh_teams()

        # ── Lead changes ─────────────────────────────────────────────────────
        self._check_lead_change()

        # ── Callbacks ────────────────────────────────────────────────────────
        self._fire(event.event_type.value, event, frame_id, momentum)

    def record_event(self, ev: Union[GameEvent, Dict], frame_id: int = 0) -> None:
        """
        Convenience wrapper around process_event.
        Accepts either a GameEvent or a plain dict (from pipeline).
        """
        if isinstance(ev, GameEvent):
            self.process_event(ev, frame_id or ev.frame_id)
        elif isinstance(ev, dict):
            # Convert dict → GameEvent
            etype_str = ev.get("type", "unknown")
            try:
                etype = EventType(etype_str)
            except ValueError:
                etype = EventType.UNKNOWN
            game_ev = GameEvent(
                event_type       = etype,
                frame_id         = ev.get("frame", ev.get("frame_idx", frame_id)),
                primary_track_id = ev.get("player") if isinstance(ev.get("player"), int) else None,
                primary_jersey   = ev.get("jersey", ""),
                primary_team_id  = ev.get("team_id"),
                ball_position    = ev.get("ball_position"),
                court_zone       = ev.get("court_zone"),
                meta             = {k: v for k, v in ev.items() if k not in (
                    "type", "frame", "frame_idx", "player", "jersey",
                    "team_id", "ball_position", "court_zone")},
            )
            self.process_event(game_ev, game_ev.frame_id)
        else:
            logger.warning("record_event: unsupported type %s", type(ev))

    def get_summary(self) -> Dict:
        """Return game_stats.to_dict() — convenience for pipeline."""
        self._refresh_teams()
        return self.game_stats.to_dict()

    def process_frame_meta(
        self,
        frame_id: int,
        is_dead:  bool,
    ) -> None:
        """
        Call every frame (even if no events) to track live/dead ratio.
        """
        self.game_stats.total_frames += 1
        if is_dead:
            self.game_stats.dead_frames += 1
        else:
            self.game_stats.live_frames += 1
        self._current_frame = frame_id

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _handle_score_2pt(self, event: GameEvent, frame_id: int) -> None:
        self._handle_score(event, frame_id, points=2, is_three=False)

    def _handle_score_3pt(self, event: GameEvent, frame_id: int) -> None:
        self._handle_score(event, frame_id, points=3, is_three=True)

    def _handle_score(
        self,
        event:    GameEvent,
        frame_id: int,
        points:   int,
        is_three: bool,
    ) -> None:
        if event.primary_track_id is None:
            return

        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )

        # Hot hand bonus for highlight
        if player.is_hot:
            event.meta["hot_hand_bonus"] = 3.0

        player.record_fg_attempt(
            made=True, is_three=is_three, zone=event.court_zone
        )

        # Shot chart
        if event.ball_position:
            self.shot_chart.record_shot(
                track_id  = event.primary_track_id,
                team_id   = event.primary_team_id,
                jersey    = event.primary_jersey,
                position  = event.ball_position,
                made      = True,
                is_three  = is_three,
                zone      = event.court_zone,
                frame_id  = frame_id,
            )

        # Score history
        if event.primary_team_id is not None:
            team = self.game_stats.teams.get(event.primary_team_id)
            if team:
                self.game_stats.score_history.append(
                    (frame_id, event.primary_team_id, team.points + points)
                )
                self.momentum_tracker.update_run(
                    event.primary_team_id, points
                )

        # Clear pending shot
        self._pending_shot = None

    def _handle_free_throw_made(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.record_ft_attempt(made=True)

    def _handle_free_throw_miss(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.record_ft_attempt(made=False)

    def _handle_shot_attempt(self, event: GameEvent, frame_id: int) -> None:
        """Store pending shot; resolved on score or miss."""
        self._pending_shot = event

    def _handle_shot_miss(self, event: GameEvent, frame_id: int) -> None:
        ref = self._pending_shot or event
        if ref.primary_track_id is None:
            self._pending_shot = None
            return

        player = self._get_or_create_player(
            ref.primary_track_id,
            ref.primary_jersey,
            ref.primary_team_id,
        )
        is_three = ref.is_three_point or (
            ref.court_zone in ("three_point",) if ref.court_zone else False
        )
        player.record_fg_attempt(
            made=False, is_three=is_three, zone=ref.court_zone
        )
        if ref.ball_position:
            self.shot_chart.record_shot(
                track_id  = ref.primary_track_id,
                team_id   = ref.primary_team_id,
                jersey    = ref.primary_jersey,
                position  = ref.ball_position,
                made      = False,
                is_three  = is_three,
                zone      = ref.court_zone,
                frame_id  = frame_id,
            )
        self._pending_shot = None

    def _handle_rebound_off(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.offensive_rebounds += 1

    def _handle_rebound_def(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.defensive_rebounds += 1

    def _handle_steal(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        stealer = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        stealer.steals += 1

        if event.secondary_track_id is not None:
            victim = self._get_or_create_player(
                event.secondary_track_id,
                event.secondary_jersey,
                event.secondary_team_id,
            )
            victim.turnovers += 1

    def _handle_turnover(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.turnovers += 1

    def _handle_assist(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.assists += 1

    def _handle_pass(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        player = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        player.passes += 1

    def _handle_block(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        blocker = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        blocker.blocks += 1

    def _handle_foul(self, event: GameEvent, frame_id: int) -> None:
        if event.primary_track_id is None:
            return
        fouler = self._get_or_create_player(
            event.primary_track_id,
            event.primary_jersey,
            event.primary_team_id,
        )
        fouler.fouls += 1

    def _handle_fast_break(self, event: GameEvent, frame_id: int) -> None:
        # Fast break is recorded in highlight score only
        pass

    def _handle_dead_ball(self, event: GameEvent, frame_id: int) -> None:
        pass   # tracked via process_frame_meta

    def _handle_live_ball(self, event: GameEvent, frame_id: int) -> None:
        pass

    def _handle_out_of_bounds(self, event: GameEvent, frame_id: int) -> None:
        pass

    # ── Handler dispatch table ────────────────────────────────────────────────
    _HANDLERS: Dict[EventType, Callable] = {
        EventType.SCORE_2PT:        _handle_score_2pt,
        EventType.SCORE_3PT:        _handle_score_3pt,
        EventType.FREE_THROW_MADE:  _handle_free_throw_made,
        EventType.FREE_THROW_MISS:  _handle_free_throw_miss,
        EventType.SHOT_ATTEMPT:     _handle_shot_attempt,
        EventType.SHOT_MISS:        _handle_shot_miss,
        EventType.REBOUND_OFF:      _handle_rebound_off,
        EventType.REBOUND_DEF:      _handle_rebound_def,
        EventType.STEAL:            _handle_steal,
        EventType.TURNOVER:         _handle_turnover,
        EventType.ASSIST:           _handle_assist,
        EventType.PASS:             _handle_pass,
        EventType.BLOCK:            _handle_block,
        EventType.FOUL:             _handle_foul,
        EventType.FAST_BREAK:       _handle_fast_break,
        EventType.DEAD_BALL:        _handle_dead_ball,
        EventType.LIVE_BALL:        _handle_live_ball,
        EventType.OUT_OF_BOUNDS:    _handle_out_of_bounds,
    }
        # =========================================================================
    # QUERIES & OUTPUT
    # =========================================================================

    def snapshot(self) -> Dict:
        """
        Full real-time stats snapshot.
        Returns everything needed for a live dashboard.
        """
        self._refresh_teams()
        return {
            "frame_id":   self._current_frame,
            "game":       self.game_stats.to_dict(),
            "momentum":   {
                str(tid): self.momentum_tracker.get_momentum(tid)
                for tid in self.team_ids
            },
            "shot_chart": self.shot_chart.to_list(),
            "highlights": self.highlight_scorer.top_moments,
        }

    def box_score(self) -> Dict:
        """
        Classic box score format:
        {team_id: [player_row, ...], "team_totals": {...}}
        """
        self._refresh_teams()
        result: Dict = {}
        for tid in self.team_ids:
            team_players = [
                p.box_score_row()
                for p in self.game_stats.players.values()
                if p.team_id == tid
            ]
            team_players.sort(key=lambda r: r["PTS"], reverse=True)
            result[str(tid)] = team_players

        result["team_totals"] = {
            str(tid): self.game_stats.teams[tid].to_dict()
            for tid in self.team_ids
            if tid in self.game_stats.teams
        }
        return result

    def player_stats(
        self,
        track_id: int,
    ) -> Optional[Dict]:
        """Return full stats dict for one player."""
        p = self.game_stats.players.get(track_id)
        return p.to_dict() if p else None

    def team_stats(
        self,
        team_id: int,
    ) -> Optional[Dict]:
        """Return full stats dict for one team."""
        t = self.game_stats.teams.get(team_id)
        return t.to_dict() if t else None

    def score(self) -> Dict[int, int]:
        """Current score {team_id: points}."""
        return {
            tid: t.points
            for tid, t in self.game_stats.teams.items()
        }

    def leading_team(self) -> Optional[int]:
        sc = self.score()
        if not sc:
            return None
        return max(sc, key=sc.get)

    def top_scorer(
        self,
        team_id: Optional[int] = None,
    ) -> Optional[PlayerStats]:
        players = list(self.game_stats.players.values())
        if team_id is not None:
            players = [p for p in players if p.team_id == team_id]
        if not players:
            return None
        return max(players, key=lambda p: p.points)

    def top_rebounder(
        self,
        team_id: Optional[int] = None,
    ) -> Optional[PlayerStats]:
        players = list(self.game_stats.players.values())
        if team_id is not None:
            players = [p for p in players if p.team_id == team_id]
        if not players:
            return None
        return max(players, key=lambda p: p.total_rebounds)

    def top_passer(
        self,
        team_id: Optional[int] = None,
    ) -> Optional[PlayerStats]:
        players = list(self.game_stats.players.values())
        if team_id is not None:
            players = [p for p in players if p.team_id == team_id]
        if not players:
            return None
        return max(players, key=lambda p: p.assists)

    def hot_players(self) -> List[PlayerStats]:
        """Return all players currently on a hot streak."""
        return [
            p for p in self.game_stats.players.values()
            if p.is_hot
        ]

    def highlight_reel_frames(
        self,
        top_n: int = 10,
    ) -> List[int]:
        """Return frame IDs of the top N highlight moments."""
        moments = self.highlight_scorer.top_moments[:top_n]
        return [m["frame_id"] for m in moments]

    def export_json(
        self,
        path: str,
        indent: int = 2,
    ) -> None:
        """Write full stats snapshot to JSON file."""
        data = self.snapshot()
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Stats exported → {path}")

    def export_box_score_json(
        self,
        path: str,
        indent: int = 2,
    ) -> None:
        """Write box score to JSON file."""
        data = self.box_score()
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Box score exported → {path}")

    def print_box_score(self) -> None:
        """Pretty-print box score to stdout."""
        bs = self.box_score()
        for tid in self.team_ids:
            name = self.team_names.get(tid, f"Team {tid}")
            print(f"\n{'═'*60}")
            print(f"  {name}  —  {self.score().get(tid, 0)} pts")
            print(f"{'═'*60}")
            header = f"{'#':>4} {'PTS':>4} {'REB':>4} {'AST':>4} "
            header += f"{'STL':>4} {'BLK':>4} {'TO':>4} "
            header += f"{'FG':>7} {'3P':>7} {'FT':>7} {'PER':>6}"
            print(header)
            print("─" * 60)
            for row in bs.get(str(tid), []):
                line = (
                    f"{str(row['#']):>4} "
                    f"{row['PTS']:>4} "
                    f"{row['REB']:>4} "
                    f"{row['AST']:>4} "
                    f"{row['STL']:>4} "
                    f"{row['BLK']:>4} "
                    f"{row['TO']:>4} "
                    f"{row['FG']:>7} "
                    f"{row['3P']:>7} "
                    f"{row['FT']:>7} "
                    f"{row['PER']:>6.1f}"
                )
                print(line)

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _get_or_create_player(
        self,
        track_id:     int,
        jersey:       Optional[str],
        team_id:      Optional[int],
    ) -> PlayerStats:
        # ── Identity resolution ───────────────────────────────────────────────
        # A player's tracker ID can change when they are occluded and re-
        # acquired, which would fragment their stats into multiple rows. To
        # keep stats unified we resolve every (track_id, jersey) pair to a
        # single canonical player key:
        #   - If we know a jersey (per team), that IS the identity.
        #   - Track IDs are aliased onto the canonical key once their jersey
        #     is known, so future events under either the track_id or the
        #     jersey land on the same PlayerStats row.
        if not hasattr(self, "_jersey_key"):
            self._jersey_key: Dict[Tuple[Optional[int], str], int] = {}
            self._alias:      Dict[int, int] = {}

        # Follow any existing alias first
        canonical = self._alias.get(track_id, track_id)

        if jersey:
            jkey = (team_id, jersey)
            if jkey in self._jersey_key:
                # This jersey already has a canonical row — merge onto it.
                canonical_j = self._jersey_key[jkey]
                if canonical != canonical_j:
                    self._merge_players(canonical, canonical_j)
                    self._alias[track_id] = canonical_j
                    self._alias[canonical] = canonical_j
                    canonical = canonical_j
            else:
                # First time we see this jersey — bind it to the current row.
                self._jersey_key[jkey] = canonical
                self._alias[track_id] = canonical

        if canonical not in self.game_stats.players:
            self.game_stats.players[canonical] = PlayerStats(
                track_id      = canonical,
                jersey_number = jersey,
                team_id       = team_id,
                first_seen_frame = self._current_frame,
            )
        else:
            p = self.game_stats.players[canonical]
            if jersey and not p.jersey_number:
                p.jersey_number = jersey
            if team_id is not None and p.team_id is None:
                p.team_id = team_id

        p = self.game_stats.players[canonical]
        p.last_seen_frame = self._current_frame
        p.updated_at      = time.time()
        return p

    def _merge_players(self, src_id: int, dst_id: int) -> None:
        """
        Merge stats accumulated under a stale track_id (src) into the
        canonical jersey row (dst). Called when OCR later reveals two
        track IDs are the same player.
        """
        src = self.game_stats.players.get(src_id)
        dst = self.game_stats.players.get(dst_id)
        if src is None or dst is None or src is dst:
            return

        # Sum the additive counting stats
        for attr in (
            "points", "fg_makes", "fg_attempts", "fg3_makes", "fg3_attempts",
            "ft_makes", "ft_attempts", "offensive_rebounds",
            "defensive_rebounds", "assists", "turnovers", "passes",
            "pass_attempts", "steals", "blocks", "fouls",
        ):
            setattr(dst, attr, getattr(dst, attr) + getattr(src, attr))
        dst.highlight_score = round(dst.highlight_score + src.highlight_score, 2)

        # Merge zone shooting lines
        for zone, line in src.zone_shooting.items():
            if zone in dst.zone_shooting:
                dst.zone_shooting[zone].attempts += line.attempts
                dst.zone_shooting[zone].makes    += line.makes
            else:
                dst.zone_shooting[zone] = line

        dst.first_seen_frame = min(dst.first_seen_frame, src.first_seen_frame)
        # Drop the stale row
        self.game_stats.players.pop(src_id, None)

    def _refresh_teams(self) -> None:
        players = list(self.game_stats.players.values())
        for team in self.game_stats.teams.values():
            team.refresh_from_players(players)

    def _check_lead_change(self) -> None:
        sc = self.score()
        if len(sc) < 2:
            return
        leader = max(sc, key=sc.get)
        if self._last_lead_team is not None and leader != self._last_lead_team:
            for team in self.game_stats.teams.values():
                team.lead_changes += 1
        self._last_lead_team = leader

        # Update largest lead
        scores = list(sc.values())
        if len(scores) >= 2:
            lead = abs(scores[0] - scores[1])
            for team in self.game_stats.teams.values():
                team.largest_lead = max(team.largest_lead, lead)

    def _fire(
        self,
        event_name: str,
        event:      GameEvent,
        frame_id:   int,
        momentum:   Dict,
    ) -> None:
        for cb in self._callbacks.get(event_name, []):
            try:
                cb(event, frame_id, momentum)
            except Exception as exc:
                logger.warning(f"StatsEngine callback error [{event_name}]: {exc}")

    def register_callback(
        self,
        event_name: str,
        callback:   Callable,
    ) -> None:
        """Register a callback for any event type name (string)."""
        self._callbacks.setdefault(event_name, []).append(callback)

    def reset(self) -> None:
        """Full reset — clears all stats for a new game."""
        self.game_stats       = GameStats(game_id=self.game_stats.game_id)
        for i, tid in enumerate(self.team_ids):
            self.game_stats.teams[tid] = TeamStats(
                team_id   = tid,
                team_name = self.team_names.get(tid),
                side      = TeamSide.HOME if i == 0 else TeamSide.AWAY,
            )
        self.momentum_tracker.reset()
        self.highlight_scorer.reset()
        self.shot_chart.reset()
        self._pending_shot  = None
        self._last_lead_team = None
        # Identity-resolution maps (jersey → canonical row, track_id aliases)
        self._jersey_key = {}
        self._alias      = {}
        logger.info("StatsEngine reset.")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums / constants
    "StatCategory",
    "TeamSide",
    "PER_WEIGHTS",
    "HIGHLIGHT_WEIGHTS",
    # Dataclasses
    "ShootingLine",
    "PlayerStats",
    "TeamStats",
    "GameStats",
    # Sub-systems
    "MomentumTracker",
    "HighlightScorer",
    "ShotChartBuilder",
    # Engine
    "StatsEngine",
]