# app.py — Purple Box Sports | Full Basketball Analytics Pipeline
# ================================================================
# Version: 2.0.0
# Author:  Purple Box Sports Engineering
# ================================================================

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# ── Internal modules ──────────────────────────────────────────────────────────
from pipeline import AnalysisPipeline
from stats_engine import StatsEngine
from dead_ball_detector import DeadBallDetector
from jersey_ocr import JerseyOCR
from tracker import MultiObjectTracker, BallTracker, RimTracker, Visualizer
# REPLACE with this:
from basketball_events import (
    EventEngine,
    EventType,
    GameEvent,
    EventLog,
)

# Alias so the rest of app.py keeps working without renaming every reference
BasketballEventDetector = EventEngine

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("purple_box.log", mode="w"),
    ],
)
logger = logging.getLogger("PurpleBox")


# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG: Dict[str, Any] = {
    # Models
    "player_model":         "yolov8n.pt",
    "ball_model":           "basketball_rim_best.pt",
    "device":               "cpu",

    # Detection
    "conf_threshold":       0.35,
    "iou_threshold":        0.45,
    "player_classes":       [0],          # COCO person class
    "ball_class":           "basketball",
    "rim_class":            "rim",

    # Tracking
    "track_buffer":         30,
    "max_disappeared":      20,
    "min_hits":             3,
    "max_distance":         100,

    # Highlights
    "highlight_margin":     5,            # seconds before/after event
    "min_clip_gap":         3,            # seconds between clips
    "highlight_quality":    "high",       # high | medium | low

    # Output
    "output_dir":           "output",
    "highlights_dir":       "output/highlights",
    "stats_file":           "output/stats.json",
    "events_file":          "output/events.json",
    "summary_file":         "output/summary.json",
    "log_file":             "purple_box.log",

    # Features
    "ocr_enabled":          True,
    "dead_ball_filter":     True,
    "draw_trails":          True,
    "draw_zones":           True,
    "draw_heatmap":         False,

    # Display
    "display_width":        1280,
    "display_height":       720,
    "overlay_alpha":        0.6,

    # Stats
    "team_a_name":          "Team A",
    "team_b_name":          "Team B",
    "period_length":        600,          # seconds (10 min period)
    "num_periods":          4,
}


# ═════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════════════════
class Palette:
    PURPLE      = (180,   0, 180)
    LIGHT_PURPLE= (220, 100, 220)
    WHITE       = (255, 255, 255)
    BLACK       = (  0,   0,   0)
    GOLD        = (  0, 215, 255)
    GREEN       = (  0, 200,   0)
    RED         = (  0,   0, 200)
    BLUE        = (200,   0,   0)
    ORANGE      = (  0, 140, 255)
    GRAY        = (128, 128, 128)
    DARK        = ( 20,   0,  20)
    TRANSPARENT = (  0,   0,   0)

    TEAM_A      = (255,  80,  80)
    TEAM_B      = ( 80,  80, 255)

    EVENT_COLORS = {
        "score":    (  0, 215, 255),
        "rebound":  (  0, 200,   0),
        "steal":    (255, 165,   0),
        "block":    (  0,   0, 200),
        "assist":   (200,   0, 200),
        "turnover": (  0, 100, 200),
        "foul":     (  0,  50, 255),
        "default":  (255, 255, 255),
    }

    @classmethod
    def for_event(cls, event_type: str) -> Tuple[int, int, int]:
        return cls.EVENT_COLORS.get(event_type.lower(), cls.EVENT_COLORS["default"])


# ═════════════════════════════════════════════════════════════════════════════
# FRAME BUFFER  (rolling circular buffer for highlight recording)
# ═════════════════════════════════════════════════════════════════════════════
class FrameBuffer:
    """Thread-safe rolling frame buffer."""

    def __init__(self, max_frames: int):
        self.max_frames = max_frames
        self._buf: deque = deque(maxlen=max_frames)
        self._frame_indices: deque = deque(maxlen=max_frames)

    def push(self, frame: np.ndarray, frame_idx: int) -> None:
        self._buf.append(frame.copy())
        self._frame_indices.append(frame_idx)

    def get_range(self, start_idx: int, end_idx: int) -> List[np.ndarray]:
        """Return frames whose index falls in [start_idx, end_idx]."""
        result = []
        for fi, fr in zip(self._frame_indices, self._buf):
            if start_idx <= fi <= end_idx:
                result.append(fr)
        return result

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def oldest_index(self) -> int:
        return self._frame_indices[0] if self._frame_indices else 0

    @property
    def newest_index(self) -> int:
        return self._frame_indices[-1] if self._frame_indices else 0


# ═════════════════════════════════════════════════════════════════════════════
# HIGHLIGHT RECORDER
# ═════════════════════════════════════════════════════════════════════════════
class HighlightRecorder:
    """
    Records highlight clips around key game events.

    Strategy
    --------
    • Maintains a rolling FrameBuffer of recent annotated frames.
    • When an event is marked, stores the current frame index.
    • On flush(), groups nearby events and writes one MP4 per group.
    """

    def __init__(self, config: Dict[str, Any], fps: float, frame_size: Tuple[int, int]):
        self.config       = config
        self.fps          = max(fps, 1.0)
        self.frame_size   = frame_size   # (width, height)
        self.out_dir      = Path(config["highlights_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.margin_frames = int(config["highlight_margin"] * self.fps)
        self.gap_frames    = int(config["min_clip_gap"]     * self.fps)

        # Buffer holds enough frames for 2× the margin on each side
        # Increased to 60s to give periodic flush enough headroom
        buf_seconds  = config["highlight_margin"] * 2 + 60
        self._buffer = FrameBuffer(max_frames=int(buf_seconds * self.fps))

        # Periodic flush interval (flush every 20s of video to avoid losing frames)
        self._flush_interval_frames = int(20 * self.fps)
        self._frames_since_flush    = 0

        self._event_marks: List[Dict[str, Any]] = []   # {frame_idx, event}
        self._clip_index  = 0
        self._saved_clips : List[str] = []

        # Per-player event marks for player-specific highlights
        self._player_event_marks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        quality_map = {"high": 28, "medium": 20, "low": 14}
        self._crf   = quality_map.get(config.get("highlight_quality", "high"), 28)

        logger.info(
            "HighlightRecorder ready | margin=%ds gap=%ds buffer=%d frames flush_interval=%d frames",
            config["highlight_margin"], config["min_clip_gap"],
            self._buffer.max_frames, self._flush_interval_frames,
        )

    # ── public ───────────────────────────────────────────────────────────────
    def push_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        self._buffer.push(frame, frame_idx)
        self._frames_since_flush += 1

        # Periodic flush: write clips whose events are old enough that
        # we've accumulated the full margin of trailing frames.
        if self._frames_since_flush >= self._flush_interval_frames:
            self._periodic_flush(frame_idx)

    def mark_event(self, frame_idx: int, event: Dict[str, Any]) -> None:
        self._event_marks.append({"frame_idx": frame_idx, "event": event})
        logger.debug("Highlight mark: frame=%d event=%s", frame_idx, event.get("type"))

        # Track per-player for player-specific highlight reels
        player_key = event.get("jersey") or event.get("player") or ""
        if player_key:
            self._player_event_marks[player_key].append(
                {"frame_idx": frame_idx, "event": event}
            )

    def flush(self) -> List[str]:
        """Write all pending highlight clips. Returns list of file paths."""
        if not self._event_marks:
            return []

        groups = self._group_events(self._event_marks)
        new_clips: List[str] = []

        for group in groups:
            clip_path = self._write_clip(group)
            if clip_path:
                new_clips.append(clip_path)
                self._saved_clips.append(clip_path)

        self._event_marks.clear()
        return new_clips

    def flush_player_highlights(self) -> Dict[str, List[str]]:
        """
        Write per-player highlight reels from accumulated player event marks.
        Returns {player_key: [clip_paths]}.
        Called at end of analysis after all periodic flushes.
        Note: since frames may no longer be in buffer, this uses whatever
        frames are available. For best results, periodic flush handles most clips.
        """
        result: Dict[str, List[str]] = {}
        player_dir = self.out_dir / "players"
        player_dir.mkdir(parents=True, exist_ok=True)

        for player_key, marks in self._player_event_marks.items():
            if not marks:
                continue
            groups = self._group_events(marks)
            player_clips: List[str] = []
            for i, group in enumerate(groups):
                clip_path = self._write_player_clip(player_key, i, group, player_dir)
                if clip_path:
                    player_clips.append(clip_path)
            if player_clips:
                result[player_key] = player_clips
                logger.info(
                    "Player '%s' highlights: %d clips", player_key, len(player_clips)
                )

        return result

    @property
    def all_clips(self) -> List[str]:
        return list(self._saved_clips)

    # ── private ──────────────────────────────────────────────────────────────
    def _periodic_flush(self, current_frame_idx: int) -> None:
        """
        Flush event marks that are old enough (event + margin < current frame).
        This ensures clips are written while their frames are still in the buffer.
        """
        self._frames_since_flush = 0

        if not self._event_marks:
            return

        # Split marks into "ready" (old enough) and "pending" (too recent)
        cutoff = current_frame_idx - self.margin_frames
        ready_marks  = [m for m in self._event_marks if m["frame_idx"] <= cutoff]
        pending_marks = [m for m in self._event_marks if m["frame_idx"] > cutoff]

        if not ready_marks:
            return

        groups = self._group_events(ready_marks)
        for group in groups:
            clip_path = self._write_clip(group)
            if clip_path:
                self._saved_clips.append(clip_path)

        # Keep only the pending marks
        self._event_marks = pending_marks

    def _group_events(self, marks: List[Dict]) -> List[List[Dict]]:
        """Merge events that are within gap_frames of each other."""
        sorted_marks = sorted(marks, key=lambda m: m["frame_idx"])
        groups: List[List[Dict]] = []
        current_group: List[Dict] = []

        for mark in sorted_marks:
            if not current_group:
                current_group.append(mark)
            elif mark["frame_idx"] - current_group[-1]["frame_idx"] <= self.gap_frames:
                current_group.append(mark)
            else:
                groups.append(current_group)
                current_group = [mark]

        if current_group:
            groups.append(current_group)

        return groups

    def _write_clip(self, group: List[Dict]) -> Optional[str]:
        """Write a single highlight clip for a group of events."""
        first_frame = group[0]["frame_idx"]
        last_frame  = group[-1]["frame_idx"]

        start_idx = max(0, first_frame - self.margin_frames)
        end_idx   = last_frame + self.margin_frames

        frames = self._buffer.get_range(start_idx, end_idx)
        if not frames:
            logger.warning("No frames found for clip (group size=%d)", len(group))
            return None

        out_path = str(self.out_dir / f"highlight_{self._clip_index:04d}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        w, h     = self.frame_size
        writer   = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.error("Cannot open video writer: %s", out_path)
            return None

        for f in frames:
            resized = cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f
            writer.write(resized)

        # ── Title card ───────────────────────────────────────────────────────
        title_card = self._make_title_card(group, (w, h))
        for _ in range(int(self.fps * 2)):   # 2-second title
            writer.write(title_card)

        writer.release()
        self._clip_index += 1
        logger.info("Highlight saved: %s (%d frames)", out_path, len(frames))
        return out_path

    def _write_player_clip(
        self,
        player_key: str,
        clip_num:   int,
        group:      List[Dict],
        player_dir: Path,
    ) -> Optional[str]:
        """Write a highlight clip for a specific player."""
        first_frame = group[0]["frame_idx"]
        last_frame  = group[-1]["frame_idx"]

        start_idx = max(0, first_frame - self.margin_frames)
        end_idx   = last_frame + self.margin_frames

        frames = self._buffer.get_range(start_idx, end_idx)
        if not frames:
            # Frames already evicted from buffer (expected for earlier events)
            return None

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(player_key))
        out_path  = str(player_dir / f"player_{safe_name}_{clip_num:04d}.mp4")
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        w, h      = self.frame_size
        writer    = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.error("Cannot open player clip writer: %s", out_path)
            return None

        for f in frames:
            resized = cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f
            writer.write(resized)

        # Title card
        title_card = self._make_player_title_card(player_key, group, (w, h))
        for _ in range(int(self.fps * 2)):
            writer.write(title_card)

        writer.release()
        logger.info("Player highlight saved: %s (%d frames)", out_path, len(frames))
        return out_path

    def _make_player_title_card(
        self,
        player_key: str,
        group:      List[Dict],
        size:       Tuple[int, int],
    ) -> np.ndarray:
        """Create a title card for a player-specific highlight."""
        w, h = size
        card = np.zeros((h, w, 3), dtype=np.uint8)
        card[:] = (60, 0, 60)

        cv2.putText(card, "PURPLE BOX SPORTS", (w//2 - 200, h//2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, Palette.GOLD, 3)

        # Player identifier
        player_label = f"Player #{player_key}" if player_key.isdigit() else player_key
        cv2.putText(card, player_label, (w//2 - 120, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, Palette.WHITE, 2)

        # Event summary
        event_types = list({m["event"].get("type", "event") for m in group})
        label = " + ".join(t.upper() for t in event_types[:4])
        cv2.putText(card, label, (w//2 - 150, h//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, Palette.LIGHT_PURPLE, 2)

        return card

    def _make_title_card(self, group: List[Dict], size: Tuple[int, int]) -> np.ndarray:
        """Create a purple title card for the highlight clip."""
        w, h  = size
        card  = np.zeros((h, w, 3), dtype=np.uint8)
        card[:] = (60, 0, 60)   # dark purple background

        # Logo text
        cv2.putText(card, "PURPLE BOX SPORTS", (w//2 - 200, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, Palette.GOLD, 3)

        # Event labels
        event_types = list({m["event"].get("type", "event") for m in group})
        label = " + ".join(t.upper() for t in event_types)
        cv2.putText(card, label, (w//2 - 150, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, Palette.WHITE, 2)

        # Player name if available
        players = list({str(m["event"].get("player", "")) for m in group if m["event"].get("player")})
        if players:
            player_str = ", ".join(players)
            cv2.putText(card, player_str, (w//2 - 100, h//2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, Palette.LIGHT_PURPLE, 2)

        return card


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT VIDEO WRITER
# ═════════════════════════════════════════════════════════════════════════════
class OutputWriter:
    """Writes annotated frames to an MP4 output file."""

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int]):
        self.path       = path
        self.fps        = fps
        self.frame_size = frame_size
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self._ok     = self._writer.isOpened()
        self._count  = 0

        if self._ok:
            logger.info("OutputWriter ready: %s @ %.1f fps %dx%d", path, fps, *frame_size)
        else:
            logger.error("OutputWriter FAILED to open: %s", path)

    def write(self, frame: np.ndarray) -> None:
        if not self._ok:
            return
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)
        self._count += 1

    def release(self) -> None:
        if self._ok:
            self._writer.release()
            logger.info("OutputWriter closed: %d frames written to %s", self._count, self.path)

    @property
    def frames_written(self) -> int:
        return self._count


# ═════════════════════════════════════════════════════════════════════════════
# SCOREBOARD / HUD OVERLAY
# ═════════════════════════════════════════════════════════════════════════════
class ScoreboardOverlay:
    """
    Draws a semi-transparent scoreboard HUD on each frame.

    Sections
    --------
    • Top-left  : Team scores + period + clock
    • Left side : Last 5 events feed
    • Top-right : Frame info + dead-ball indicator
    • Bottom    : Player stat strip (optional)
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, config: Dict[str, Any]):
        self.config      = config
        self.alpha       = config.get("overlay_alpha", 0.6)
        self.team_a_name = config.get("team_a_name", "Team A")
        self.team_b_name = config.get("team_b_name", "Team B")

    def draw(
        self,
        frame:      np.ndarray,
        stats:      Dict[str, Any],
        events:     List[Dict],
        frame_idx:  int,
        fps:        float,
        dead_ball:  bool       = False,
        period:     int        = 1,
        game_clock: float      = 0.0,
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_scoreboard_panel(out, stats, period, game_clock)
        self._draw_event_feed(out, events)
        self._draw_frame_info(out, frame_idx, fps, dead_ball)
        self._draw_stat_strip(out, stats)
        return out

    # ── panels ───────────────────────────────────────────────────────────────
    def _draw_scoreboard_panel(
        self,
        frame:      np.ndarray,
        stats:      Dict[str, Any],
        period:     int,
        game_clock: float,
    ) -> None:
        h, w = frame.shape[:2]
        panel_w, panel_h = 400, 80

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), Palette.DARK, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # Brand name
        cv2.putText(frame, "PURPLE BOX SPORTS", (10, 22),
                    self.FONT_BOLD, 0.55, Palette.PURPLE, 2)

        # Scores
        score_a = stats.get("team_a_score", 0)
        score_b = stats.get("team_b_score", 0)
        score_txt = f"{self.team_a_name}  {score_a}  —  {score_b}  {self.team_b_name}"
        cv2.putText(frame, score_txt, (10, 50),
                    self.FONT_BOLD, 0.65, Palette.GOLD, 2)

        # Period + clock
        mins, secs = divmod(int(game_clock), 60)
        clock_txt  = f"Q{period}  {mins:02d}:{secs:02d}"
        cv2.putText(frame, clock_txt, (10, 72),
                    self.FONT, 0.5, Palette.WHITE, 1)

    def _draw_event_feed(self, frame: np.ndarray, events: List[Dict]) -> None:
        h, w = frame.shape[:2]
        recent = events[-6:] if len(events) >= 6 else events

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 85), (300, 85 + len(recent) * 22 + 10),
                      Palette.DARK, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        y = 102
        for ev in reversed(recent):
            ev_type  = ev.get("type", "event")
            player   = ev.get("player", "")
            jersey   = ev.get("jersey", "")
            color    = Palette.for_event(ev_type)
            label    = f"• {ev_type.upper():<10} {jersey:<4} {player}"
            cv2.putText(frame, label, (8, y), self.FONT, 0.42, color, 1)
            y += 22

    def _draw_frame_info(
        self,
        frame:     np.ndarray,
        frame_idx: int,
        fps:       float,
        dead_ball: bool,
    ) -> None:
        h, w = frame.shape[:2]

        elapsed = frame_idx / max(fps, 1)
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"Video: {mins:02d}:{secs:02d}"
        cv2.putText(frame, time_str, (w - 160, 25),
                    self.FONT, 0.55, Palette.WHITE, 1)

        fps_str = f"Frame: {frame_idx}"
        cv2.putText(frame, fps_str, (w - 160, 48),
                    self.FONT, 0.45, Palette.GRAY, 1)

        if dead_ball:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 170, 58), (w - 5, 82), (0, 0, 80), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "⬛ DEAD BALL", (w - 165, 76),
                        self.FONT, 0.5, Palette.RED, 2)

    def _draw_stat_strip(self, frame: np.ndarray, stats: Dict[str, Any]) -> None:
        h, w = frame.shape[:2]
        strip_h = 30

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - strip_h), (w, h), Palette.DARK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        items = [
            ("REB",  stats.get("total_rebounds",  0)),
            ("AST",  stats.get("total_assists",   0)),
            ("STL",  stats.get("total_steals",    0)),
            ("BLK",  stats.get("total_blocks",    0)),
            ("TO",   stats.get("total_turnovers", 0)),
            ("FGA",  stats.get("field_goal_attempts", 0)),
            ("FGM",  stats.get("field_goals_made",    0)),
        ]

        x = 10
        for label, val in items:
            txt = f"{label}: {val}"
            cv2.putText(frame, txt, (x, h - 10),
                        self.FONT, 0.45, Palette.WHITE, 1)
            x += 120
            if x > w - 120:
                break


# ═════════════════════════════════════════════════════════════════════════════
# HEATMAP GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
class HeatmapGenerator:
    """Accumulates player/ball positions and renders a court heatmap."""

    def __init__(self, frame_size: Tuple[int, int]):
        w, h = frame_size
        self._map   = np.zeros((h, w), dtype=np.float32)
        self._count = 0

    def update(self, positions: List[Tuple[int, int]]) -> None:
        for x, y in positions:
            if 0 <= x < self._map.shape[1] and 0 <= y < self._map.shape[0]:
                self._map[y, x] += 1.0
        self._count += 1

    def render(self, base_frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if self._map.max() == 0:
            return base_frame

        norm   = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        norm   = norm.astype(np.uint8)
        color  = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(base_frame, 1 - alpha, color, alpha, 0)
        return result

    def save(self, path: str) -> None:
        norm  = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        norm  = norm.astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite(path, color)
        logger.info("Heatmap saved: %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# JERSEY OCR MANAGER
# ═════════════════════════════════════════════════════════════════════════════
class JerseyOCRManager:
    """
    Wraps JerseyOCR with caching and confidence filtering.

    Caches the last confirmed jersey number per track ID to avoid
    re-running OCR on every frame (expensive).
    """

    def __init__(self, ocr: JerseyOCR, cache_ttl_frames: int = 9000):
        self._ocr       = ocr
        self._cache_ttl = cache_ttl_frames
        # {track_id: {"number": str, "age": int}}
        self._cache: Dict[int, Dict] = {}

    def process_detections(
        self,
        frame:      np.ndarray,
        detections: List[Dict],
    ) -> None:
        """
        Run OCR on player crops and attach jersey numbers to detections in-place.
        """
        for det in detections:
            if det.get("class") not in ("player", "person"):
                continue

            track_id = det.get("track_id", -1)

            # Check cache
            if track_id in self._cache:
                cached = self._cache[track_id]
                cached["age"] += 1
                if cached["age"] < self._cache_ttl:
                    det["jersey"] = cached["number"]
                    continue
                else:
                    del self._cache[track_id]

            # Crop player region
            x1, y1, x2, y2 = (int(v) for v in det.get("bbox", [0, 0, 0, 0]))
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            number = self._ocr.read(frame, [x1, y1, x2, y2], track_id=track_id)
            if number:
                det["jersey"] = number
                if track_id >= 0:
                    self._cache[track_id] = {"number": number, "age": 0}
                logger.debug("Jersey OCR: track=%d number=%s", track_id, number)

    def clear_stale(self, active_track_ids: List[int]) -> None:
        stale = [tid for tid in self._cache if tid not in active_track_ids]
        for tid in stale:
            del self._cache[tid]


# ═════════════════════════════════════════════════════════════════════════════
# GAME CLOCK MANAGER
# ═════════════════════════════════════════════════════════════════════════════
class GameClockManager:
    """
    Tracks game clock and period based on live-ball time.

    Dead-ball frames are excluded from the game clock.
    """

    def __init__(self, config: Dict[str, Any], fps: float):
        self.fps            = max(fps, 1.0)
        self.period_length  = config.get("period_length", 600)   # seconds
        self.num_periods    = config.get("num_periods", 4)
        self._live_frames   = 0
        self._period        = 1
        self._period_frames = 0

    def tick(self, is_dead: bool) -> None:
        if not is_dead:
            self._live_frames   += 1
            self._period_frames += 1

            if self._period_frames >= self.period_length * self.fps:
                self._period        = min(self._period + 1, self.num_periods)
                self._period_frames = 0
                logger.info("Period %d started", self._period)

    @property
    def period(self) -> int:
        return self._period

    @property
    def game_clock(self) -> float:
        """Seconds elapsed in current period."""
        return self._period_frames / self.fps

    @property
    def total_live_seconds(self) -> float:
        return self._live_frames / self.fps


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITOR
# ═════════════════════════════════════════════════════════════════════════════
class PerformanceMonitor:
    """Tracks processing FPS and reports every N frames."""

    def __init__(self, report_interval: int = 300):
        self._interval   = report_interval
        self._t_start    = time.time()
        self._frame_idx  = 0
        self._section_t  = time.time()

    def tick(self, frame_idx: int, total_frames: int) -> Optional[str]:
        self._frame_idx = frame_idx
        if frame_idx > 0 and frame_idx % self._interval == 0:
            elapsed   = time.time() - self._t_start
            proc_fps  = frame_idx / max(elapsed, 0.001)
            pct       = (frame_idx / total_frames * 100) if total_frames else 0
            remaining = ((total_frames - frame_idx) / proc_fps) if proc_fps > 0 else 0
            msg = (
                f"Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                f"{proc_fps:.1f} fps | ETA {remaining:.0f}s"
            )
            logger.info(msg)
            return msg
        return None

    @property
    def elapsed(self) -> float:
        return time.time() - self._t_start

    @property
    def avg_fps(self) -> float:
        return self._frame_idx / max(self.elapsed, 0.001)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS RESULT
# ═════════════════════════════════════════════════════════════════════════════
class AnalysisResult:
    """Structured container for the final analysis output."""

    def __init__(
        self,
        frames_processed:  int,
        events:            List[Dict],
        stats:             Dict[str, Any],
        highlight_clips:   List[str],
        elapsed_seconds:   float,
        live_seconds:      float,
        dead_ball_seconds: float,
        heatmap_path:      Optional[str],
        output_video:      Optional[str],
        player_highlights: Optional[Dict[str, List[str]]] = None,
    ):
        self.frames_processed  = frames_processed
        self.events            = events
        self.stats             = stats
        self.highlight_clips   = highlight_clips
        self.elapsed_seconds   = elapsed_seconds
        self.live_seconds      = live_seconds
        self.dead_ball_seconds = dead_ball_seconds
        self.heatmap_path      = heatmap_path
        self.output_video      = output_video
        self.player_highlights = player_highlights or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_processed":  self.frames_processed,
            "events":            self.events,
            "stats":             self.stats,
            "highlight_clips":   self.highlight_clips,
            "elapsed_seconds":   round(self.elapsed_seconds,   2),
            "live_seconds":      round(self.live_seconds,       2),
            "dead_ball_seconds": round(self.dead_ball_seconds,  2),
            "heatmap_path":      self.heatmap_path,
            "output_video":      self.output_video,
            "total_events":      len(self.events),
            "total_clips":       len(self.highlight_clips),
            "player_highlights": self.player_highlights,
        }

    def log_summary(self) -> None:
        logger.info("═" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("  Frames processed : %d",   self.frames_processed)
        logger.info("  Live game time   : %.1fs", self.live_seconds)
        logger.info("  Dead ball time   : %.1fs", self.dead_ball_seconds)
        logger.info("  Total events     : %d",    len(self.events))
        logger.info("  Highlight clips  : %d",    len(self.highlight_clips))
        logger.info("  Wall time        : %.1fs", self.elapsed_seconds)
        logger.info("  Avg proc speed   : %.1f fps", self.frames_processed / max(self.elapsed_seconds, 1))
        logger.info("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# PURPLE BOX APP  ← main orchestrator
# ═════════════════════════════════════════════════════════════════════════════
class PurpleBoxApp:
    """
    Top-level orchestrator for the Purple Box Sports analytics pipeline.

    Integrates:
      • AnalysisPipeline    — YOLO detection + multi-object tracking + event detection
      • StatsEngine         — rolling game statistics
      • DeadBallDetector    — identifies and filters dead-ball frames
      • JerseyOCRManager    — reads jersey numbers from player crops
      • HighlightRecorder   — writes event-based highlight clips
      • ScoreboardOverlay   — draws HUD on annotated frames
      • HeatmapGenerator    — builds player/ball position heatmaps
      • GameClockManager    — tracks live game time across periods
      • PerformanceMonitor  — logs processing throughput
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._setup_output_dirs()

        logger.info("╔══════════════════════════════════════╗")
        logger.info("║   Purple Box Sports Analytics v2.0   ║")
        logger.info("╚══════════════════════════════════════╝")

        # ── Core modules ─────────────────────────────────────────────────────
        logger.info("Loading AnalysisPipeline …")
        self.pipeline    = AnalysisPipeline(self.config)

        logger.info("Loading StatsEngine …")
        self.stats       = StatsEngine()

        logger.info("Loading DeadBallDetector …")
        self.dead_ball   = DeadBallDetector()

        logger.info("Loading JerseyOCR …")
        raw_ocr          = JerseyOCR() if self.config["ocr_enabled"] else None
        self.ocr_manager : Optional[JerseyOCRManager] = (
            JerseyOCRManager(raw_ocr) if raw_ocr else None
        )

        # ── Overlay ───────────────────────────────────────────────────────────
        self.overlay     = ScoreboardOverlay(self.config)

        # ── State ─────────────────────────────────────────────────────────────
        self._all_events : List[Dict]  = []
        self._dead_frames: int         = 0
        self._live_frames: int         = 0

        # ── Per-session objects (built once video metadata is known) ──────────
        self._highlight  : Optional[HighlightRecorder] = None
        self._heatmap    : Optional[HeatmapGenerator]  = None
        self._clock      : Optional[GameClockManager]  = None
        self._perf       : Optional[PerformanceMonitor] = None

        logger.info("PurpleBoxApp ready.")

    # ════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════════════
    def analyze(
        self,
        source:       str,
        output_video: bool = True,
        display:      bool = False,
    ) -> AnalysisResult:
        """
        Run complete analysis on a video file.

        Parameters
        ----------
        source       : path to input video
        output_video : write annotated MP4 to output_dir
        display      : show live OpenCV preview window

        Returns
        -------
        AnalysisResult with all stats, events, clips, and timing.
        """
        # ── Open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (fw, fh)

        logger.info(
            "Source: %s | %.1f fps | %d frames | %dx%d",
            source, fps, total, fw, fh,
        )

        # ── Session objects ───────────────────────────────────────────────────
        self._highlight = HighlightRecorder(self.config, fps, frame_size)
        self._heatmap   = HeatmapGenerator(frame_size) if self.config["draw_heatmap"] else None
        self._clock     = GameClockManager(self.config, fps)
        self._perf      = PerformanceMonitor(report_interval=300)

        # ── Output writer ─────────────────────────────────────────────────────
        out_path   = str(Path(self.config["output_dir"]) / "analyzed_output.mp4")
        out_writer : Optional[OutputWriter] = None
        if output_video:
            out_writer = OutputWriter(out_path, fps, frame_size)

        display_size = (self.config["display_width"], self.config["display_height"])
        frame_idx    = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Core pipeline (detection + tracking + events) ─────────────
                timestamp = frame_idx / fps
                result = self.pipeline.process_frame(frame, timestamp=timestamp)

                # ── Dead-ball state comes from pipeline's internal detector ───
                is_dead = result.get("is_dead_ball", False)

                if is_dead:
                    self._dead_frames += 1
                else:
                    self._live_frames += 1

                # ── Game clock ────────────────────────────────────────────────
                self._clock.tick(is_dead)

                annotated  = result.get("annotated_frame", frame)
                detections = result.get("detections", [])
                new_events = result.get("events", [])

                # ── Jersey OCR (every 300 frames to avoid bottleneck) ────────
                if self.ocr_manager and not is_dead and detections and frame_idx % 300 == 0:
                    self.ocr_manager.process_detections(frame, detections)
                    active_ids = [d.get("track_id", -1) for d in detections]
                    self.ocr_manager.clear_stale(active_ids)

                # ── Stats + events ────────────────────────────────────────────
                for ev in new_events:
                    ev["game_clock"]  = self._clock.game_clock
                    ev["period"]      = self._clock.period
                    ev["frame_idx"]   = frame_idx

                    # Enrich with jersey from stats engine if available
                    track_id = ev.get("player")
                    if track_id and isinstance(track_id, int):
                        ps = self.stats.game_stats.players.get(track_id)
                        if ps and ps.jersey_number and not ev.get("jersey"):
                            ev["jersey"] = ps.jersey_number

                    self.stats.record_event(ev)
                    self._all_events.append(ev)
                    # self._highlight.mark_event(frame_idx, ev)  # disabled to save disk
                    logger.info(
                        "EVENT [Q%d %02d:%02d] %s | player=%s jersey=%s",
                        ev["period"],
                        int(ev["game_clock"]) // 60,
                        int(ev["game_clock"]) % 60,
                        ev.get("type", "?").upper(),
                        ev.get("player", ""),
                        ev.get("jersey", ""),
                    )

                # ── Heatmap ───────────────────────────────────────────────────
                if self._heatmap and not is_dead:
                    positions = [
                        self._center(d["bbox"])
                        for d in detections
                        if "bbox" in d
                    ]
                    self._heatmap.update(positions)

                # ── Overlay ───────────────────────────────────────────────────
                annotated = self.overlay.draw(
                    annotated,
                    self.stats.game_stats.to_dict(),
                    self._all_events,
                    frame_idx,
                    fps,
                    dead_ball  = is_dead,
                    period     = self._clock.period,
                    game_clock = self._clock.game_clock,
                )

                # ── Highlight buffer (disabled to save disk) ─────────────────
                # self._highlight.push_frame(annotated, frame_idx)

                # ── Write output ──────────────────────────────────────────────
                if out_writer:
                    out_writer.write(annotated)

                # ── Live display ──────────────────────────────────────────────
                if display:
                    preview = cv2.resize(annotated, display_size)
                    cv2.imshow("Purple Box Sports", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User requested quit.")
                        break
                    elif key == ord("h"):
                        # Manual highlight mark
                        self._highlight.mark_event(
                            frame_idx,
                            {"type": "manual", "player": "", "jersey": ""},
                        )

                # ── Progress ──────────────────────────────────────────────────
                self._perf.tick(frame_idx, total)
                frame_idx += 1

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()

        # ════════════════════════════════════════════════════════════════════
        # POST-PROCESSING
        # ════════════════════════════════════════════════════════════════════

        # ── Flush highlight clips (disabled to save disk) ─────────────────
        clips = []

        # ── Per-player highlight reels (disabled to save disk) ────────────
        player_clips = {}

        # ── Save heatmap ──────────────────────────────────────────────────────
        heatmap_path: Optional[str] = None
        if self._heatmap:
            heatmap_path = str(Path(self.config["output_dir"]) / "heatmap.png")
            self._heatmap.save(heatmap_path)

        # ── Final stats ───────────────────────────────────────────────────────
        stats_summary = self.stats.game_stats.to_dict()

        # ── Save JSON outputs ─────────────────────────────────────────────────
        self._save_json(self.config["stats_file"],  stats_summary)
        self._save_json(self.config["events_file"], self._all_events)

        # ── Per-player stats JSON ─────────────────────────────────────────────
        player_stats_path = str(Path(self.config["output_dir"]) / "player_stats.json")
        player_stats_data = {
            str(pid): pstats.to_dict()
            for pid, pstats in self.stats.game_stats.players.items()
        }
        self._save_json(player_stats_path, player_stats_data)

        # ── Per-player highlights JSON ────────────────────────────────────────
        if player_clips:
            player_highlights_path = str(Path(self.config["output_dir"]) / "player_highlights.json")
            self._save_json(player_highlights_path, player_clips)

        # ── Build result ──────────────────────────────────────────────────────
        # Include all clips: periodic flushes + final flush
        all_highlight_clips = self._highlight.all_clips
        result_obj = AnalysisResult(
            frames_processed  = frame_idx,
            events            = self._all_events,
            stats             = stats_summary,
            highlight_clips   = all_highlight_clips,
            elapsed_seconds   = self._perf.elapsed,
            live_seconds      = self._clock.total_live_seconds,
            dead_ball_seconds = self._dead_frames / fps,
            heatmap_path      = heatmap_path,
            output_video      = out_path if output_video else None,
            player_highlights = player_clips,
        )

        self._save_json(self.config["summary_file"], result_obj.to_dict())
        result_obj.log_summary()

        return result_obj

    def cleanup(self) -> None:
        """Release all module resources."""
        try:
            self.pipeline.cleanup()
        except Exception as e:
            logger.warning("Pipeline cleanup error: %s", e)
        cv2.destroyAllWindows()
        logger.info("PurpleBoxApp cleaned up.")

    # ════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════════
    def _setup_output_dirs(self) -> None:
        for key in ("output_dir", "highlights_dir"):
            Path(self.config[key]).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _center(bbox: List[float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @staticmethod
    def _save_json(path: str, data: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Saved: %s", path)
        except Exception as e:
            logger.error("Failed to save %s: %s", path, e)


# ═════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="purple_box",
        description=(
            "Purple Box Sports — Automated Basketball Analytics\n"
            "Records games, detects events, generates highlights & stats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py game.mp4
  python app.py game.mp4 --display
  python app.py game.mp4 --ball-model basketball_rim_best.pt --device cuda
  python app.py game.mp4 --no-ocr --no-dead-ball-filter --output-dir results/
  python app.py game.mp4 --config my_config.json --display
        """,
    )

    # ── Positional ────────────────────────────────────────────────────────────
    p.add_argument(
        "video",
        help="Path to input video file (mp4, avi, mov …)",
    )

    # ── Models ────────────────────────────────────────────────────────────────
    model_grp = p.add_argument_group("Model Options")
    model_grp.add_argument(
        "--player-model",
        default=None,
        dest="player_model",
        metavar="PATH",
        help="YOLO weights for player detection (default: yolov8n.pt)",
    )
    model_grp.add_argument(
        "--ball-model",
        default=None,
        dest="ball_model",
        metavar="PATH",
        help="YOLO weights for ball/rim detection (default: basketball_rim_best.pt)",
    )
    model_grp.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Inference device",
    )
    model_grp.add_argument(
        "--conf",
        default=None,
        type=float,
        metavar="FLOAT",
        help="Detection confidence threshold 0-1 (default: 0.35)",
    )
    model_grp.add_argument(
        "--iou",
        default=None,
        type=float,
        metavar="FLOAT",
        help="NMS IoU threshold 0-1 (default: 0.45)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_grp = p.add_argument_group("Output Options")
    out_grp.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="Directory for all output files (default: output/)",
    )
    out_grp.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated output video",
    )
    out_grp.add_argument(
        "--display",
        action="store_true",
        help="Show live preview window (press Q to quit, H to mark highlight)",
    )
    out_grp.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate court heatmap from player positions",
    )

    # ── Features ──────────────────────────────────────────────────────────────
    feat_grp = p.add_argument_group("Feature Flags")
    feat_grp.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable jersey number OCR",
    )
    feat_grp.add_argument(
        "--no-dead-ball-filter",
        action="store_true",
        help="Disable dead-ball frame filtering",
    )
    feat_grp.add_argument(
        "--highlight-margin",
        default=None,
        type=float,
        dest="highlight_margin",
        metavar="SECS",
        help="Seconds of footage before/after event in highlight clip (default: 5)",
    )
    feat_grp.add_argument(
        "--team-a",
        default=None,
        dest="team_a_name",
        metavar="NAME",
        help="Team A name for scoreboard",
    )
    feat_grp.add_argument(
        "--team-b",
        default=None,
        dest="team_b_name",
        metavar="NAME",
        help="Team B name for scoreboard",
    )

    # ── Config ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to JSON config file (overrides defaults, CLI overrides config)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    return p


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Verbose logging ───────────────────────────────────────────────────────
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled.")

    # ── Load JSON config (if provided) ────────────────────────────────────────
    config: Dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            with open(cfg_path) as f:
                config = json.load(f)
            logger.info("Loaded config from: %s", cfg_path)
        else:
            logger.warning("Config file not found: %s — using defaults", cfg_path)

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    cli_map = {
        "player_model":    args.player_model,
        "ball_model":      args.ball_model,
        "device":          args.device,
        "conf_threshold":  args.conf,
        "iou_threshold":   args.iou,
        "output_dir":      args.output_dir,
        "team_a_name":     args.team_a_name,
        "team_b_name":     args.team_b_name,
        "highlight_margin":args.highlight_margin,
    }
    for key, val in cli_map.items():
        if val is not None:
            config[key] = val

    if args.no_ocr:
        config["ocr_enabled"]     = False
    if args.no_dead_ball_filter:
        config["dead_ball_filter"]= False
    if args.heatmap:
        config["draw_heatmap"]    = True

    # ── Derive highlights_dir from output_dir if changed ─────────────────────
    if "output_dir" in config and "highlights_dir" not in config:
        config["highlights_dir"] = str(Path(config["output_dir"]) / "highlights")

    # ── Validate input video ──────────────────────────────────────────────────
    video_path = args.video
    if not Path(video_path).exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    # ── Build app and run ─────────────────────────────────────────────────────
    app = PurpleBoxApp(config)

    try:
        result = app.analyze(
            source       = video_path,
            output_video = not args.no_video,
            display      = args.display,
        )

        # ── Print final summary to console ────────────────────────────────────
        print("\n" + "═" * 55)
        print("  PURPLE BOX SPORTS — ANALYSIS SUMMARY")
        print("═" * 55)
        print(f"  Frames processed  : {result.frames_processed:,}")
        print(f"  Live game time    : {result.live_seconds:.1f}s")
        print(f"  Dead ball time    : {result.dead_ball_seconds:.1f}s")
        print(f"  Total events      : {len(result.events)}")
        print(f"  Highlight clips   : {len(result.highlight_clips)}")
        if result.output_video:
            print(f"  Output video      : {result.output_video}")
        if result.heatmap_path:
            print(f"  Heatmap           : {result.heatmap_path}")
        print(f"  Wall time         : {result.elapsed_seconds:.1f}s")
        print("═" * 55)

        # ── Event breakdown ───────────────────────────────────────────────────
        event_counts: Dict[str, int] = defaultdict(int)
        for ev in result.events:
            event_counts[ev.get("type", "unknown")] += 1

        if event_counts:
            print("\n  EVENT BREAKDOWN:")
            for etype, count in sorted(event_counts.items()):
                print(f"    {etype:<15} : {count}")

        # ── Highlight clip list ───────────────────────────────────────────────
        if result.highlight_clips:
            print(f"\n  HIGHLIGHT CLIPS ({len(result.highlight_clips)}):")
            for clip in result.highlight_clips:
                print(f"    {clip}")

        # ── Per-player highlights ─────────────────────────────────────────────
        if result.player_highlights:
            print(f"\n  PER-PLAYER HIGHLIGHTS:")
            for pkey, pclips in result.player_highlights.items():
                print(f"    Player #{pkey}: {len(pclips)} clips")
                for pc in pclips:
                    print(f"      {pc}")

        # ── Box Score ─────────────────────────────────────────────────────────
        if result.stats.get("players"):
            print("\n" + "=" * 80)
            print("  BOX SCORE")
            print("=" * 80)
            print(f"  {'ID':<5} {'Jersey':<8} {'PTS':<5} {'FGM':<5} {'FGA':<5} {'FG%':<7} {'3PM':<5} {'3PA':<5} {'REB':<5} {'AST':<5} {'STL':<5} {'BLK':<5} {'TO':<5}")
            print("  " + "-" * 76)
            for pid, pstats in result.stats.get("players", {}).items():
                pts = pstats.get("points", 0)
                fgm = pstats.get("fg_makes", 0)
                fga = pstats.get("fg_attempts", 0)
                fg_pct = f"{100*fgm/fga:.0f}%" if fga > 0 else "-"
                three_m = pstats.get("fg3_makes", 0)
                three_a = pstats.get("fg3_attempts", 0)
                reb = pstats.get("total_rebounds", 0)
                ast = pstats.get("assists", 0)
                stl = pstats.get("steals", 0)
                blk = pstats.get("blocks", 0)
                to = pstats.get("turnovers", 0)
                jersey = pstats.get("jersey_number", "-")
                print(f"  {pid:<5} {jersey:<8} {pts:<5} {fgm:<5} {fga:<5} {fg_pct:<7} {three_m:<5} {three_a:<5} {reb:<5} {ast:<5} {stl:<5} {blk:<5} {to:<5}")
            print("=" * 80)

        print()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error during analysis: %s", e)
        sys.exit(1)
    finally:
        app.cleanup()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
