# =============================================================================
# config.py — Purple Box Sports
# System & Pipeline Configuration Settings
# =============================================================================

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).parent.resolve()
MODELS_DIR     = ROOT_DIR / "models"
OUTPUT_DIR     = ROOT_DIR / "output"
HIGHLIGHTS_DIR = OUTPUT_DIR / "highlights"
STATS_DIR      = OUTPUT_DIR / "stats"
LOGS_DIR       = ROOT_DIR / "logs"

PLAYER_MODEL_PATH = str(MODELS_DIR / "player_detect.pt")
BALL_MODEL_PATH   = str(MODELS_DIR / "ball_detect.pt")
RIM_MODEL_PATH    = str(MODELS_DIR / "basketball_rim_best.pt")


def ensure_directories_exist() -> None:
    """Safely creates required system directories if they do not exist."""
    for path in (MODELS_DIR, OUTPUT_DIR, HIGHLIGHTS_DIR, STATS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_default_device() -> str:
    """
    Checks for GPU hardware acceleration (CUDA / MPS).
    Safely falls back to 'cpu' if no GPU or acceleration drivers are detected.
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# -----------------------------------------------------------------------------
# Sub-Configurations
# -----------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    player_model_path: str   = PLAYER_MODEL_PATH
    ball_model_path:   str   = BALL_MODEL_PATH
    rim_model_path:    str   = RIM_MODEL_PATH
    player_conf:       float = 0.45
    # Ball conf is intentionally low: the basketball is a small, fast object
    # whose per-box confidence is naturally low (avg ~0.15-0.25). Detecting at
    # 0.40 threw away ~86% of real balls. We detect at 0.10 to catch nearly
    # every ball, then reject false positives via trajectory gating in
    # BallTracker._select_candidate.
    ball_conf:          float = 0.10
    rim_conf:           float = 0.30
    player_iou:        float = 0.45
    ball_iou:          float = 0.40
    rim_iou:           float = 0.50
    device:            str   = field(default_factory=get_default_device)
    imgsz:             int   = 640
    half:              bool  = False
    player_class_id:   int   = 0
    ball_class_id:     int   = 0
    rim_class_id:      int   = 0


@dataclass
class TrackingConfig:
    tracker_type:      str   = "bytetrack"
    track_high_thresh: float = 0.50
    track_low_thresh:  float = 0.10
    new_track_thresh:  float = 0.60
    track_buffer:      int   = 30
    match_thresh:      float = 0.80


@dataclass
class OCRConfig:
    enabled:             bool  = True
    jersey_top_frac:     float = 0.10
    jersey_bot_frac:     float = 0.55  # Expanded to prevent clipping 2-digit numbers
    ocr_every_n_frames:  int   = 10
    min_confidence:      float = 0.50


@dataclass
class DeadBallConfig:
    enabled:                  bool  = True
    stationary_threshold_px:  float = 6.0
    stationary_frames:        int   = 60
    missing_frames_threshold: int   = 22
    out_of_bounds_margin:     int   = 30
    min_dead_duration_sec:    float = 0.5


@dataclass
class EventConfig:
    rim_proximity_px:     float = 65.0
    shot_arc_min_height:  float = 40.0
    make_basket_y_delta:  float = 15.0


@dataclass
class HighlightConfig:
    pre_buffer_sec:  float = 3.0
    post_buffer_sec: float = 2.0
    output_fps:      float = 30.0


@dataclass
class VideoConfig:
    input_path:   Optional[str] = None
    output_path:  Optional[str] = None
    show_preview: bool          = False
    save_video:   bool          = True


@dataclass
class TeamConfig:
    team_0_name:  str                 = "Home"
    team_1_name:  str                 = "Away"
    team_0_color: Tuple[int, int, int] = (255, 100, 50)
    team_1_color: Tuple[int, int, int] = (50, 100, 255)


# -----------------------------------------------------------------------------
# Main System Configuration
# -----------------------------------------------------------------------------

@dataclass
class AppConfig:
    detection:         DetectionConfig = field(default_factory=DetectionConfig)
    tracking:          TrackingConfig  = field(default_factory=TrackingConfig)
    ocr:               OCRConfig       = field(default_factory=OCRConfig)
    dead_ball:         DeadBallConfig  = field(default_factory=DeadBallConfig)
    events:            EventConfig     = field(default_factory=EventConfig)
    highlights:        HighlightConfig = field(default_factory=HighlightConfig)
    video:             VideoConfig     = field(default_factory=VideoConfig)
    teams:             TeamConfig      = field(default_factory=TeamConfig)
    log_level:         str             = "INFO"
    stats_output_path: str             = str(STATS_DIR / "game_stats.json")
    game_id:           Optional[str]   = None

    def __post_init__(self) -> None:
        ensure_directories_exist()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AppConfig:
        """Deep instantiation helper to prevent raw dict overrides on nested dataclasses."""
        return cls(
            detection=DetectionConfig(**d.get("detection", {})),
            tracking=TrackingConfig(**d.get("tracking", {})),
            ocr=OCRConfig(**d.get("ocr", {})),
            dead_ball=DeadBallConfig(**d.get("dead_ball", {})),
            events=EventConfig(**d.get("events", {})),
            highlights=HighlightConfig(**d.get("highlights", {})),
            video=VideoConfig(**d.get("video", {})),
            teams=TeamConfig(**d.get("teams", {})),
            log_level=d.get("log_level", "INFO"),
            stats_output_path=d.get("stats_output_path", str(STATS_DIR / "game_stats.json")),
            game_id=d.get("game_id", None),
        )


__all__ = [
    "ROOT_DIR", "MODELS_DIR", "OUTPUT_DIR", "HIGHLIGHTS_DIR", "STATS_DIR", "LOGS_DIR",
    "PLAYER_MODEL_PATH", "BALL_MODEL_PATH", "RIM_MODEL_PATH",
    "ensure_directories_exist", "get_default_device",
    "DetectionConfig", "TrackingConfig", "OCRConfig", "DeadBallConfig",
    "EventConfig", "HighlightConfig", "VideoConfig", "TeamConfig", "AppConfig",
]