"""Run configuration.

Defaults target a static 4K sideline camera covering the whole court. Detection runs on a
downscaled frame; the ball gets a second pass on native-resolution tiles because at 4K a
basketball is roughly 25 px across and disappears when the frame is scaled to 1280.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    model: str = "yolov8x.pt"
    ball_model: str = "yolov8x.pt"
    # Long edge the frame is resized to for the player pass.
    player_imgsz: int = 1536
    player_conf: float = 0.25
    # The ball pass slices the frame into overlapping tiles at native resolution.
    ball_tile: int = 960
    ball_tile_overlap: int = 160
    ball_conf: float = 0.10
    # Class id the ball model emits for the ball. Stock YOLO uses COCO "sports ball"
    # (32); a fine-tuned basketball model (basketball_rim_best.pt) uses "basketball" (0).
    # Pointing this at a basketball-specific class is the main ball-recall accuracy lever.
    ball_class: int = 32
    # Tiling the frame into native-res crops is what lets a stock model see a tiny ball on
    # a 4K frame, but it runs several inferences per frame. A basketball-trained model
    # detects the ball on the whole (downscaled) frame in ONE pass — far cheaper and the
    # main throughput lever. Set False to skip tiling and do a single full-frame ball pass.
    ball_tiled: bool = True
    ball_imgsz: int = 1280
    # Only look for the ball inside the court polygon, dilated by this many pixels.
    ball_search_margin: int = 120
    device: str = "cpu"
    half: bool = False
    batch: int = 8


@dataclass
class TrackConfig:
    # ByteTrack-style two-stage association.
    high_thresh: float = 0.55
    low_thresh: float = 0.15
    # Maximum IoU distance (1 - IoU) accepted by each association pass.
    match_thresh: float = 0.80
    second_match_thresh: float = 0.50
    max_age: int = 45
    min_hits: int = 3
    # Tracks shorter than this are dropped before identity resolution.
    min_track_len: int = 12


@dataclass
class JerseyConfig:
    # Torso crop as a fraction of the player box (numbers sit on the upper back/chest).
    torso_top: float = 0.18
    torso_bottom: float = 0.55
    torso_inset: float = 0.12
    min_crop_height: int = 48
    upscale_to: int = 128
    ocr_conf: float = 0.40
    # A track needs this many agreeing reads before its number is trusted.
    min_votes: int = 3
    vote_margin: float = 0.55
    sample_every: int = 3


@dataclass
class EventConfig:
    # Possession: a player holds the ball while within this distance (court feet) of it.
    possession_radius_ft: float = 3.5
    possession_min_frames: int = 5
    # A held ball travels with its handler. Anything faster is a pass or a shot in
    # flight, which would otherwise be credited to whichever defender it flew past.
    possession_max_ball_speed_fps: float = 14.0
    # Hysteresis: a defender standing next to the handler is often momentarily the
    # closest player, so the ball only changes hands when someone is clearly closer.
    possession_switch_margin_ft: float = 1.5
    possession_hold_radius_mult: float = 1.6
    # Shot detection around the calibrated rim boxes.
    rim_approach_ft: float = 4.0
    made_descent_frames: int = 4
    three_point_radius_ft: float = 22.15
    corner_three_x_ft: float = 22.0
    # Rebound: first player to gain possession within this window after a miss.
    rebound_window_s: float = 4.0
    assist_window_s: float = 3.0
    clip_pre_s: float = 4.0
    clip_post_s: float = 3.0


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    jersey: JerseyConfig = field(default_factory=JerseyConfig)
    events: EventConfig = field(default_factory=EventConfig)
    # Process every Nth frame. 2 is plenty at 60 fps, 1 at 30 fps.
    frame_stride: int = 1

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls(
            detection=DetectionConfig(**raw.get("detection", {})),
            track=TrackConfig(**raw.get("track", {})),
            jersey=JerseyConfig(**raw.get("jersey", {})),
            events=EventConfig(**raw.get("events", {})),
            frame_stride=raw.get("frame_stride", 1),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
