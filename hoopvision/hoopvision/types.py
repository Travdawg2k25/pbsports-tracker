"""Shared data types and (de)serialisation for the stage artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels


@dataclass
class Detection:
    box: Box
    conf: float
    cls: str  # "player" | "ball"


@dataclass
class TrackFrame:
    frame: int
    box: Box
    conf: float


@dataclass
class Track:
    track_id: int
    cls: str
    frames: list[TrackFrame] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0].frame

    @property
    def end(self) -> int:
        return self.frames[-1].frame

    def box_at(self, frame: int) -> Box | None:
        # Frames are appended in order, so a bisect would work; tracks are short enough
        # that a dict lookup built once is simpler and faster in practice.
        if not hasattr(self, "_index"):
            self._index = {f.frame: f.box for f in self.frames}
        return self._index.get(frame)


@dataclass
class Identity:
    track_id: int
    team: str | None  # "home" | "away" | None
    jersey: str | None
    jersey_confidence: float
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class Event:
    kind: str  # shot_made, shot_missed, rebound, assist, turnover, steal, block, possession
    frame: int
    time_s: float
    player: str | None  # player key: "{team}:{jersey}"
    track_id: int | None = None
    value: int = 0  # points for shots
    confidence: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_encode(obj), indent=2))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _encode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_encode(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def tracks_from_json(raw: list[dict[str, Any]]) -> list[Track]:
    return [
        Track(
            track_id=t["track_id"],
            cls=t["cls"],
            frames=[TrackFrame(f["frame"], tuple(f["box"]), f["conf"]) for f in t["frames"]],
        )
        for t in raw
    ]


def player_key(team: str | None, jersey: str | None) -> str | None:
    if jersey is None:
        return None
    return f"{team or 'unknown'}:{jersey}"
