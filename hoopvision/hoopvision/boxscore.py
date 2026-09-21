"""Box score aggregation and per-player stat lookup.

Every counting stat keeps the list of events that produced it, so picking a player in the
UI gives both the number and the clip behind each one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import EventConfig
from .types import Event, load_json


@dataclass
class PlayerLine:
    player: str
    team: str | None
    jersey: str | None
    name: str | None = None
    points: int = 0
    fgm: int = 0
    fga: int = 0
    tpm: int = 0
    tpa: int = 0
    ftm: int = 0
    fta: int = 0
    rebounds: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    possessions: int = 0
    seconds_with_ball: float = 0.0
    event_indices: list[int] = field(default_factory=list)

    @property
    def fg_pct(self) -> float | None:
        return round(self.fgm / self.fga, 3) if self.fga else None

    @property
    def tp_pct(self) -> float | None:
        return round(self.tpm / self.tpa, 3) if self.tpa else None

    @property
    def ft_pct(self) -> float | None:
        return round(self.ftm / self.fta, 3) if self.fta else None


def aggregate(events: list[Event], fps: float, roster: dict[str, str] | None = None) -> dict:
    lines: dict[str, PlayerLine] = {}
    roster = roster or {}

    def line(key: str) -> PlayerLine:
        if key not in lines:
            team, _, jersey = key.partition(":")
            lines[key] = PlayerLine(
                player=key,
                team=None if team == "unknown" else team,
                jersey=jersey or None,
                name=roster.get(key),
            )
        return lines[key]

    for idx, ev in enumerate(events):
        if ev.player is None:
            continue
        pl = line(ev.player)
        pl.event_indices.append(idx)
        if ev.kind == "shot_made":
            pl.fga += 1
            pl.fgm += 1
            pl.points += ev.value
            if ev.value == 3:
                pl.tpa += 1
                pl.tpm += 1
        elif ev.kind == "shot_missed":
            pl.fga += 1
            if ev.detail.get("attempt_value") == 3:
                pl.tpa += 1
        elif ev.kind == "rebound":
            pl.rebounds += 1
            if ev.detail.get("type") == "offensive":
                pl.offensive_rebounds += 1
            else:
                pl.defensive_rebounds += 1
        elif ev.kind == "free_throw_made":
            pl.ftm += 1
            pl.fta += 1
            pl.points += 1
        elif ev.kind == "free_throw_missed":
            pl.fta += 1
        elif ev.kind == "block":
            pl.blocks += 1
        elif ev.kind == "assist":
            pl.assists += 1
        elif ev.kind == "steal":
            pl.steals += 1
        elif ev.kind == "turnover":
            pl.turnovers += 1
        elif ev.kind == "possession":
            pl.possessions += 1
            end = ev.detail.get("end_frame", ev.frame)
            pl.seconds_with_ball += max(0, end - ev.frame) / fps

    players = []
    for pl in sorted(lines.values(), key=lambda p: (-p.points, p.player)):
        d = asdict(pl)
        d["fg_pct"] = pl.fg_pct
        d["tp_pct"] = pl.tp_pct
        d["ft_pct"] = pl.ft_pct
        d["seconds_with_ball"] = round(pl.seconds_with_ball, 1)
        players.append(d)

    teams: dict[str, dict[str, int]] = {}
    for pl in lines.values():
        t = pl.team or "unknown"
        agg = teams.setdefault(t, {"points": 0, "fgm": 0, "fga": 0, "rebounds": 0, "assists": 0})
        agg["points"] += pl.points
        agg["fgm"] += pl.fgm
        agg["fga"] += pl.fga
        agg["rebounds"] += pl.rebounds
        agg["assists"] += pl.assists

    return {"players": players, "teams": teams}


def player_clips(
    events: list[Event], player: str, fps: float, cfg: EventConfig | None = None
) -> list[dict[str, Any]]:
    """Clip windows for every counted event of one player."""
    cfg = cfg or EventConfig()
    out = []
    for ev in events:
        if ev.player != player or ev.kind == "possession":
            continue
        out.append(
            {
                "kind": ev.kind,
                "time_s": round(ev.time_s, 2),
                "start_s": round(max(0.0, ev.time_s - cfg.clip_pre_s), 2),
                "end_s": round(ev.time_s + cfg.clip_post_s, 2),
                "value": ev.value,
                "confidence": round(ev.confidence, 2),
                "detail": ev.detail,
            }
        )
    return out


def load_roster(path: str | Path | None) -> dict[str, str]:
    """Roster file maps ``{"home:23": "A. Smith"}``."""
    if path is None:
        return {}
    return {str(k): str(v) for k, v in load_json(path).items()}
