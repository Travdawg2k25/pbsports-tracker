"""Event detection: possession, shots, rebounds, assists, turnovers.

Everything here is derived geometry, not a learned model. The chain is:

1. **possession** — per frame, the player closest to the ball in court space (within a
   radius) holds it; runs of frames become possession segments;
2. **rim interactions** — the ball entering the calibrated rim box from above and leaving
   below, descending throughout, is a make; approaching the rim without that is a miss;
3. **attribution** — a rim interaction belongs to the last player who held the ball
   before it, the shot's value comes from where that player was standing;
4. **derived events** — rebounds are the next possession after a miss, assists are a
   teammate's pass shortly before a make, turnovers are possession changing hands
   between teams with no shot in between.

Each event carries a ``confidence``. Shots are the most reliable, assists and turnovers
the least; the UI shows the confidence so a human can spot-check the shaky ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EventConfig
from .court import Calibration, shot_value
from .types import Event, Track


@dataclass
class Possession:
    player: str | None
    track_id: int
    team: str | None
    start: int
    end: int
    court_xy: tuple[float, float]


def _centre(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def ball_speeds(ball: Track, calib: Calibration, fps: float, smooth: int = 2) -> dict[int, float]:
    """Ball speed in court feet per second, per frame of the ball track."""
    court = [(bf.frame, calib.to_court(*_centre(bf.box), bf.frame)) for bf in ball.frames]
    out: dict[int, float] = {}
    for i, (frame, _pos) in enumerate(court):
        j = max(0, i - smooth)
        k = min(len(court) - 1, i + smooth)
        dt = (court[k][0] - court[j][0]) / fps if k != j else 0.0
        if dt <= 0:
            out[frame] = 0.0
            continue
        dist = float(np.hypot(court[k][1][0] - court[j][1][0], court[k][1][1] - court[j][1][1]))
        out[frame] = dist / dt
    return out


def possessions(
    player_tracks: list[Track],
    ball: Track,
    calib: Calibration,
    identities: dict[int, tuple[str | None, str | None]],
    cfg: EventConfig,
    fps: float = 30.0,
) -> list[Possession]:
    """Segment the ball track into runs of frames held by one player."""
    by_frame: dict[int, list[Track]] = {}
    for t in player_tracks:
        for tf in t.frames:
            by_frame.setdefault(tf.frame, []).append(t)

    speeds = ball_speeds(ball, calib, fps)
    holder: int | None = None
    holder_at: dict[int, int] = {}
    pos_at: dict[int, tuple[float, float]] = {}
    for bf in ball.frames:
        if speeds.get(bf.frame, 0.0) > cfg.possession_max_ball_speed_fps:
            continue
        bx, by = _centre(bf.box)
        ball_ct = calib.to_court(bx, by, bf.frame)
        distances: dict[int, tuple[float, Track]] = {}
        for t in by_frame.get(bf.frame, []):
            box = t.box_at(bf.frame)
            if box is None:
                continue
            ct = calib.foot_point(box, bf.frame)
            distances[t.track_id] = (
                float(np.hypot(ct[0] - ball_ct[0], ct[1] - ball_ct[1])),
                t,
            )
        if not distances:
            continue
        tid, (dist, track) = min(distances.items(), key=lambda kv: kv[1][0])

        held = distances.get(holder)
        if holder is not None and held is not None:
            keep = held[0] <= cfg.possession_radius_ft * cfg.possession_hold_radius_mult
            takeover = dist <= held[0] - cfg.possession_switch_margin_ft
            if keep and not takeover:
                tid, (dist, track) = holder, held

        limit = cfg.possession_radius_ft * (
            cfg.possession_hold_radius_mult if tid == holder else 1.0
        )
        if dist > limit:
            holder = None
            continue
        holder = tid
        holder_at[bf.frame] = tid
        box = track.box_at(bf.frame)
        pos_at[bf.frame] = calib.foot_point(box, bf.frame) if box else ball_ct

    out: list[Possession] = []
    current: Possession | None = None
    for frame in sorted(holder_at):
        tid = holder_at[frame]
        if current and current.track_id == tid and frame - current.end <= cfg.possession_min_frames:
            current.end = frame
            continue
        if current and current.end - current.start + 1 >= cfg.possession_min_frames:
            out.append(current)
        team, jersey = identities.get(tid, (None, None))
        # A player key always exists so their events survive aggregation. Jersey OCR only
        # resolves a fraction of tracks on real footage, but the product identifies the
        # selected player by appearance + the parent-typed number, not by OCR — so an
        # un-OCR'd track keys by its track id ("home:t5") instead of being dropped.
        if jersey:
            player_key = f"{team or 'unknown'}:{jersey}"
        else:
            player_key = f"{team or 'unknown'}:t{tid}"
        current = Possession(
            player=player_key,
            track_id=tid,
            team=team,
            start=frame,
            end=frame,
            court_xy=pos_at[frame],
        )
    if current and current.end - current.start + 1 >= cfg.possession_min_frames:
        out.append(current)
    return out


@dataclass
class RimEvent:
    frame: int
    hoop: str
    made: bool
    confidence: float


def rim_events(ball: Track, calib: Calibration, cfg: EventConfig) -> list[RimEvent]:
    """Find rim interactions in the ball track using the calibrated rim boxes."""
    out: list[RimEvent] = []
    if not calib.rim_boxes:
        return out
    # Rim boxes are marked on the reference frame, so the ball is warped onto it too.
    frames = [
        _RefFrame(tf.frame, calib.to_reference(*_centre(tf.box), tf.frame))
        for tf in ball.frames
    ]
    for hoop, rim in calib.rim_boxes.items():
        rx1, ry1, rx2, ry2 = rim
        width, height = rx2 - rx1, ry2 - ry1
        near_x1, near_x2 = rx1 - width, rx2 + width
        near_y1, near_y2 = ry1 - 4 * height, ry2 + 4 * height
        i = 0
        while i < len(frames):
            cx, cy = frames[i].point
            if not (near_x1 <= cx <= near_x2 and near_y1 <= cy <= near_y2):
                i += 1
                continue
            window = [f for f in frames[i : i + 30]]
            made, conf = _through_rim(window, rim, cfg)
            out.append(RimEvent(frame=frames[i].frame, hoop=hoop, made=made, confidence=conf))
            # Skip past this interaction so one shot is not counted twice.
            i += max(1, len(window))
    return sorted(out, key=lambda e: e.frame)


@dataclass
class _RefFrame:
    frame: int
    point: tuple[float, float]


def _through_rim(window: list[_RefFrame], rim, cfg: EventConfig) -> tuple[bool, float]:
    """A make: the ball is above the rim, then below it, inside the cylinder, descending."""
    rx1, ry1, rx2, ry2 = rim
    above = below = False
    descent = 0
    prev_y = None
    inside_when_crossing = False
    for tf in window:
        cx, cy = tf.point
        inside_x = rx1 <= cx <= rx2
        if prev_y is not None and cy > prev_y:
            descent += 1
        if cy < ry1 and inside_x:
            above = True
        if above and ry1 <= cy <= ry2 and inside_x:
            inside_when_crossing = True
        if above and cy > ry2 and inside_x:
            below = True
            break
        prev_y = cy
    made = above and below and inside_when_crossing and descent >= cfg.made_descent_frames
    if made:
        return True, 0.8
    # The ball reached the rim area but did not pass cleanly through it.
    return False, 0.6 if above else 0.4


def build_events(
    player_tracks: list[Track],
    ball: Track,
    calib: Calibration,
    identities: dict[int, tuple[str | None, str | None]],
    fps: float,
    cfg: EventConfig,
) -> list[Event]:
    poss = possessions(player_tracks, ball, calib, identities, cfg, fps)
    rims = rim_events(ball, calib, cfg)
    events: list[Event] = []

    for p in poss:
        events.append(
            Event(
                kind="possession",
                frame=p.start,
                time_s=p.start / fps,
                player=p.player,
                track_id=p.track_id,
                confidence=0.7,
                detail={"end_frame": p.end, "court_xy": list(p.court_xy)},
            )
        )

    def last_possession_before(frame: int, window_s: float = 6.0) -> Possession | None:
        window = window_s * fps
        cands = [p for p in poss if p.end <= frame and frame - p.end <= window]
        return max(cands, key=lambda p: p.end) if cands else None

    def first_possession_after(frame: int, window_s: float) -> Possession | None:
        window = window_s * fps
        cands = [p for p in poss if p.start >= frame and p.start - frame <= window]
        return min(cands, key=lambda p: p.start) if cands else None

    for rim in rims:
        shooter = last_possession_before(rim.frame)
        value = (
            shot_value(calib, shooter.court_xy, cfg.three_point_radius_ft) if shooter else 2
        )
        shot = Event(
            kind="shot_made" if rim.made else "shot_missed",
            frame=rim.frame,
            time_s=rim.frame / fps,
            player=shooter.player if shooter else None,
            track_id=shooter.track_id if shooter else None,
            value=value if rim.made else 0,
            confidence=rim.confidence * (1.0 if shooter else 0.5),
            detail={
                "hoop": rim.hoop,
                "attempt_value": value,
                "shot_court_xy": list(shooter.court_xy) if shooter else None,
                "shot_distance_ft": (
                    round(calib.hoop_distance_ft(shooter.court_xy, rim.hoop), 1)
                    if shooter
                    else None
                ),
            },
        )
        events.append(shot)

        if rim.made and shooter is not None:
            passer = _passer_before(poss, shooter, fps, cfg, rims)
            if passer is not None:
                events.append(
                    Event(
                        kind="assist",
                        frame=passer.end,
                        time_s=passer.end / fps,
                        player=passer.player,
                        track_id=passer.track_id,
                        confidence=0.45,
                        detail={"shot_frame": rim.frame, "shooter": shooter.player},
                    )
                )
        if not rim.made:
            board = first_possession_after(rim.frame, cfg.rebound_window_s)
            if board is not None:
                offensive = shooter is not None and board.team == shooter.team
                events.append(
                    Event(
                        kind="rebound",
                        frame=board.start,
                        time_s=board.start / fps,
                        player=board.player,
                        track_id=board.track_id,
                        confidence=0.55,
                        detail={
                            "type": "offensive" if offensive else "defensive",
                            "shot_frame": rim.frame,
                        },
                    )
                )

    events.extend(_turnovers(poss, rims, fps, cfg))
    return sorted(events, key=lambda e: e.frame)


def _passer_before(
    poss: list[Possession],
    shooter: Possession,
    fps: float,
    cfg: EventConfig,
    rims: list[RimEvent],
) -> Possession | None:
    window = cfg.assist_window_s * fps
    prior = [
        p
        for p in poss
        if p.end <= shooter.start
        and shooter.start - p.end <= window
        and p.track_id != shooter.track_id
    ]
    if not prior:
        return None
    passer = max(prior, key=lambda p: p.end)
    # A rim interaction in between means the shooter got the ball off the glass, not off
    # a pass: a putback is not an assisted basket.
    if any(passer.end <= r.frame <= shooter.start for r in rims):
        return None
    # Only a teammate's pass can be an assist; an opponent losing the ball is not.
    return passer if passer.team == shooter.team and passer.player else None


def _turnovers(
    poss: list[Possession], rims: list[RimEvent], fps: float, cfg: EventConfig
) -> list[Event]:
    out: list[Event] = []
    rim_frames = [r.frame for r in rims]
    for prev, nxt in zip(poss, poss[1:], strict=False):
        if not prev.team or not nxt.team or prev.team == nxt.team:
            continue
        if any(prev.end <= f <= nxt.start for f in rim_frames):
            continue  # possession changed because of a shot, not a turnover
        out.append(
            Event(
                kind="turnover",
                frame=prev.end,
                time_s=prev.end / fps,
                player=prev.player,
                track_id=prev.track_id,
                confidence=0.35,
                detail={"stolen_by": nxt.player},
            )
        )
        out.append(
            Event(
                kind="steal",
                frame=nxt.start,
                time_s=nxt.start / fps,
                player=nxt.player,
                track_id=nxt.track_id,
                confidence=0.35,
                detail={"from": prev.player},
            )
        )
    return out
