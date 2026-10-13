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

from .audio import Whistle
from .config import EventConfig
from .court import LANE_WIDTH_FT, Calibration, shot_value
from .pose import PoseTrack, contest_score, shooting_score
from .review import ensure_ids
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
        # A shot approaches the rim from above and falls just below it. The trigger
        # region is generous above the rim (the incoming arc) but tight to the sides
        # and only just below it — otherwise a ball dribbled in the backcourt drifts
        # into an oversized neighbourhood and every pass is mistaken for a shot.
        # One rim-width either side still catches a shot that clanks off the front or
        # back iron, which a half-width margin missed entirely (no misses were counted).
        near_x1, near_x2 = rx1 - 1.0 * width, rx2 + 1.0 * width
        near_y1, near_y2 = ry1 - 3 * height, ry2 + 1.5 * height
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
    whistles: list[Whistle] | None = None,
    poses: dict[int, PoseTrack] | None = None,
) -> list[Event]:
    events = _rule_events(player_tracks, ball, calib, identities, fps, cfg)
    if poses:
        _apply_pose(events, poses, cfg)
    if whistles:
        events.extend(
            _fouls(whistles, events, player_tracks, ball, calib, identities, fps, cfg)
        )
    return ensure_ids(sorted(events, key=lambda e: e.frame))


def _rule_events(
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
        is_ft = shooter is not None and _is_free_throw(
            shooter, rim.frame, poss, rims, calib, cfg, fps
        )
        if is_ft:
            # A free throw is one point and tracked separately from field goals.
            shot = Event(
                kind="free_throw_made" if rim.made else "free_throw_missed",
                frame=rim.frame,
                time_s=rim.frame / fps,
                player=shooter.player if shooter else None,
                track_id=shooter.track_id if shooter else None,
                value=1 if rim.made else 0,
                confidence=rim.confidence * 0.9,
                detail={
                    "hoop": rim.hoop,
                    "free_throw": True,
                    "shot_court_xy": list(shooter.court_xy) if shooter else None,
                },
            )
            events.append(shot)
            continue

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

        if not rim.made:
            block = _detect_block(
                shooter, rim, player_tracks, ball, calib, identities, cfg, fps
            )
            if block is not None:
                events.append(block)

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


def _is_free_throw(
    shooter: Possession,
    frame: int,
    poss: list[Possession],
    rims: list[RimEvent],
    calib: Calibration,
    cfg: EventConfig,
    fps: float,
) -> bool:
    """A shot is a free throw when it is taken from the FT line during dead-ball play.

    Signals (all cheap, from data already computed):
      * the shooter stands near the free-throw line (its distance from the baseline is
        known: ``ft_line_ft``), and
      * play was stopped beforehand — no rim interaction in the seconds prior, which is a
        proxy for the whistle/dead ball that precedes a free throw.

    Distance geometry is only trustworthy with a homography; without one we cannot tell a
    free throw from a close jump shot, so we do not guess (returns False on the rim-only
    path, and the shot counts as a normal field goal).
    """
    if not calib.distance_reliable:
        return False
    # Court x of the FT line for the shooter's end.
    hoop = calib.nearest_hoop(shooter.court_xy)
    baseline_x = 0.0 if hoop == "left" else calib.court_length_ft
    inward = 1.0 if hoop == "left" else -1.0
    ft_line_x = baseline_x + inward * calib.ft_line_ft
    near_ft_line = abs(shooter.court_xy[0] - ft_line_x) <= cfg.free_throw_line_tol_ft
    # The shooter must also be inside the FT circle (lane width), not out on the elbow
    # extended, where a jump shot is the same distance from the baseline.
    in_circle = abs(shooter.court_xy[1] - calib.court_width_ft / 2) <= LANE_WIDTH_FT / 2 + 1.0
    if not (near_ft_line and in_circle):
        return False
    # Dead ball before: no rim interaction in the window preceding the shot.
    dead_window = cfg.free_throw_dead_s * fps
    recent_rim = any(0 < frame - r.frame <= dead_window for r in rims)
    return not recent_rim


def _detect_block(
    shooter: Possession | None,
    rim: RimEvent,
    player_tracks: list[Track],
    ball: Track,
    calib: Calibration,
    identities: dict[int, tuple[str | None, str | None]],
    cfg: EventConfig,
    fps: float,
) -> Event | None:
    """A missed shot where a defender sat between shooter and rim is a likely block.

    This is a deliberately low-confidence, single-camera heuristic: it flags a *candidate*
    block for a human to confirm rather than asserting one. It fires only on missed shots
    (a made basket was not blocked) when an opponent's feet fall between the shooter and
    the rim around the shot frame.
    """
    if shooter is None or shooter.team is None:
        return None
    hoop_xy = calib.hoops.get(rim.hoop)
    if hoop_xy is None:
        return None
    sx, sy = shooter.court_xy
    hx, hy = hoop_xy
    # Look for an opposing player whose court position lies between shooter and rim near
    # the shot frame.
    window = cfg.block_window_s * fps
    seg_len = float(np.hypot(hx - sx, hy - sy)) or 1.0
    best: tuple[float, int] | None = None
    for t in player_tracks:
        box = t.box_at(rim.frame)
        if box is None:
            # allow a nearby frame within the block window
            near = [tf for tf in t.frames if abs(tf.frame - rim.frame) <= window]
            if not near:
                continue
            box = min(near, key=lambda tf: abs(tf.frame - rim.frame)).box
        px, py = calib.foot_point(box, rim.frame)
        # Perpendicular distance of the player from the shooter->rim line.
        # Project (p - s) onto (h - s); the residual is the gap.
        t_proj = ((px - sx) * (hx - sx) + (py - sy) * (hy - sy)) / (seg_len**2)
        if not (0.15 <= t_proj <= 0.95):  # between shooter and rim, not on top of either
            continue
        projx, projy = sx + t_proj * (hx - sx), sy + t_proj * (hy - sy)
        gap = float(np.hypot(px - projx, py - projy))
        if gap <= cfg.block_defender_gap_ft and (best is None or gap < best[0]):
            best = (gap, t.track_id)
    if best is None:
        return None
    blocker_id = best[1]
    b_team, b_jersey = identities.get(blocker_id, (None, None))
    if b_jersey:
        blocker_key = f"{b_team or 'unknown'}:{b_jersey}"
    else:
        blocker_key = f"{b_team or 'unknown'}:t{blocker_id}"
    return Event(
        kind="block",
        frame=rim.frame,
        time_s=rim.frame / fps,
        player=blocker_key,
        track_id=blocker_id,
        confidence=0.3,
        detail={"shooter": shooter.player, "gap_ft": round(best[0], 1), "candidate": True},
    )


def _apply_pose(events: list[Event], poses: dict[int, PoseTrack], cfg: EventConfig) -> None:
    """Raise or lower shot and block confidence by what the players' bodies were doing.

    A shooter with hands up near release confirms the shot; a "shot" from a player whose
    hands never rose is more likely a tip or a bad ball track. Blocks need a raised arm.
    """
    for i, ev in enumerate(events):
        pose = poses.get(ev.track_id) if ev.track_id is not None else None
        if pose is None:
            continue
        if ev.kind in ("shot_made", "shot_missed", "free_throw_made", "free_throw_missed"):
            # Release happens shortly before the rim interaction.
            release = ev.frame - int(cfg.pose_max_gap * 3)
            best = max(
                (shooting_score(pose.nearest(f, cfg.pose_max_gap))
                 for f in range(release - cfg.pose_max_gap * 3, ev.frame + 1, 2)),
                default=0.0,
            )
            factor = 0.75 + 0.4 * best
            events[i] = _with(ev, min(1.0, ev.confidence * factor), shooting_pose=best)
        elif ev.kind == "block":
            arm = contest_score(pose.nearest(ev.frame, cfg.pose_max_gap * 2))
            events[i] = _with(ev, min(1.0, ev.confidence * (0.5 + 1.2 * arm)), contest_pose=arm)


def _with(ev: Event, confidence: float, **detail) -> Event:
    from dataclasses import replace

    return replace(ev, confidence=round(confidence, 3), detail={**ev.detail, **detail})


def _fouls(
    whistles: list[Whistle],
    events: list[Event],
    player_tracks: list[Track],
    ball: Track,
    calib: Calibration,
    identities: dict[int, tuple[str | None, str | None]],
    fps: float,
    cfg: EventConfig,
) -> list[Event]:
    """One candidate foul per whistle blown during live play.

    The fouled player is whoever held (or last held) the ball just before the whistle.
    The fouler is ranked among opponents by distance to that player at the whistle; the
    top few go in ``detail.candidates`` so a reviewer picks with one click. A whistle
    straight after a made basket or with no one on the ball (timeouts, subs, violations)
    is not a foul.
    """
    poss = [e for e in events if e.kind == "possession"]
    makes = [e.frame for e in events if e.kind in ("shot_made", "free_throw_made")]
    fts = [e.frame for e in events if e.kind.startswith("free_throw")]
    out: list[Event] = []
    for w in whistles:
        frame = int(w.start_s * fps)
        if any(0 <= frame - m <= 1.5 * fps for m in makes):
            continue
        lookback = cfg.foul_lookback_s * fps
        held = [
            p for p in poss
            if p.frame <= frame and frame - p.detail.get("end_frame", p.frame) <= lookback
        ]
        if not held:
            continue
        fouled = max(held, key=lambda p: p.detail.get("end_frame", p.frame))
        fouled_team = identities.get(fouled.track_id, (None, None))[0]
        anchor = _court_at(fouled.track_id, player_tracks, calib, frame, fps)
        if anchor is None:
            continue
        ranked: list[tuple[float, Track]] = []
        for t in player_tracks:
            team = identities.get(t.track_id, (None, None))[0]
            if t.track_id == fouled.track_id or (fouled_team and team == fouled_team):
                continue
            pos = _court_at(t.track_id, player_tracks, calib, frame, fps, track=t)
            if pos is None:
                continue
            ranked.append((float(np.hypot(pos[0] - anchor[0], pos[1] - anchor[1])), t))
        ranked.sort(key=lambda r: r[0])
        cands = [
            {"player": _key(t.track_id, identities), "track_id": t.track_id,
             "distance_ft": round(d, 1)}
            for d, t in ranked[: cfg.foul_candidates]
        ]
        shooting = any(0 < f - frame <= cfg.foul_ft_window_s * fps for f in fts)
        top = cands[0] if cands else None
        margin = (ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else 3.0
        conf = w.confidence * (0.35 + 0.1 * min(margin, 3.0)) * (1.15 if shooting else 1.0)
        # No opponent within reach of the ball handler: off-ball foul or a whistle for
        # something else. Keep the nearest as candidates, but at low confidence.
        if not ranked or ranked[0][0] > cfg.foul_max_gap_ft:
            conf *= 0.4
        out.append(
            Event(
                kind="foul",
                frame=frame,
                time_s=frame / fps,
                player=top["player"] if top else None,
                track_id=top["track_id"] if top else None,
                confidence=round(min(conf, 0.8), 3),
                detail={
                    "fouled": fouled.player,
                    "candidates": cands,
                    "shooting_foul": shooting,
                    "whistle": {"start_s": w.start_s, "end_s": w.end_s},
                },
            )
        )
    return out


def _court_at(tid, player_tracks, calib, frame, fps, track=None):
    t = track or next((x for x in player_tracks if x.track_id == tid), None)
    if t is None or not t.frames:
        return None
    near = min(t.frames, key=lambda tf: abs(tf.frame - frame))
    if abs(near.frame - frame) > fps:
        return None
    return calib.foot_point(near.box, near.frame)


def _key(tid: int, identities) -> str:
    team, jersey = identities.get(tid, (None, None))
    return f"{team or 'unknown'}:{jersey}" if jersey else f"{team or 'unknown'}:t{tid}"
