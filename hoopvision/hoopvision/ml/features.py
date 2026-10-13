"""Fixed-length feature windows around candidate events, for the learned event model.

Each candidate (a rim interaction, a possession change, a whistle) becomes a T x F array
sampled around its frame: ball position relative to the nearest rim, ball speed, the
nearest players' distances and team, and the involved player's pose scores. The same
function builds training windows and inference windows, so they can never drift apart.
"""

from __future__ import annotations

import numpy as np

from ..court import Calibration
from ..pose import PoseTrack, contest_score, shooting_score
from ..types import Event, Track

KINDS = [
    "none", "shot_made", "shot_missed", "free_throw_made", "free_throw_missed",
    "rebound", "assist", "steal", "turnover", "block", "foul",
]
WINDOW = 32  # frames sampled per candidate
SPAN_S = 2.4  # seconds covered (before and after the candidate frame)
FEATURES = 12


def _centre(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def window(
    frame: int,
    track_id: int | None,
    ball: Track,
    players: list[Track],
    identities: dict[int, tuple[str | None, str | None]],
    calib: Calibration,
    fps: float,
    poses: dict[int, PoseTrack] | None = None,
) -> np.ndarray:
    poses = poses or {}
    ball_at = {bf.frame: bf.box for bf in ball.frames}
    team = identities.get(track_id, (None, None))[0] if track_id is not None else None
    half = SPAN_S * fps / 2
    frames = np.linspace(frame - half, frame + half, WINDOW).astype(int)
    out = np.zeros((WINDOW, FEATURES), dtype=np.float32)
    prev = None
    for i, f in enumerate(frames):
        box = ball_at.get(int(f))
        if box is None:
            out[i, 0] = 1.0  # ball not seen
            prev = None
            continue
        bx, by = calib.to_court(*_centre(box), int(f))
        hoop = calib.nearest_hoop((bx, by))
        hx, hy = calib.hoops[hoop]
        out[i, 1] = (bx - hx) / 47.0
        out[i, 2] = (by - hy) / 25.0
        ref = calib.to_reference(*_centre(box), int(f))
        rim = calib.rim_boxes.get(hoop)
        if rim:
            w = max(rim[2] - rim[0], 1.0)
            out[i, 3] = np.clip((ref[0] - (rim[0] + rim[2]) / 2) / (4 * w), -1, 1)
            out[i, 4] = np.clip((ref[1] - (rim[1] + rim[3]) / 2) / (4 * w), -1, 1)
        if prev is not None:
            out[i, 5] = min(np.hypot(bx - prev[0], by - prev[1]) * fps / 40.0, 2.0)
        prev = (bx, by)
        dists = []
        for t in players:
            pb = t.box_at(int(f))
            if pb is None:
                continue
            px, py = calib.foot_point(pb, int(f))
            tt = identities.get(t.track_id, (None, None))[0]
            dists.append((np.hypot(px - bx, py - by), tt == team and team is not None,
                          t.track_id == track_id))
        dists.sort()
        if dists:
            out[i, 6] = min(dists[0][0] / 20.0, 1.0)
            out[i, 7] = float(dists[0][1])
            out[i, 8] = float(dists[0][2])
        if len(dists) > 1:
            out[i, 9] = min(dists[1][0] / 20.0, 1.0)
        pose = poses.get(track_id) if track_id is not None else None
        if pose is not None:
            kp = pose.nearest(int(f), 4)
            out[i, 10] = shooting_score(kp)
            out[i, 11] = contest_score(kp)
    return out


def candidates(events: list[Event]) -> list[Event]:
    """Every rule-detected event except bookkeeping possessions."""
    return [e for e in events if e.kind != "possession"]
