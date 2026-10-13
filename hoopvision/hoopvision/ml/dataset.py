"""Training data from reviewed runs.

A run directory that has been through review holds the AI ``events.json`` and the
reviewer's ``edits.json``. Every AI candidate becomes one example: its label is what the
reviewer decided (kept, re-labelled, or deleted -> ``none``). Human-added events are
examples the rules missed, which is exactly what the model most needs to learn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..court import Calibration
from ..pose import load_poses
from ..review import apply_edits, ensure_ids, load_edits
from ..stabilize import Motion
from ..types import Event, load_json, tracks_from_json
from .features import KINDS, window

log = logging.getLogger("hoopvision.ml")


def load_run(run: Path):
    run = Path(run)
    meta = load_json(run / "video.json")
    calib = Calibration.load(run / "calibration.json")
    if (run / "motion.json").exists():
        calib.motion = Motion.load(run / "motion.json")
    players = tracks_from_json(load_json(run / "tracks.json"))
    ball = tracks_from_json([load_json(run / "ball.json")])[0]
    ids = {i["track_id"]: (i["team"], i["jersey"]) for i in load_json(run / "identities.json")}
    events = ensure_ids([Event(**e) for e in load_json(run / "events.json")])
    return meta, calib, players, ball, ids, events, load_poses(run / "pose.json")


def examples_from_run(run: Path) -> tuple[np.ndarray, np.ndarray]:
    meta, calib, players, ball, ids, events, poses = load_run(run)
    edits = load_edits(run)
    if not edits:
        return np.zeros((0, 1, 1), np.float32), np.zeros(0, np.int64)
    fps = meta["fps"]
    final = {e.event_id: e for e in apply_edits(events, edits, fps)}
    xs, ys = [], []
    for ev in events:
        if ev.kind == "possession":
            continue
        kept = final.get(ev.event_id)
        label = kept.kind if kept is not None else "none"
        xs.append(window(ev.frame, ev.track_id, ball, players, ids, calib, fps, poses))
        ys.append(KINDS.index(label) if label in KINDS else 0)
    for ev in final.values():
        if ev.event_id.startswith("manual-") and ev.kind in KINDS:
            xs.append(window(ev.frame, ev.track_id, ball, players, ids, calib, fps, poses))
            ys.append(KINDS.index(ev.kind))
    return np.stack(xs) if xs else np.zeros((0, 1, 1), np.float32), np.array(ys, np.int64)


def build_dataset(runs: list[Path], out: Path) -> dict:
    xs, ys = [], []
    for run in runs:
        x, y = examples_from_run(run)
        if len(y):
            xs.append(x)
            ys.append(y)
            log.info("%s: %d examples", run, len(y))
    if not xs:
        raise ValueError("no reviewed runs (edits.json) found; review a game first")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(out, X=X, y=Y, kinds=np.array(KINDS))
    return {"examples": int(len(Y)), "per_kind": {KINDS[k]: int((Y == k).sum()) for k in set(Y)}}
