"""Pose signals, whistle fouls, review edits, the learned event model and evaluation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hoopvision import synthetic
from hoopvision.audio import Whistle, detect_whistles
from hoopvision.boxscore import aggregate
from hoopvision.config import AudioConfig, Config
from hoopvision.events import build_events
from hoopvision.ml.evaluate import evaluate
from hoopvision.ml.features import FEATURES, WINDOW, window
from hoopvision.pipeline import stage_boxscore, stage_events
from hoopvision.pose import PoseTrack, contest_score, shooting_score
from hoopvision.review import apply_edits, record_edit, review_queue
from hoopvision.types import Event, VideoMeta, load_json, tracks_from_json


def _kp(wrist_y: float, elbow_y: float) -> np.ndarray:
    kp = np.zeros((17, 3))
    kp[:, 2] = 1.0
    kp[0] = (100, 100, 1)  # nose
    kp[5], kp[6] = (90, 130, 1), (110, 130, 1)  # shoulders
    kp[11], kp[12] = (92, 200, 1), (108, 200, 1)  # hips
    kp[7], kp[8] = (88, elbow_y, 1), (112, elbow_y, 1)
    kp[9], kp[10] = (88, wrist_y, 1), (112, wrist_y, 1)
    return kp


def test_shooting_pose_beats_dribbling_pose():
    shooting = shooting_score(_kp(wrist_y=55, elbow_y=105))
    dribbling = shooting_score(_kp(wrist_y=210, elbow_y=170))
    assert shooting > 0.8 > 0.1 > dribbling
    assert contest_score(_kp(55, 105)) > 0.5 > contest_score(_kp(210, 170))
    assert shooting_score(None) == 0.0


def test_whistle_detected_but_not_crowd_noise():
    sr, cfg = 16000, AudioConfig()
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * 3.0)) / sr
    audio = 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    blow = (t > 1.0) & (t < 1.6)
    audio[blow] += 0.5 * np.sin(2 * np.pi * 3150 * t[blow]).astype(np.float32)
    found = detect_whistles(audio, sr, cfg)
    assert len(found) == 1
    assert 0.9 <= found[0].start_s <= 1.1 and 1.5 <= found[0].end_s <= 1.7
    assert 3000 <= found[0].peak_hz <= 3300
    noise_only = 0.3 * rng.standard_normal(len(t)).astype(np.float32)
    assert detect_whistles(noise_only, sr, cfg) == []


def _synthetic(tmp_path):
    synthetic.generate(tmp_path)
    tracks = tracks_from_json(load_json(tmp_path / "tracks.json"))
    ball = tracks_from_json([load_json(tmp_path / "ball.json")])[0]
    ids = {i["track_id"]: (i["team"], i["jersey"]) for i in load_json(tmp_path / "identities.json")}
    return tracks, ball, ids, synthetic.calibration()


def test_whistle_during_play_becomes_a_ranked_foul(tmp_path):
    tracks, ball, ids, calib = _synthetic(tmp_path)
    base = build_events(tracks, ball, calib, ids, synthetic.FPS, Config().events)
    poss = next(e for e in base if e.kind == "possession" and e.frame > 100)
    when = (poss.frame + 10) / synthetic.FPS
    events = build_events(
        tracks, ball, calib, ids, synthetic.FPS, Config().events,
        whistles=[Whistle(when, when + 0.4, 0.9, 3100)],
    )
    fouls = [e for e in events if e.kind == "foul"]
    assert len(fouls) == 1
    foul = fouls[0]
    assert foul.detail["fouled"] == poss.player
    fouled_team = poss.player.split(":")[0]
    assert foul.player and not foul.player.startswith(fouled_team)
    dists = [c["distance_ft"] for c in foul.detail["candidates"]]
    assert dists == sorted(dists) and 0 < foul.confidence <= 0.8
    assert aggregate(events, synthetic.FPS)["teams"][foul.player.split(":")[0]]["fouls"] == 1
    assert len({e.event_id for e in events}) == len(events)


def test_pose_lowers_confidence_of_a_shot_without_shooting_form(tmp_path):
    tracks, ball, ids, calib = _synthetic(tmp_path)
    cfg = Config().events
    plain = build_events(tracks, ball, calib, ids, synthetic.FPS, cfg)
    shot = next(e for e in plain if e.kind == "shot_made")
    flat = {f: _kp(210, 170).tolist() for f in range(shot.frame - 60, shot.frame + 1)}
    up = {f: _kp(55, 105).tolist() for f in range(shot.frame - 60, shot.frame + 1)}
    low = build_events(tracks, ball, calib, ids, synthetic.FPS, cfg,
                       poses={shot.track_id: PoseTrack(shot.track_id, flat)})
    high = build_events(tracks, ball, calib, ids, synthetic.FPS, cfg,
                        poses={shot.track_id: PoseTrack(shot.track_id, up)})
    conf = {name: next(e for e in evs if e.event_id == shot.event_id).confidence
            for name, evs in (("low", low), ("high", high))}
    assert conf["low"] < shot.confidence < conf["high"]


def test_review_edits_are_layered_and_recomputed(tmp_path):
    tracks, ball, ids, calib = _synthetic(tmp_path)
    meta = VideoMeta(**json.loads((tmp_path / "video.json").read_text()))
    stage_events(tmp_path, Config(), calib, meta)
    before = stage_boxscore(tmp_path, meta, roster=None)
    events = [Event(**e) for e in load_json(tmp_path / "events.json")]
    made = next(e for e in events if e.kind == "shot_made")
    assist = next(e for e in events if e.kind == "assist")

    record_edit(tmp_path, {"op": "modify", "event_id": made.event_id,
                           "changes": {"player": "home:11"}})
    record_edit(tmp_path, {"op": "delete", "event_id": assist.event_id})
    record_edit(tmp_path, {"op": "add", "event": {"kind": "foul", "frame": 200,
                                                   "player": "away:4"}})
    with pytest.raises(ValueError):
        record_edit(tmp_path, {"op": "modify", "event_id": made.event_id,
                               "changes": {"confidence": 1}})
    after = stage_boxscore(tmp_path, meta, roster=None)

    pts = lambda box, k: next((p["points"] for p in box["players"] if p["player"] == k), 0)  # noqa: E731
    assert pts(after, made.player) == pts(before, made.player) - made.value
    assert pts(after, "home:11") == pts(before, "home:11") + made.value
    assert after["teams"]["away"]["fouls"] == 1
    total_ast = lambda box: sum(t["assists"] for t in box["teams"].values())  # noqa: E731
    assert total_ast(after) == total_ast(before) - 1
    assert after["review"]["edits"] == 3
    # The AI events file is untouched; the edit log is what changed.
    assert [Event(**e) for e in load_json(tmp_path / "events.json")] == events
    final = apply_edits(events, load_json(tmp_path / "edits.json"), meta.fps)
    moved = next(e for e in final if e.event_id == made.event_id)
    assert moved.detail["ai_original"]["player"] == made.player
    assert all(q["event_id"] != made.event_id for q in review_queue(final, 0.6))


def test_feature_window_shape(tmp_path):
    tracks, ball, ids, calib = _synthetic(tmp_path)
    x = window(76, 1, ball, tracks, ids, calib, synthetic.FPS)
    assert x.shape == (WINDOW, FEATURES) and np.isfinite(x).all()


def test_event_model_trains_from_reviewed_run_and_rescores(tmp_path):
    torch = pytest.importorskip("torch")
    from hoopvision.ml.dataset import build_dataset
    from hoopvision.ml.event_model import EventRescorer, train

    tracks, ball, ids, calib = _synthetic(tmp_path)
    calib.save(tmp_path / "calibration.json")
    meta = VideoMeta(**json.loads((tmp_path / "video.json").read_text()))
    stage_events(tmp_path, Config(), calib, meta)
    events = [Event(**e) for e in load_json(tmp_path / "events.json")]
    for e in events:
        if e.kind != "possession":
            record_edit(tmp_path, {"op": "confirm", "event_id": e.event_id})
    info = build_dataset([tmp_path], tmp_path / "ds.npz")
    assert info["examples"] == len([e for e in events if e.kind != "possession"])
    torch.manual_seed(0)
    metrics = train(tmp_path / "ds.npz", tmp_path / "events.pt", epochs=60, val_frac=0.0)
    assert metrics["train_examples"] == info["examples"]
    out = EventRescorer(tmp_path / "events.pt").rescore(
        events, ball, tracks, ids, calib, synthetic.FPS
    )
    rescored = [e for e in out if e.detail.get("model")]
    assert rescored and all(0.0 <= e.confidence <= 1.0 for e in rescored)
    assert sum(e.kind == "shot_made" for e in out) == 3


def test_evaluate_against_hand_kept_box_score(tmp_path):
    box = {"players": [
        {"player": "home:23", "points": 5, "rebounds": 1, "fouls": 0},
        {"player": "away:4", "points": 2, "rebounds": 0, "fouls": 2},
    ]}
    truth = tmp_path / "truth.csv"
    truth.write_text("team,jersey,pts,reb,pf\nhome,23,5,2,0\naway,4,2,0,1\naway,9,3,1,0\n")
    from hoopvision.ml.evaluate import load_truth

    rep = evaluate(box, load_truth(truth))
    assert rep["players_matched"] == 2
    assert rep["stats"]["pts"]["team_error"] == {"home": 0, "away": -3}
    assert rep["stats"]["reb"]["exact_pct"] == round(1 / 3, 3)
    assert rep["stats"]["pf"]["mae_per_player"] == round(1 / 3, 2)


def test_player_clips_follow_review_edits(tmp_path):
    from fastapi.testclient import TestClient

    from hoopvision.api import create_app

    tracks, ball, ids, calib = _synthetic(tmp_path)
    meta = VideoMeta(**json.loads((tmp_path / "video.json").read_text()))
    stage_events(tmp_path, Config(), calib, meta)
    stage_boxscore(tmp_path, meta, roster=None)
    block = next(Event(**e) for e in load_json(tmp_path / "events.json") if e["kind"] == "block")
    client = TestClient(create_app(tmp_path))
    client.post(f"/api/events/{block.event_id}/edit",
                json={"op": "modify", "changes": {"player": "home:11"}})
    kinds = lambda p: [c["kind"] for c in client.get(f"/api/players/{p}").json()["clips"]]  # noqa: E731
    assert kinds(block.player).count("block") == 0
    assert "block" in kinds("home:11")
