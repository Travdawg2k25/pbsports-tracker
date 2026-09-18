"""Camera motion compensation, checked against a synthetically panned clip."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from hoopvision import synthetic
from hoopvision.boxscore import aggregate
from hoopvision.config import Config
from hoopvision.events import build_events
from hoopvision.pipeline import stage_motion
from hoopvision.stabilize import Motion, displacement, estimate_motion, rescale
from hoopvision.types import VideoMeta, load_json, tracks_from_json

PAN_PX = 40.0


def _load(run_dir):
    tracks = tracks_from_json(load_json(run_dir / "tracks.json"))
    ball = tracks_from_json([load_json(run_dir / "ball.json")])[0]
    identities = {
        i["track_id"]: (i["team"], i["jersey"])
        for i in load_json(run_dir / "identities.json")
    }
    return tracks, ball, identities


def test_static_clip_has_no_motion(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    motion = estimate_motion(tmp_path / "synthetic.mp4", stride=10)
    assert displacement(motion, synthetic.W, synthetic.H) < 2.0


def test_pan_is_recovered(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False, pan_px=PAN_PX)
    motion = estimate_motion(tmp_path / "synthetic.mp4", stride=5)
    assert displacement(motion, synthetic.W, synthetic.H) > PAN_PX / 2

    # Warping a frame back onto the reference should undo the pan it was rendered with.
    for frame in (60, 120, 180):
        dx = synthetic._pan_offset(frame, PAN_PX)
        x, y = motion.warp(640 + dx, 360, frame)
        assert x == pytest.approx(640, abs=4.0)
        assert y == pytest.approx(360, abs=4.0)


def test_matching_at_reduced_width_gives_full_resolution_pixels():
    """A 20 px shift seen at half size is a 40 px shift on the full frame."""
    half = np.array([[1, 0, -20], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    full = rescale(half, 0.5)
    pt = np.array([[[640.0, 360.0]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(pt, full)[0][0]
    assert (x, y) == pytest.approx((600.0, 360.0))


def test_motion_stage_skips_a_static_camera(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    cfg = Config()
    cfg.frame_stride = 10
    meta = VideoMeta(
        path=str(tmp_path / "synthetic.mp4"),
        fps=synthetic.FPS,
        width=synthetic.W,
        height=synthetic.H,
        frame_count=100,
    )
    stage_motion(tmp_path / "synthetic.mp4", tmp_path, cfg, meta, max_frames=None)
    assert not (tmp_path / "motion.json").exists()


def test_boxscore_survives_a_panning_camera(tmp_path):
    synthetic.generate(tmp_path, pan_px=PAN_PX)
    meta = VideoMeta(**json.loads((tmp_path / "video.json").read_text()))
    cfg = Config()
    calib = synthetic.calibration()
    tracks, ball, identities = _load(tmp_path)

    uncorrected = aggregate(
        build_events(tracks, ball, calib, identities, meta.fps, cfg.events), meta.fps
    )
    calib.motion = estimate_motion(tmp_path / "synthetic.mp4")
    corrected = aggregate(
        build_events(tracks, ball, calib, identities, meta.fps, cfg.events), meta.fps
    )

    expected = synthetic.expected_boxscore()
    lines = {p["player"]: p for p in corrected["players"]}
    for player, want in expected.items():
        for stat, value in want.items():
            assert lines[player][stat] == value, (player, stat, lines[player])

    # The pan is what the correction is for: without it the same clip scores differently.
    assert uncorrected["players"] != corrected["players"]


def test_motion_roundtrips_through_json(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False, pan_px=PAN_PX)
    motion = estimate_motion(tmp_path / "synthetic.mp4", stride=20)
    motion.save(tmp_path / "motion.json")
    back = Motion.load(tmp_path / "motion.json")
    assert back.reference_frame == motion.reference_frame
    assert back.warp(600, 300, 100) == pytest.approx(motion.warp(600, 300, 100))


def test_missing_frames_fall_back_to_the_nearest_estimate():
    motion = Motion(reference_frame=0, homographies={})
    assert motion.warp(10, 20, 5) == (10.0, 20.0)
