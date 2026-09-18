from __future__ import annotations

import numpy as np
import pytest

from hoopvision import synthetic
from hoopvision.boxscore import aggregate, player_clips
from hoopvision.config import Config
from hoopvision.court import HOOPS_FT, Calibration
from hoopvision.events import ball_speeds, build_events, possessions
from hoopvision.jersey import merge_identities, resolve_identity
from hoopvision.pipeline import stage_boxscore
from hoopvision.teams import assign_teams
from hoopvision.track import ByteTracker, iou_matrix
from hoopvision.types import (
    Detection,
    Event,
    Identity,
    Track,
    TrackFrame,
    VideoMeta,
    dump_json,
    tracks_from_json,
)


def test_iou_matrix_matches_hand_computation():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10], [0, 0, 10, 10]], dtype=float)
    out = iou_matrix(a, b)
    assert out[0, 0] == pytest.approx(50 / 150)
    assert out[0, 1] == pytest.approx(1.0)


def test_tracker_keeps_one_id_through_a_gap():
    cfg = Config().track
    cfg.min_track_len = 3
    tracker = ByteTracker(cfg)
    for f in range(20):
        # A single target drifting right, invisible for three frames in the middle.
        if 8 <= f < 11:
            tracker.update([], f)
            continue
        x = 100 + 6 * f
        tracker.update([Detection((x, 50, x + 40, 150), 0.9, "player")], f)
    tracks = tracker.tracks()
    assert len(tracks) == 1
    assert len(tracks[0].frames) == 17


def test_tracker_separates_two_targets():
    tracker = ByteTracker(Config().track)
    for f in range(30):
        tracker.update(
            [
                Detection((100 + 4 * f, 50, 140 + 4 * f, 150), 0.9, "player"),
                Detection((600 - 4 * f, 50, 640 - 4 * f, 150), 0.9, "player"),
            ],
            f,
        )
    assert len(tracker.tracks()) == 2


def test_jersey_vote_needs_a_clear_winner():
    cfg = Config().jersey
    clear = resolve_identity(1, {"23": 4.0, "3": 0.5}, "home", cfg)
    assert clear.jersey == "23"

    split = resolve_identity(2, {"23": 2.0, "25": 2.0}, "home", cfg)
    assert split.jersey is None

    thin = resolve_identity(3, {"8": 0.5}, "home", cfg)
    assert thin.jersey is None


def test_merge_identities_groups_tracks_of_one_player():
    ids = [
        Identity(1, "home", "23", 0.9),
        Identity(2, "home", "23", 0.8),
        Identity(3, "away", "23", 0.8),
        Identity(4, "home", None, 0.0),
    ]
    groups = merge_identities(ids)
    assert groups["home:23"] == [1, 2]
    assert groups["away:23"] == [3]
    assert len(groups) == 2


def test_assign_teams_splits_two_kits():
    colours = {
        1: np.array([230.0, 128.0, 128.0]),
        2: np.array([225.0, 130.0, 126.0]),
        3: np.array([40.0, 120.0, 140.0]),
        4: np.array([45.0, 122.0, 138.0]),
    }
    teams = assign_teams(colours)
    assert teams[1] == teams[2]
    assert teams[3] == teams[4]
    assert teams[1] != teams[3]


def test_court_homography_roundtrip():
    calib = synthetic.calibration()
    for x, y in [(0.0, 0.0), (47.0, 25.0), (94.0, 50.0), (75.0, 10.0)]:
        px = synthetic.court_to_px(x, y)
        back = calib.to_court(*px)
        assert back[0] == pytest.approx(x, abs=0.2)
        assert back[1] == pytest.approx(y, abs=0.2)


def test_shot_value_from_distance():
    calib = synthetic.calibration()
    hx, hy = HOOPS_FT["right"]
    assert calib.hoop_distance_ft((hx - 5, hy)) == pytest.approx(5.0)
    from hoopvision.court import shot_value

    assert shot_value(calib, (hx - 5, hy), 22.15) == 2
    assert shot_value(calib, (hx - 26, hy), 22.15) == 3


def _synthetic_run(tmp_path):
    synthetic.generate(tmp_path)
    tracks = tracks_from_json(__import__("json").loads((tmp_path / "tracks.json").read_text()))
    ball = tracks_from_json([__import__("json").loads((tmp_path / "ball.json").read_text())])[0]
    identities = {
        i + 1: (synthetic.PLAYERS[i][1], synthetic.PLAYERS[i][0])
        for i in range(len(synthetic.PLAYERS))
    }
    return tracks, ball, identities, synthetic.calibration()


def test_possession_assigns_the_nearest_player(tmp_path):
    tracks, ball, identities, calib = _synthetic_run(tmp_path)
    cfg = Config().events
    poss = possessions(tracks, ball, calib, identities, cfg)
    assert poss, "expected possessions from the synthetic clip"
    assert any(p.player == "home:23" for p in poss)


def test_events_reproduce_the_scripted_box_score(tmp_path):
    tracks, ball, identities, calib = _synthetic_run(tmp_path)
    events = build_events(tracks, ball, calib, identities, synthetic.FPS, Config().events)
    box = aggregate(events, synthetic.FPS)
    got = {p["player"]: p for p in box["players"]}
    expected = synthetic.expected_boxscore()

    assert set(expected) <= set(got), f"missing players: {set(expected) - set(got)}"
    for key, want in expected.items():
        for stat, value in want.items():
            assert got[key][stat] == value, f"{key} {stat}: got {got[key][stat]}, want {value}"


def test_no_phantom_turnovers_from_passes(tmp_path):
    # Every scripted possession change happens on a shot, so nothing should be logged as
    # a turnover; a pass flying past a defender used to create one.
    tracks, ball, identities, calib = _synthetic_run(tmp_path)
    events = build_events(tracks, ball, calib, identities, synthetic.FPS, Config().events)
    assert [e for e in events if e.kind == "turnover"] == []


def test_possession_ignores_the_ball_in_flight(tmp_path):
    tracks, ball, identities, calib = _synthetic_run(tmp_path)
    cfg = Config().events
    poss = possessions(tracks, ball, calib, identities, cfg, synthetic.FPS)
    # A shot in flight travels much faster than a handler can run, so no possession may
    # start while the ball is moving at shot speed.
    speeds = ball_speeds(ball, calib, synthetic.FPS)
    for p in poss:
        assert speeds[p.start] <= cfg.possession_max_ball_speed_fps


def test_shots_are_classified_made_and_missed(tmp_path):
    tracks, ball, identities, calib = _synthetic_run(tmp_path)
    events = build_events(tracks, ball, calib, identities, synthetic.FPS, Config().events)
    made = [e for e in events if e.kind == "shot_made"]
    missed = [e for e in events if e.kind == "shot_missed"]
    assert len(made) == 3
    assert len(missed) == 2


def test_aggregate_counts_each_stat_once():
    events = [
        Event("shot_made", 10, 0.3, "home:23", 1, value=3, confidence=0.8),
        Event("shot_missed", 40, 1.3, "home:23", 1, detail={"attempt_value": 2}),
        Event("rebound", 50, 1.6, "home:7", 2, detail={"type": "offensive"}),
        Event("assist", 8, 0.26, "home:7", 2),
        Event("turnover", 90, 3.0, "home:23", 1),
        Event("steal", 92, 3.06, "away:4", 3),
        Event("possession", 5, 0.16, "home:23", 1, detail={"end_frame": 35}),
    ]
    box = aggregate(events, 30.0, roster={"home:23": "A. Smith"})
    lines = {p["player"]: p for p in box["players"]}
    assert lines["home:23"]["name"] == "A. Smith"
    assert lines["home:23"]["points"] == 3
    assert lines["home:23"]["fga"] == 2
    assert lines["home:23"]["tpa"] == 1
    assert lines["home:23"]["turnovers"] == 1
    assert lines["home:23"]["seconds_with_ball"] == pytest.approx(1.0)
    assert lines["home:7"]["offensive_rebounds"] == 1
    assert lines["away:4"]["steals"] == 1
    assert box["teams"]["home"]["points"] == 3


def test_boxscore_records_the_clip_duration(tmp_path):
    # The review header reads video.duration_s, which is a property and so absent from
    # a plain dataclass dump.
    dump_json([], tmp_path / "events.json")
    box = stage_boxscore(tmp_path, VideoMeta("clip.mp4", 30.0, 1920, 1080, 900), None)
    assert box["video"]["duration_s"] == pytest.approx(30.0)


def test_player_clips_window_around_events():
    events = [Event("shot_made", 300, 10.0, "home:23", 1, value=2, confidence=0.8)]
    clips = player_clips(events, "home:23", 30.0, Config().events)
    assert clips[0]["start_s"] == pytest.approx(6.0)
    assert clips[0]["end_s"] == pytest.approx(13.0)


def test_track_box_lookup():
    t = Track(1, "player", [TrackFrame(5, (0, 0, 10, 10), 0.9)])
    assert t.box_at(5) == (0, 0, 10, 10)
    assert t.box_at(6) is None


def test_calibration_requires_four_points():
    calib = Calibration(image_points={"left_baseline_near_sideline": (0, 0)})
    with pytest.raises(ValueError, match="at least 4"):
        _ = calib.homography
