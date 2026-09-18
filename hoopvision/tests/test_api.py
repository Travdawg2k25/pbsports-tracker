from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hoopvision import synthetic
from hoopvision.api import create_app, create_calibration_app
from hoopvision.config import Config
from hoopvision.court import Calibration
from hoopvision.pipeline import stage_boxscore, stage_events
from hoopvision.types import VideoMeta


@pytest.fixture
def run_dir(tmp_path):
    synthetic.generate(tmp_path)
    meta = VideoMeta(**json.loads((tmp_path / "video.json").read_text()))
    stage_events(tmp_path, Config(), synthetic.calibration(), meta)
    stage_boxscore(tmp_path, meta, roster=None)
    return tmp_path


def test_boxscore_and_player_endpoints(run_dir):
    client = TestClient(create_app(run_dir))
    assert client.get("/").status_code == 200

    players = client.get("/api/players").json()
    assert {p["player"] for p in players} >= set(synthetic.expected_boxscore())

    detail = client.get("/api/players/home:23").json()
    assert detail["stats"]["points"] == 2
    kinds = {c["kind"] for c in detail["clips"]}
    assert {"shot_made", "shot_missed"} <= kinds
    for clip in detail["clips"]:
        assert clip["start_s"] < clip["end_s"]

    assert client.get("/api/players/home:99").status_code == 404


def test_events_filter(run_dir):
    client = TestClient(create_app(run_dir))
    made = client.get("/api/events", params={"kind": "shot_made"}).json()
    assert len(made) == 3
    mine = client.get("/api/events", params={"player": "home:23"}).json()
    assert all(e["player"] == "home:23" for e in mine)


def test_video_is_served(run_dir):
    client = TestClient(create_app(run_dir, video=run_dir / "synthetic.mp4"))
    res = client.get("/video")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"


def test_calibration_app_roundtrip(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    out = tmp_path / "calibration.json"
    client = TestClient(create_calibration_app(tmp_path / "synthetic.mp4", out))

    assert "left_baseline_near_sideline" in client.get("/api/landmarks").json()

    truth = synthetic.calibration()
    payload = {
        "image_points": {k: list(v) for k, v in truth.image_points.items()},
        "rim_boxes": {k: list(v) for k, v in truth.rim_boxes.items()},
    }
    assert client.post("/api/calibration", json=payload).status_code == 200

    saved = Calibration.load(out)
    back = saved.to_court(*synthetic.court_to_px(47.0, 25.0))
    assert back[0] == pytest.approx(47.0, abs=0.2)
    assert back[1] == pytest.approx(25.0, abs=0.2)


def test_calibration_rejects_too_few_points(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    client = TestClient(create_calibration_app(tmp_path / "synthetic.mp4", tmp_path / "c.json"))
    res = client.post(
        "/api/calibration", json={"image_points": {"halfcourt_near_sideline": [10, 10]}}
    )
    assert res.status_code == 400
    assert not (tmp_path / "c.json").exists()
