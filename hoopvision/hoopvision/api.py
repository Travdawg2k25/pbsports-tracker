"""HTTP API and web UI.

Two apps share the module: the review app (box score, player stats, clips) and a small
calibration app used once per camera position to mark the court landmarks and the rims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .boxscore import player_clips
from .config import Config
from .court import LANDMARKS_FT, Calibration
from .types import Event, load_json

WEB = Path(__file__).parent / "web"


def create_app(run_dir: str | Path, video: str | Path | None = None) -> FastAPI:
    run = Path(run_dir)
    app = FastAPI(title="hoopvision", version="0.1.0")

    def boxscore() -> dict[str, Any]:
        path = run / "boxscore.json"
        if not path.exists():
            raise HTTPException(404, f"no boxscore in {run}; run `hoopvision analyze` first")
        return load_json(path)

    def events() -> list[Event]:
        return [Event(**e) for e in load_json(run / "events.json")]

    def video_path() -> Path:
        if video:
            return Path(video)
        return Path(load_json(run / "video.json")["path"])

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB / "index.html").read_text()

    @app.get("/api/boxscore")
    def api_boxscore() -> dict[str, Any]:
        return boxscore()

    @app.get("/api/players")
    def api_players() -> list[dict[str, Any]]:
        return [
            {
                "player": p["player"],
                "team": p["team"],
                "jersey": p["jersey"],
                "name": p["name"],
                "points": p["points"],
            }
            for p in boxscore()["players"]
        ]

    @app.get("/api/players/{player}")
    def api_player(player: str) -> dict[str, Any]:
        box = boxscore()
        for p in box["players"]:
            if p["player"] == player:
                fps = box["video"]["fps"]
                return {"stats": p, "clips": player_clips(events(), player, fps, Config().events)}
        raise HTTPException(404, f"unknown player {player}")

    @app.get("/api/events")
    def api_events(kind: str | None = None, player: str | None = None) -> list[dict[str, Any]]:
        out = load_json(run / "events.json")
        if kind:
            out = [e for e in out if e["kind"] == kind]
        if player:
            out = [e for e in out if e["player"] == player]
        return out

    @app.get("/video")
    def api_video() -> FileResponse:
        path = video_path()
        if not path.exists():
            raise HTTPException(404, f"video not found: {path}")
        return FileResponse(path, media_type="video/mp4")

    return app


class CalibrationPayload(BaseModel):
    image_points: dict[str, tuple[float, float]]
    rim_boxes: dict[str, tuple[float, float, float, float]] = {}
    court_length_ft: float = 94.0
    court_width_ft: float = 50.0


def create_calibration_app(frame_image: str | Path, out_path: str | Path) -> FastAPI:
    app = FastAPI(title="hoopvision calibration")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB / "calibrate.html").read_text()

    @app.get("/frame.jpg")
    def frame() -> FileResponse:
        return FileResponse(frame_image, media_type="image/jpeg")

    @app.get("/api/landmarks")
    def landmarks() -> dict[str, tuple[float, float]]:
        return LANDMARKS_FT

    @app.post("/api/calibration")
    def save(payload: CalibrationPayload) -> dict[str, Any]:
        calib = Calibration(
            image_points=dict(payload.image_points),
            rim_boxes=dict(payload.rim_boxes),
            court_length_ft=payload.court_length_ft,
            court_width_ft=payload.court_width_ft,
        )
        try:
            _ = calib.homography  # solvable? reject the marks rather than write them
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        calib.save(out_path)
        return {"saved": str(out_path), "points": len(calib.image_points)}

    return app
