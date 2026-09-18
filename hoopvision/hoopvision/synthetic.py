"""Synthetic game generator.

Renders a scripted possession sequence — players, jersey numbers, ball, rims — to an mp4
and emits the matching ground-truth tracks. It exists so the geometry half of the
pipeline (possession, shot classification, rebounds, box score, UI) can be exercised and
regression-tested without a real game file, and so the expected box score is known
exactly.

It is *not* a substitute for validating detection and OCR on real footage.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .court import COURT_LENGTH_FT, COURT_WIDTH_FT, HOOPS_FT, Calibration
from .types import Track, TrackFrame, dump_json

W, H = 1280, 720
MARGIN = 60
FPS = 30.0


def _to_h264(path: Path) -> None:
    """Re-encode in place: browsers refuse the MPEG-4 Part 2 stream OpenCV writes."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = path.with_suffix(".h264.mp4")
    done = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=False,
    )
    if done.returncode == 0 and tmp.exists():
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


def court_to_px(x_ft: float, y_ft: float) -> tuple[float, float]:
    """Plan view: the court fills the frame with a margin. Static, like the real camera."""
    sx = (W - 2 * MARGIN) / COURT_LENGTH_FT
    sy = (H - 2 * MARGIN) / COURT_WIDTH_FT
    return MARGIN + x_ft * sx, MARGIN + y_ft * sy


@dataclass
class ScriptedShot:
    shooter: int  # index into players
    passer: int | None
    made: bool
    from_ft: tuple[float, float]
    hoop: str
    rebounder: int | None


PLAYERS = [
    # (jersey, team, colour BGR, home end)
    ("23", "home", (235, 235, 235)),
    ("7", "home", (235, 235, 235)),
    ("11", "home", (235, 235, 235)),
    ("4", "away", (60, 60, 60)),
    ("15", "away", (60, 60, 60)),
]

SCRIPT: list[ScriptedShot] = [
    ScriptedShot(0, passer=1, made=True, from_ft=(70.0, 25.0), hoop="right", rebounder=None),
    ScriptedShot(0, passer=None, made=False, from_ft=(64.0, 12.0), hoop="right", rebounder=2),
    ScriptedShot(2, passer=None, made=True, from_ft=(84.0, 22.0), hoop="right", rebounder=None),
    ScriptedShot(3, passer=4, made=True, from_ft=(24.0, 25.0), hoop="left", rebounder=None),
    ScriptedShot(4, passer=None, made=False, from_ft=(18.0, 33.0), hoop="left", rebounder=0),
]

POSSESSION_FRAMES = 90  # 3 s per scripted possession
TAIL_FRAMES = 45  # extra frames after the final shot so the last rebound is visible


def expected_boxscore() -> dict[str, dict[str, int]]:
    """Ground truth the pipeline should reproduce from the rendered video."""
    out: dict[str, dict[str, int]] = {}

    def line(i: int) -> dict[str, int]:
        jersey, team, _ = PLAYERS[i]
        return out.setdefault(
            f"{team}:{jersey}",
            {"points": 0, "fgm": 0, "fga": 0, "rebounds": 0, "assists": 0},
        )

    for shot in SCRIPT:
        value = 3 if _distance(shot.from_ft, shot.hoop) >= 22.15 else 2
        s = line(shot.shooter)
        s["fga"] += 1
        if shot.made:
            s["fgm"] += 1
            s["points"] += value
            if shot.passer is not None:
                line(shot.passer)["assists"] += 1
        elif shot.rebounder is not None:
            line(shot.rebounder)["rebounds"] += 1
    return out


def _distance(pt: tuple[float, float], hoop: str) -> float:
    hx, hy = HOOPS_FT[hoop]
    return math.hypot(pt[0] - hx, pt[1] - hy)


def generate(
    out_dir: str | Path, write_tracks: bool = True, pan_px: float = 0.0
) -> dict[str, Path]:
    """Render the clip, and optionally the ground-truth tracks and calibration.

    ``pan_px`` swings the camera horizontally by up to that many pixels, which makes the
    calibration frame-dependent and exercises the motion compensation.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    player_frames: dict[int, list[TrackFrame]] = {i: [] for i in range(len(PLAYERS))}
    ball_frames: list[TrackFrame] = []
    frame_idx = 0

    for shot in SCRIPT:
        # A tail after the last possession so the closing rebound has somewhere to happen.
        length = POSSESSION_FRAMES + (TAIL_FRAMES if shot is SCRIPT[-1] else 0)
        for t in range(length):
            img = _draw_court()
            phase = min(t / POSSESSION_FRAMES, 1.0)
            positions = _player_positions(shot, phase)
            boxes = {}
            for i, pos in positions.items():
                boxes[i] = _draw_player(img, pos, PLAYERS[i])
            ball_px = _ball_position(shot, phase, positions)
            ball_box = _draw_ball(img, ball_px)

            dx = _pan_offset(frame_idx, pan_px)
            if dx:
                img = cv2.warpAffine(
                    img, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderValue=(46, 82, 128)
                )
            for i, box in boxes.items():
                player_frames[i].append(TrackFrame(frame_idx, _shift(box, dx), 0.95))
            ball_frames.append(TrackFrame(frame_idx, _shift(ball_box, dx), 0.9))
            writer.write(img)
            frame_idx += 1

    writer.release()
    _to_h264(video_path)

    paths = {"video": video_path}
    if write_tracks:
        tracks = [
            Track(track_id=i + 1, cls="player", frames=frames)
            for i, frames in player_frames.items()
        ]
        dump_json(tracks, out / "tracks.json")
        dump_json(Track(track_id=0, cls="ball", frames=ball_frames), out / "ball.json")
        dump_json(
            [
                {
                    "track_id": i + 1,
                    "team": PLAYERS[i][1],
                    "jersey": PLAYERS[i][0],
                    "jersey_confidence": 1.0,
                    "votes": {PLAYERS[i][0]: 10},
                }
                for i in range(len(PLAYERS))
            ],
            out / "identities.json",
        )
        dump_json(
            {
                "teams": {str(i + 1): PLAYERS[i][1] for i in range(len(PLAYERS))},
                "kit_separation": 9.9,
            },
            out / "teams.json",
        )
        dump_json(
            {
                "path": str(video_path),
                "fps": FPS,
                "width": W,
                "height": H,
                "frame_count": frame_idx,
            },
            out / "video.json",
        )
        calibration(out / "calibration.json")
        paths["tracks"] = out / "tracks.json"
        paths["calibration"] = out / "calibration.json"
    return paths


def calibration(path: str | Path | None = None) -> Calibration:
    """Exact calibration for the synthetic camera."""
    pts = {
        "left_baseline_near_sideline": court_to_px(0, 0),
        "left_baseline_far_sideline": court_to_px(0, COURT_WIDTH_FT),
        "right_baseline_near_sideline": court_to_px(COURT_LENGTH_FT, 0),
        "right_baseline_far_sideline": court_to_px(COURT_LENGTH_FT, COURT_WIDTH_FT),
        "halfcourt_near_sideline": court_to_px(COURT_LENGTH_FT / 2, 0),
        "halfcourt_far_sideline": court_to_px(COURT_LENGTH_FT / 2, COURT_WIDTH_FT),
    }
    rims = {}
    for side, (hx, hy) in HOOPS_FT.items():
        cx, cy = court_to_px(hx, hy)
        rims[side] = (cx - 12, cy - 9, cx + 12, cy + 9)
    calib = Calibration(image_points=pts, rim_boxes=rims)
    if path:
        calib.save(path)
    return calib


def _pan_offset(frame_idx: int, pan_px: float) -> float:
    if not pan_px:
        return 0.0
    return pan_px * math.sin(2 * math.pi * frame_idx / 240.0)


def _shift(box: tuple, dx: float) -> tuple:
    return (box[0] + dx, box[1], box[2] + dx, box[3])


def _draw_court() -> np.ndarray:
    img = np.full((H, W, 3), (46, 82, 128), dtype=np.uint8)
    # Static clutter off the floor: a bare court gives a feature matcher almost nothing
    # to lock onto, whereas a real gym has seats, banners and line judges.
    rng = np.random.default_rng(7)
    for x, y in rng.integers([0, 0], [W, MARGIN - 8], size=(140, 2)):
        cv2.circle(img, (int(x), int(y)), 3, (200, 200, 200), -1)
        cv2.circle(img, (int(x), int(H - y - 1)), 3, (170, 190, 210), -1)
    corners = [
        court_to_px(0, 0),
        court_to_px(COURT_LENGTH_FT, 0),
        court_to_px(COURT_LENGTH_FT, COURT_WIDTH_FT),
        court_to_px(0, COURT_WIDTH_FT),
    ]
    cv2.polylines(img, [np.int32(corners)], True, (240, 240, 240), 2)
    mid_a, mid_b = court_to_px(COURT_LENGTH_FT / 2, 0), court_to_px(COURT_LENGTH_FT / 2, 50)
    cv2.line(img, np.int32(mid_a), np.int32(mid_b), (240, 240, 240), 2)
    for hx, hy in HOOPS_FT.values():
        cv2.circle(img, np.int32(court_to_px(hx, hy)), 8, (0, 140, 255), 2)
    return img


def _player_positions(shot: ScriptedShot, phase: float) -> dict[int, tuple[float, float]]:
    """Everyone drifts toward the active hoop; the shooter stands at the shot spot."""
    hx, hy = HOOPS_FT[shot.hoop]
    out: dict[int, tuple[float, float]] = {}
    for i, _ in enumerate(PLAYERS):
        if i == shot.shooter:
            out[i] = shot.from_ft
            continue
        if i == shot.passer:
            # Passer stands a few feet away until the pass leaves at 40% of the possession.
            out[i] = (shot.from_ft[0] - 10, shot.from_ft[1] + 6)
            continue
        if i == shot.rebounder:
            out[i] = (hx + (6 if hx < 47 else -6), hy + 2)
            continue
        spread = 8 + 4 * i
        angle = phase * 2 * math.pi + i
        cx = hx + (20 if hx < 47 else -20)
        out[i] = (
            float(np.clip(cx + spread * math.cos(angle), 2, COURT_LENGTH_FT - 2)),
            float(np.clip(hy + spread * math.sin(angle), 2, COURT_WIDTH_FT - 2)),
        )
    return out


def _ball_position(
    shot: ScriptedShot, phase: float, positions: dict[int, tuple[float, float]]
) -> tuple[float, float]:
    """Ball is with the passer, then the shooter, then arcs to the rim.

    The flight is split so the ball arrives *above* the rim and then drops vertically:
    a diagonal line into the hoop would never look like a made basket to the rim logic,
    and it does not look like one on real footage either.
    """
    hx, hy = HOOPS_FT[shot.hoop]
    rim_px = court_to_px(hx, hy)
    if phase < 0.35 and shot.passer is not None:
        return _at_player(positions[shot.passer])
    if phase < 0.6:
        return _at_player(positions[shot.shooter])

    start = _at_player(positions[shot.shooter])
    front = rim_px[0] - 30 if hx > COURT_LENGTH_FT / 2 else rim_px[0] + 30
    apex = (rim_px[0], rim_px[1] - 55) if shot.made else (front, rim_px[1] - 38)

    if phase < 0.85:
        t = (phase - 0.6) / 0.25
        return (
            start[0] + (apex[0] - start[0]) * t,
            start[1] + (apex[1] - start[1]) * t - 70 * math.sin(math.pi * t),
        )

    t = (phase - 0.85) / 0.15
    if shot.made:
        # Straight down the cylinder: above the rim, inside it, then below it.
        return (apex[0], apex[1] + (rim_px[1] + 45 - apex[1]) * t)
    # A clank off the front rim: it bounces back out, never through the hoop, and lands
    # with whoever is scripted to grab the board.
    if shot.rebounder is not None:
        away = _at_player(positions[shot.rebounder])
    else:
        away = (front - 90 if hx > COURT_LENGTH_FT / 2 else front + 90, apex[1])
    return (
        apex[0] + (away[0] - apex[0]) * t,
        apex[1] + (away[1] - apex[1]) * t + 30 * math.sin(math.pi * t),
    )


def _at_player(pos_ft: tuple[float, float]) -> tuple[float, float]:
    px, py = court_to_px(*pos_ft)
    return px + 14, py - 26


def _draw_player(img: np.ndarray, pos_ft: tuple[float, float], player) -> tuple:
    jersey, _team, colour = player
    px, py = court_to_px(*pos_ft)
    w, h = 26, 64
    x1, y1, x2, y2 = px - w / 2, py - h, px + w / 2, py
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, -1)
    text_colour = (20, 20, 20) if sum(colour) > 380 else (240, 240, 240)
    cv2.putText(
        img, jersey, (int(x1) + 3, int(y1) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_colour, 2
    )
    return (x1, y1, x2, y2)


def _draw_ball(img: np.ndarray, px: tuple[float, float]) -> tuple:
    r = 9
    cv2.circle(img, (int(px[0]), int(px[1])), r, (30, 120, 235), -1)
    return (px[0] - r, px[1] - r, px[0] + r, px[1] + r)
