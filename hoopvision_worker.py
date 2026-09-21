#!/usr/bin/env python3
# =============================================================================
# hoopvision_worker.py — GPU-side analysis worker (HoopVision backend)
# =============================================================================
# A drop-in ALTERNATIVE to pbsports_worker.py that speaks the SAME job-queue
# protocol but runs the HoopVision pipeline instead of the legacy one. It does
# NOT modify or replace pbsports_worker.py; run whichever you want.
#
# Why a separate worker: HoopVision is offline/stage-based (motion -> track ->
# teams -> jersey -> events -> boxscore) rather than the legacy frame-streaming
# process_frame() loop, so the analyze path is structurally different. Keeping
# it separate lets the live legacy worker stay untouched and lets you A/B them.
#
# Job lifecycle (identical to the legacy worker):
#   uploaded  -> scan     -> selecting
#   analyzing -> analyze  -> completed
#
# Differences from pbsports_worker.py:
#   * do_analyze runs hoopvision.pipeline.run() once, then translates the box
#     score into the pbsports stats.json shape (pbsports_adapter).
#   * No manual calibration: rims are auto-detected (hoopvision.autorim). Points
#     and FG% are reliable; distance-dependent stats are flagged best-effort.
#   * Highlight reels are written as H.264 (avc1), transcoding via ffmpeg when
#     OpenCV lacks an H.264 encoder, because Chrome refuses to play mp4v.
#   * The parent-entered jersey/name (now forwarded in the next-job payload) are
#     stamped onto the selected player's box score.
#
# Env (same as the legacy worker):
#   PB_API_BASE, PB_WORKER_SECRET, PB_BUCKET, PB_POLL_SECONDS, PB_WORK_DIR,
#   AWS_REGION, PB_SCAN_SECONDS
# HoopVision-specific:
#   HV_DEVICE          (default "cuda:0")
#   HV_RIM_MODEL       (default "/opt/pbsports/basketball_rim_best.pt") — auto-rim ONLY
#   HV_PLAYER_MODEL    (default "yolov8x.pt" — HoopVision's default)
#   HV_BALL_MODEL      (optional COCO ball model; default keeps HoopVision's yolov8x.pt.
#                       Must detect COCO "sports ball" — NOT the custom rim model.)
#   HV_COURT_LENGTH_FT (default 84.0 — high school)
#   HV_MAX_FRAMES      (optional cap for quick tests)
#   FFMPEG_BIN         (optional path to ffmpeg for H.264 transcode)
# =============================================================================

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import boto3
import cv2
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] hv-worker — %(message)s"
)
log = logging.getLogger("hoopvision.worker")

API_BASE = os.environ.get("PB_API_BASE", "https://www.pbsportstech.com/api")
WORKER_SECRET = os.environ.get("PB_WORKER_SECRET", "")
BUCKET = os.environ.get("PB_BUCKET", "pbsports-games-east2")
POLL_SECONDS = int(os.environ.get("PB_POLL_SECONDS", "15"))
WORK_DIR = Path(os.environ.get("PB_WORK_DIR", "/opt/hoopvision-repo/work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = os.environ.get("HV_DEVICE", "cuda:0")
RIM_MODEL = os.environ.get("HV_RIM_MODEL", "/opt/pbsports/basketball_rim_best.pt")
# Player model choice is THE throughput lever (measured on the T4 @ imgsz1536):
# yolov8x 5.8fps, yolov8m 15.6fps, yolov8s 33fps, yolov8n 52fps. Players are large easy
# targets (production uses yolov8n for its scan), so yolov8s gives ~5.7x speedup over the
# yolov8x default at negligible player-detection cost. Ball stays on the basketball model.
PLAYER_MODEL = os.environ.get("HV_PLAYER_MODEL", "yolov8s.pt")
PLAYER_IMGSZ = int(os.environ.get("HV_PLAYER_IMGSZ", "1280"))
# Ball detection model + class. The basketball-trained model gives far better ball
# recall than COCO "sports ball", which is the main accuracy lever for made-shot / FG%.
# Default to the rim model's "basketball" class (0); override to a COCO model + class 32
# if preferred. HV_BALL_MODEL="" forces HoopVision's own yolov8x default.
BALL_MODEL = os.environ.get("HV_BALL_MODEL", RIM_MODEL) or None
BALL_CLASS = int(os.environ.get("HV_BALL_CLASS", "0"))
# Single-pass ball detection (no tiling) — the main throughput lever that doesn't touch
# the tracker. A basketball-trained model finds the ball in one full-frame pass, avoiding
# the several-inferences-per-frame tiling cost. Default on for the basketball model.
BALL_TILED = os.environ.get("HV_BALL_TILED", "0") == "1"
COURT_LENGTH_FT = float(os.environ.get("HV_COURT_LENGTH_FT", "84.0"))
MAX_FRAMES = int(os.environ["HV_MAX_FRAMES"]) if os.environ.get("HV_MAX_FRAMES") else None
# Process every Nth frame. MEASURED: striding fragments the ByteTrack player tracker
# badly — stride 2 gave 147 tracks and stride 3 gave 134, vs 85 at stride 1, because the
# IoU association can't match players across multi-frame jumps. The fragments inflate
# phantom turnovers/steals. So striding is NOT a usable throughput lever here; default 1.
# Throughput comes instead from single-pass ball detection and dead-ball segment skipping.
FRAME_STRIDE = int(os.environ.get("HV_FRAME_STRIDE", "1"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")

HEADERS = {"X-Worker-Secret": WORKER_SECRET}
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))


# ─────────────────────────────────────────────────────────────────────────────
# Web API helpers (identical protocol to pbsports_worker.py)
# ─────────────────────────────────────────────────────────────────────────────
def claim_job():
    try:
        r = requests.get(f"{API_BASE}/internal/next-job", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data if data.get("job_id") else None
    except Exception as e:  # noqa: BLE001
        log.warning("claim_job failed: %s", e)
    return None


def post_status(job_id, status=None, progress=None, error=None):
    body = {}
    if status is not None:
        body["status"] = status
    if progress is not None:
        body["progress"] = int(progress)
    if error is not None:
        body["error"] = str(error)[:1000]
    try:
        requests.post(
            f"{API_BASE}/internal/jobs/{job_id}/status", headers=HEADERS, json=body, timeout=20
        )
    except Exception as e:  # noqa: BLE001
        log.warning("post_status failed: %s", e)


def post_players(job_id, players):
    requests.post(
        f"{API_BASE}/internal/jobs/{job_id}/players",
        headers=HEADERS,
        json={"players": players},
        timeout=60,
    )


def post_results(job_id, stats_key, reels):
    requests.post(
        f"{API_BASE}/internal/jobs/{job_id}/results",
        headers=HEADERS,
        json={"stats_key": stats_key, "reels": reels},
        timeout=60,
    )


# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────
def s3_download(key, dest):
    log.info("S3 download s3://%s/%s -> %s", BUCKET, key, dest)
    s3.download_file(BUCKET, key, str(dest))


def s3_upload(src, key, content_type=None):
    extra = {"ContentType": content_type} if content_type else {}
    s3.upload_file(str(src), BUCKET, key, ExtraArgs=extra)
    log.info("S3 upload %s -> s3://%s/%s", src, BUCKET, key)
    return key


# ─────────────────────────────────────────────────────────────────────────────
# H.264 clip writing — Chrome will not play mp4v, so reels must be avc1/H.264
# ─────────────────────────────────────────────────────────────────────────────
def _opencv_can_avc1(w: int, h: int) -> bool:
    """Whether this OpenCV build can actually open an avc1 writer (often it can't)."""
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        probe = Path(tempfile.gettempdir()) / "_hv_avc1_probe.mp4"
        vw = cv2.VideoWriter(str(probe), fourcc, 30.0, (w, h))
        ok = vw.isOpened()
        vw.release()
        probe.unlink(missing_ok=True)
        return ok
    except Exception:  # noqa: BLE001
        return False


def write_clip(frames, path: Path, fps: float, size: tuple[int, int]) -> bool:
    """Write frames to an H.264 mp4. Falls back to mp4v + ffmpeg transcode.

    Returns True on a playable H.264 file. The two-step fallback exists because most
    prebuilt OpenCV wheels ship without an H.264 encoder, so we write mp4v then let
    ffmpeg remux/transcode to avc1 — the format browsers actually play.
    """
    w, h = size
    if not frames:
        return False

    if _opencv_can_avc1(w, h):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        return path.exists() and path.stat().st_size > 0

    # Fallback: write mp4v to a temp file, then transcode to H.264 with ffmpeg.
    tmp = path.with_suffix(".mp4v.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    if not (tmp.exists() and tmp.stat().st_size > 0):
        return False
    if not FFMPEG_BIN:
        log.error("no ffmpeg for H.264 transcode; leaving mp4v (Chrome may not play it)")
        tmp.replace(path)
        return path.exists()
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(tmp),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if res.returncode != 0:
        log.error("ffmpeg transcode failed: %s", res.stderr[-500:])
        return False
    return path.exists() and path.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Job handlers
# ─────────────────────────────────────────────────────────────────────────────
def do_scan(job):
    """Player-detection scan — reuse the legacy scanner so selection is unchanged."""
    job_id = job["job_id"]
    video_key = job["video_key"]
    post_status(job_id, status="detecting", progress=5)

    import base64

    sys.path.insert(0, "/opt/pbsports")
    from player_select import PlayerScanner

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
        vpath = Path(td) / "game.mp4"
        s3_download(video_key, vpath)
        scanner = PlayerScanner(
            model_path=os.environ.get("PB_PLAYER_MODEL", "/opt/pbsports/yolov8n.pt"),
            device="cuda",
            scan_seconds=float(os.environ.get("PB_SCAN_SECONDS", "20")),
        )
        players = scanner.scan(str(vpath))
        post_status(job_id, progress=80)
        out = []
        for p in players:
            thumb = p.to_thumbnail((150, 200))
            ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode() if ok else ""
            out.append(
                {
                    "track_id": p.track_id,
                    "appearances": p.appearances,
                    "confidence": round(p.confidence, 3),
                    "thumbnail": b64,
                    "appearance": _appearance_sig(p.crop),
                }
            )
        post_players(job_id, out)
    post_status(job_id, status="selecting", progress=100)
    log.info("scan complete job=%s players=%d", job_id, len(out))


def do_analyze(job):
    """Full HoopVision analysis of the selected player: stats + H.264 reels."""
    job_id = job["job_id"]
    video_key = job["video_key"]
    jersey_number = job.get("jersey_number")
    player_name = job.get("player_name")
    post_status(job_id, status="analyzing", progress=2)

    # HoopVision package (installed in the box venv this worker runs under).
    from hoopvision.autorim import AutoRimConfig, detect_rims
    from hoopvision.boxscore import player_clips
    from hoopvision.config import Config
    from hoopvision.pbsports_adapter import to_pbsports_stats
    from hoopvision.pipeline import run as hv_run
    from hoopvision.types import Event, load_json
    from hoopvision.video import probe

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
        td = Path(td)
        vpath = td / "game.mp4"
        s3_download(video_key, vpath)
        meta = probe(str(vpath))

        # 1. Auto-detect rims -> rim-only calibration (no parent calibration step).
        post_status(job_id, progress=8)
        calib = detect_rims(
            str(vpath),
            AutoRimConfig(model=RIM_MODEL, device=DEVICE),
            court_length_ft=COURT_LENGTH_FT,
        )
        if calib is None:
            raise RuntimeError(
                "auto-rim found no stable rim; a manual calibration step is needed for "
                "this footage"
            )
        calib_path = td / "calibration.json"
        calib.save(calib_path)

        # 2. Run the HoopVision pipeline once (offline, stage-based).
        cfg = Config()
        cfg.detection.device = DEVICE
        cfg.detection.half = DEVICE != "cpu"
        cfg.detection.model = PLAYER_MODEL
        cfg.detection.player_imgsz = PLAYER_IMGSZ
        cfg.frame_stride = FRAME_STRIDE
        # Made-shot detection counts descending *sampled* frames through the rim. Striding
        # makes each sampled frame span FRAME_STRIDE real frames, so the descent
        # requirement must scale down or real makes are missed. A shot descends through
        # the rim in ~6-8 real frames; keep at least 2 sampled frames of descent.
        if FRAME_STRIDE > 1:
            cfg.events.made_descent_frames = max(2, cfg.events.made_descent_frames // FRAME_STRIDE)
            cfg.events.possession_min_frames = max(
                2, cfg.events.possession_min_frames // FRAME_STRIDE
            )
        # Ball detection: use the basketball-trained model + its class, so the ball
        # track is clean enough for made-shot detection. detect_ball filters by
        # cfg.detection.ball_class, so model and class must agree.
        if BALL_MODEL:
            cfg.detection.ball_model = BALL_MODEL
            cfg.detection.ball_class = BALL_CLASS
        cfg.detection.ball_tiled = BALL_TILED
        post_status(job_id, progress=15)
        boxscore = hv_run(
            video=str(vpath),
            out_dir=str(td / "run"),
            calibration=str(calib_path),
            config=cfg,
            max_frames=MAX_FRAMES,
        )
        post_status(job_id, progress=80)

        # 3. Translate to the pbsports stats.json shape, stamping parent jersey/name
        #    onto the top-scoring player (the one the parent selected/filmed).
        focus_key = boxscore["players"][0]["player"] if boxscore["players"] else None
        stats = to_pbsports_stats(
            boxscore,
            game_id=job_id,
            distance_reliable=calib.distance_reliable,
            focus_player_key=focus_key,
            jersey_number=jersey_number,
            player_name=player_name,
            duration_sec=meta.duration_s,
            total_frames=meta.frame_count,
        )
        stats_path = td / "stats.json"
        stats_path.write_text(json.dumps(stats, default=str, indent=2))
        stats_key = f"results/{job_id}/stats.json"
        s3_upload(stats_path, stats_key, "application/json")

        # 4. Cut an H.264 highlight reel for the focus player from its event windows.
        reel_keys = {}
        if focus_key:
            events = [Event(**e) for e in load_json(td / "run" / "events.json")]
            clips = player_clips(events, focus_key, meta.fps, cfg.events)
            frames = _gather_clip_frames(str(vpath), clips, meta.fps)
            if frames:
                # Use the focus player's own track_id key for the reel filename.
                focus_pid = _focus_pid(stats, jersey_number, player_name)
                reel_path = td / f"reel_{focus_pid}.mp4"
                if write_clip(frames, reel_path, meta.fps, (meta.width, meta.height)):
                    k = f"results/{job_id}/reel_{focus_pid}.mp4"
                    s3_upload(reel_path, k, "video/mp4")
                    reel_keys[str(focus_pid)] = k

        post_results(job_id, stats_key, reel_keys)
    post_status(job_id, status="completed", progress=100)
    log.info("analyze complete job=%s reels=%d", job_id, len(reel_keys))


def _focus_pid(stats, jersey_number, player_name):
    """The track_id key of the player the parent selected (matched by stamped name/jersey)."""
    for pid, row in stats["players"].items():
        if player_name and row.get("name") == player_name:
            return pid
        if jersey_number and row.get("jersey_number") == jersey_number:
            return pid
    # Fall back to the top-scoring player.
    return next(iter(stats["players"]), "0")


def _gather_clip_frames(video: str, clips, fps: float):
    """Collect the frames spanning every clip window, in order, deduped."""
    from hoopvision.video import FrameSeeker

    wanted: list[tuple[int, int]] = []
    for c in clips:
        s = int(c["start_s"] * fps)
        e = int(c["end_s"] * fps)
        wanted.append((max(0, s), max(0, e)))
    wanted.sort()
    frames = []
    if not wanted:
        return frames
    with FrameSeeker(video) as seeker:
        for s, e in wanted:
            for fidx in range(s, e + 1):
                img = seeker.get(fidx)
                if img is not None:
                    frames.append(img)
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Appearance signature (kept identical to the web/legacy helper for parity)
# ─────────────────────────────────────────────────────────────────────────────
def _appearance_sig(crop):
    try:
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        hh, ww = crop.shape[:2]
        torso = crop[int(hh * 0.15) : int(hh * 0.55), int(ww * 0.15) : int(ww * 0.85)]
        if torso.size == 0:
            torso = crop
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return [float(x) for x in hist.flatten()]
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not WORKER_SECRET:
        log.error("PB_WORKER_SECRET not set — refusing to start.")
        sys.exit(1)
    log.info(
        "HoopVision worker started. API=%s bucket=%s poll=%ss device=%s",
        API_BASE, BUCKET, POLL_SECONDS, DEVICE,
    )
    while True:
        job = claim_job()
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        jid = job["job_id"]
        kind = job.get("kind", "scan")
        log.info("Claimed job=%s kind=%s", jid, kind)
        try:
            if kind == "scan":
                do_scan(job)
            elif kind == "analyze":
                do_analyze(job)
            else:
                post_status(jid, status="failed", error=f"unknown job kind {kind}")
        except Exception as e:  # noqa: BLE001
            log.error("Job %s failed: %s\n%s", jid, e, traceback.format_exc())
            post_status(jid, status="failed", error=str(e))


if __name__ == "__main__":
    main()
