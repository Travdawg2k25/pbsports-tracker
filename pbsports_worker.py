#!/usr/bin/env python3
# =============================================================================
# pbsports_worker.py — GPU-side analysis worker
# =============================================================================
# Runs on the GPU box. Polls the web API for jobs, pulls the video from S3,
# runs the tracker pipeline, and uploads results (thumbnails / stats / reels)
# back to S3, posting status updates to the web API as it goes.
#
# Job lifecycle (status column on the web side):
#   uploaded   -> worker claims, runs player scan  -> selecting  (players ready)
#   analyzing  -> worker claims, runs full analysis -> completed  (stats+reels)
#
# Communication:
#   - Web API base:      PB_API_BASE   (e.g. https://www.pbsportstech.com/api)
#   - Shared secret:     PB_WORKER_SECRET  (sent as X-Worker-Secret header)
#   - S3 bucket:         PB_BUCKET     (videos in videos/, results in results/)
#
# The web API exposes these internal endpoints (see games_router.py):
#   GET  /api/internal/next-job            -> claim a pending job or {}
#   POST /api/internal/jobs/{job_id}/status  {status, progress, error?}
#   POST /api/internal/jobs/{job_id}/players {players:[...]}
#   POST /api/internal/jobs/{job_id}/results {stats_key, reels:{pid:key}}
# All require the X-Worker-Secret header.
# =============================================================================

import os
import sys
import time
import json
import tempfile
import logging
import traceback
from pathlib import Path

import requests
import boto3
import cv2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] worker — %(message)s")
log = logging.getLogger("pbsports.worker")

API_BASE      = os.environ.get("PB_API_BASE", "https://www.pbsportstech.com/api")
WORKER_SECRET = os.environ.get("PB_WORKER_SECRET", "")
BUCKET        = os.environ.get("PB_BUCKET", "pbsports-games-east2")
POLL_SECONDS  = int(os.environ.get("PB_POLL_SECONDS", "15"))
WORK_DIR      = Path(os.environ.get("PB_WORK_DIR", "/opt/pbsports/work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"X-Worker-Secret": WORKER_SECRET}
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))

# Lazy pipeline imports (heavy) — only load when a job actually runs.
sys.path.insert(0, "/opt/pbsports")


# ─────────────────────────────────────────────────────────────────────────────
# Web API helpers
# ─────────────────────────────────────────────────────────────────────────────
def claim_job():
    try:
        r = requests.get(f"{API_BASE}/internal/next-job", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data if data.get("job_id") else None
    except Exception as e:
        log.warning("claim_job failed: %s", e)
    return None


def post_status(job_id, status=None, progress=None, error=None):
    body = {}
    if status is not None:   body["status"] = status
    if progress is not None: body["progress"] = int(progress)
    if error is not None:    body["error"] = str(error)[:1000]
    try:
        requests.post(f"{API_BASE}/internal/jobs/{job_id}/status",
                      headers=HEADERS, json=body, timeout=20)
    except Exception as e:
        log.warning("post_status failed: %s", e)


def post_players(job_id, players):
    requests.post(f"{API_BASE}/internal/jobs/{job_id}/players",
                  headers=HEADERS, json={"players": players}, timeout=60)


def post_results(job_id, stats_key, reels):
    requests.post(f"{API_BASE}/internal/jobs/{job_id}/results",
                  headers=HEADERS, json={"stats_key": stats_key, "reels": reels}, timeout=60)


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
# Job handlers
# ─────────────────────────────────────────────────────────────────────────────
def do_scan(job):
    """Player-detection scan: produce selectable player thumbnails."""
    job_id = job["job_id"]
    video_key = job["video_key"]
    post_status(job_id, status="detecting", progress=5)

    from player_select import PlayerScanner
    import base64

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
        vpath = Path(td) / "game.mp4"
        s3_download(video_key, vpath)

        scanner = PlayerScanner(
            model_path="/opt/pbsports/yolov8n.pt",
            device="cuda",
            scan_seconds=float(os.environ.get("PB_SCAN_SECONDS", "20")),
        )
        players = scanner.scan(str(vpath))
        post_status(job_id, progress=80)

        out = []
        for p in players:
            # thumbnail as base64 (frontend renders inline), plus appearance sig
            thumb = p.to_thumbnail((150, 200))
            ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode() if ok else ""
            out.append({
                "track_id": p.track_id,
                "appearances": p.appearances,
                "confidence": round(p.confidence, 3),
                "thumbnail": b64,
                "appearance": _appearance_sig(p.crop),
            })

        post_players(job_id, out)
    post_status(job_id, status="selecting", progress=100)
    log.info("scan complete job=%s players=%d", job_id, len(out))


def do_analyze(job):
    """Full analysis of the selected player(s): stats + highlight reels."""
    job_id = job["job_id"]
    video_key = job["video_key"]
    selected = job.get("selected_players") or []
    sel_sigs = job.get("selected_appearances") or {}   # {track_id: sig}
    post_status(job_id, status="analyzing", progress=2)

    from pipeline import AnalysisPipeline

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
        td = Path(td)
        vpath = td / "game.mp4"
        s3_download(video_key, vpath)

        cfg = {"device": "cuda", "player_model": "/opt/pbsports/yolov8n.pt",
               "ball_model": "/opt/pbsports/basketball_rim_best.pt", "fps": 30.0}
        pipeline = AnalysisPipeline(cfg)

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        reels, reel_paths = {}, {}
        pre_buf, post_hold = int(fps * 3), int(fps * 2)
        recent, record_until = [], {int(pid): -1 for pid in selected}
        NEAR = 140.0

        def ensure_reel(pid):
            if pid not in reels:
                rp = td / f"reel_{pid}.mp4"
                reel_paths[pid] = rp
                reels[pid] = cv2.VideoWriter(str(rp), fourcc, fps, (w, h))
                for fb in recent:
                    reels[pid].write(fb)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            res = pipeline.process_frame(frame, timestamp=idx / fps)
            ann = res["annotated_frame"]
            ball = res.get("ball_position")
            recent.append(ann)
            if len(recent) > pre_buf:
                recent.pop(0)

            matched = {}
            if sel_sigs and pipeline.player_tracker:
                for t in pipeline.player_tracker.active_tracks:
                    x1, y1, x2, y2 = (int(v) for v in t.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 < 8 or y2 - y1 < 16:
                        continue
                    tsig = _appearance_sig(frame[y1:y2, x1:x2])
                    if not tsig:
                        continue
                    for sid, ssig in sel_sigs.items():
                        d = _appearance_dist(ssig, tsig)
                        if d <= 0.35:
                            prev = matched.get(int(sid))
                            if prev is None or d < prev[1]:
                                matched[int(sid)] = (t, d)

            for ev in res["events"]:
                bp = ev.get("ball_position") or ball
                if not bp:
                    continue
                for sid, (t, _d) in matched.items():
                    cx, cy = t.center
                    if ((cx - bp[0]) ** 2 + (cy - bp[1]) ** 2) ** 0.5 <= NEAR * 1.5:
                        record_until[sid] = idx + post_hold
            if ball:
                for sid, (t, _d) in matched.items():
                    cx, cy = t.center
                    if ((cx - ball[0]) ** 2 + (cy - ball[1]) ** 2) ** 0.5 <= NEAR:
                        record_until[sid] = idx + post_hold

            for pid, until in record_until.items():
                if idx <= until:
                    ensure_reel(pid)
                    reels[pid].write(ann)

            if idx % 120 == 0:
                post_status(job_id, progress=min(95, int(idx / total * 90) + 2))

        cap.release()
        for wr in reels.values():
            wr.release()

        # Upload results to S3
        summary = pipeline.get_stats()
        stats_path = td / "stats.json"
        stats_path.write_text(json.dumps(summary, default=str, indent=2))
        stats_key = f"results/{job_id}/stats.json"
        s3_upload(stats_path, stats_key, "application/json")

        reel_keys = {}
        for pid, rp in reel_paths.items():
            if rp.exists() and rp.stat().st_size > 0:
                k = f"results/{job_id}/reel_{pid}.mp4"
                s3_upload(rp, k, "video/mp4")
                reel_keys[str(pid)] = k

        post_results(job_id, stats_key, reel_keys)
    post_status(job_id, status="completed", progress=100)
    log.info("analyze complete job=%s reels=%d", job_id, len(reel_keys))


# ─────────────────────────────────────────────────────────────────────────────
# Appearance signature (must match the web-side helper)
# ─────────────────────────────────────────────────────────────────────────────
def _appearance_sig(crop):
    try:
        import numpy as np
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        hh, ww = crop.shape[:2]
        torso = crop[int(hh * 0.15):int(hh * 0.55), int(ww * 0.15):int(ww * 0.85)]
        if torso.size == 0:
            torso = crop
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return [float(x) for x in hist.flatten()]
    except Exception:
        return None


def _appearance_dist(a, b):
    import numpy as np
    aa = np.array(a, dtype="float32")
    bb = np.array(b, dtype="float32")
    return 1.0 - cv2.compareHist(aa, bb, cv2.HISTCMP_CORREL)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not WORKER_SECRET:
        log.error("PB_WORKER_SECRET not set — refusing to start.")
        sys.exit(1)
    log.info("Worker started. API=%s bucket=%s poll=%ss", API_BASE, BUCKET, POLL_SECONDS)
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
        except Exception as e:
            log.error("Job %s failed: %s\n%s", jid, e, traceback.format_exc())
            post_status(jid, status="failed", error=str(e))


if __name__ == "__main__":
    main()
