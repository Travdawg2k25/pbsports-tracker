"""
games_router.py — Purple Box Sports API
========================================
Game upload → player detection → analysis → results flow.

Frontend (upload.html) endpoints:
  POST /api/games/upload                       (multipart video) -> {job_id, video_key}
  POST /api/games/{job_id}/detect-players      -> enqueue scan; poll status for players
  GET  /api/games/{job_id}/players             -> {players:[...]}  (after scan)
  POST /api/games/{job_id}/analyze  {track_id, jersey_number, player_name}
  GET  /api/games/{job_id}/status              -> {status, progress, ...}
  GET  /api/games/{job_id}/stats               -> stats JSON
  GET  /api/games/{job_id}/highlights          -> {reels:{pid: presigned_url}}

Worker (GPU box) internal endpoints  (X-Worker-Secret header):
  GET  /api/internal/next-job
  POST /api/internal/jobs/{job_id}/status   {status, progress, error?}
  POST /api/internal/jobs/{job_id}/players  {players:[...]}
  POST /api/internal/jobs/{job_id}/results  {stats_key, reels:{pid:key}}

Heavy compute runs on the GPU worker; this router only manages jobs, credits,
and S3 objects. Job payloads (player lists, selected appearance signatures)
are stored as JSON in S3 so the worker can read them without DB schema changes.
"""

import json
import uuid
import logging

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header, Request
from fastapi.responses import JSONResponse

from app.auth import get_current_user
from app.database import database, users, game_uploads
from app.config import AWS_BUCKET, AWS_REGION
import os

log = logging.getLogger("PBSports.Games")
router = APIRouter(prefix="/games", tags=["games"])
internal = APIRouter(prefix="/internal", tags=["internal"])

WORKER_SECRET = os.getenv("PB_WORKER_SECRET", "")
s3 = boto3.client("s3", region_name=AWS_REGION)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}


# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────
def _s3_put_json(key: str, obj) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key,
                  Body=json.dumps(obj, default=str).encode(),
                  ContentType="application/json")


def _s3_get_json(key: str):
    try:
        r = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        return json.loads(r["Body"].read())
    except ClientError:
        return None


def _presigned(key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url("get_object",
                                     Params={"Bucket": AWS_BUCKET, "Key": key},
                                     ExpiresIn=expires)


async def _get_job(job_id: str):
    row = await database.fetch_one(
        game_uploads.select().where(game_uploads.c.job_id == job_id))
    return dict(row._mapping) if row else None


# ═════════════════════════════════════════════════════════════════════════════
# FRONTEND ENDPOINTS  (require a logged-in user)
# ═════════════════════════════════════════════════════════════════════════════
@router.post("/upload")
async def upload_game(file: UploadFile = File(...), user=Depends(get_current_user)):
    if user["games_remaining"] <= 0:
        raise HTTPException(402, "No game credits remaining. Purchase a package.")
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Allowed types: {sorted(ALLOWED_EXT)}")

    job_id = str(uuid.uuid4())
    video_key = f"videos/{user['id']}/{job_id}.{ext}"

    # Stream the upload straight to S3 (avoids buffering large files in RAM).
    try:
        s3.upload_fileobj(file.file, AWS_BUCKET, video_key)
    except ClientError as e:
        log.error("S3 upload failed: %s", e)
        raise HTTPException(500, "Upload to storage failed.")

    await database.execute(game_uploads.insert().values(
        user_id=user["id"], job_id=job_id, video_key=video_key,
        status="uploaded", progress=0))
    return {"job_id": job_id, "video_key": video_key}


@router.post("/{job_id}/detect-players")
async def detect_players(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    # Mark as pending-scan; the GPU worker claims 'uploaded' jobs.
    await database.execute(game_uploads.update()
                           .where(game_uploads.c.job_id == job_id)
                           .values(status="uploaded", progress=0))
    return {"status": "queued", "message": "Player detection started. Poll status."}


@router.get("/{job_id}/players")
async def list_players(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    players = _s3_get_json(f"results/{job_id}/players.json") or []
    # Strip heavy appearance vectors from the client payload.
    slim = [{k: v for k, v in p.items() if k != "appearance"} for p in players]
    return {"status": job["status"], "players": slim}


@router.post("/{job_id}/analyze")
async def analyze_game(job_id: str, request: Request, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    body = await request.json()
    track_id = body.get("track_id")
    if track_id is None:
        raise HTTPException(400, "track_id required")
    track_ids = [int(track_id)] if not isinstance(track_id, list) else [int(x) for x in track_id]

    # Spend one credit per analyzed game (atomic-ish check).
    if user["games_remaining"] <= 0:
        raise HTTPException(402, "No game credits remaining.")
    await database.execute(users.update().where(users.c.id == user["id"])
                           .values(games_remaining=user["games_remaining"] - 1))

    # Persist the selected players' appearance signatures for the worker.
    players = _s3_get_json(f"results/{job_id}/players.json") or []
    sel_sigs = {str(p["track_id"]): p.get("appearance")
                for p in players if p["track_id"] in track_ids and p.get("appearance")}
    _s3_put_json(f"results/{job_id}/selected.json",
                 {"selected_players": track_ids, "selected_appearances": sel_sigs})

    await database.execute(game_uploads.update()
                           .where(game_uploads.c.job_id == job_id)
                           .values(status="analyzing", progress=0,
                                   focus_track_id=track_ids[0],
                                   jersey_number=body.get("jersey_number"),
                                   player_name=body.get("player_name")))
    return {"status": "queued", "credits_remaining": user["games_remaining"] - 1}


@router.get("/{job_id}/status")
async def job_status(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "progress": job.get("progress", 0),
            "error": job.get("error_message")}


@router.get("/{job_id}/stats")
async def get_stats(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    stats = _s3_get_json(f"results/{job_id}/stats.json")
    if stats is None:
        raise HTTPException(404, "Stats not ready")
    return stats


@router.get("/{job_id}/highlights")
async def get_highlights(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    reels = _s3_get_json(f"results/{job_id}/reels.json") or {}
    return {"reels": {pid: _presigned(key) for pid, key in reels.items()}}


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL WORKER ENDPOINTS  (require shared secret)
# ═════════════════════════════════════════════════════════════════════════════
def _check_worker(secret: str):
    if not WORKER_SECRET or secret != WORKER_SECRET:
        raise HTTPException(403, "Invalid worker secret")


@internal.get("/next-job")
async def next_job(x_worker_secret: str = Header(default="")):
    _check_worker(x_worker_secret)
    # Prefer analysis jobs, then scans. Claim by flipping status so two workers
    # don't grab the same job.
    for want, kind, claim in (("analyzing", "analyze", "analyzing"),
                              ("uploaded", "scan", "detecting")):
        row = await database.fetch_one(
            game_uploads.select().where(game_uploads.c.status == want)
            .order_by(game_uploads.c.created_at).limit(1))
        if row:
            job = dict(row._mapping)
            payload = {"job_id": job["job_id"], "kind": kind,
                       "video_key": job["video_key"]}
            if kind == "analyze":
                sel = _s3_get_json(f"results/{job['job_id']}/selected.json") or {}
                payload["selected_players"] = sel.get("selected_players", [])
                payload["selected_appearances"] = sel.get("selected_appearances", {})
            return payload
    return {}


@internal.post("/jobs/{job_id}/status")
async def update_status(job_id: str, request: Request, x_worker_secret: str = Header(default="")):
    _check_worker(x_worker_secret)
    body = await request.json()
    vals = {}
    if "status" in body:   vals["status"] = body["status"]
    if "progress" in body: vals["progress"] = int(body["progress"])
    if "error" in body:    vals["error_message"] = body["error"]
    if vals:
        await database.execute(game_uploads.update()
                               .where(game_uploads.c.job_id == job_id).values(**vals))
    return {"ok": True}


@internal.post("/jobs/{job_id}/players")
async def worker_players(job_id: str, request: Request, x_worker_secret: str = Header(default="")):
    _check_worker(x_worker_secret)
    body = await request.json()
    _s3_put_json(f"results/{job_id}/players.json", body.get("players", []))
    return {"ok": True}


@internal.post("/jobs/{job_id}/results")
async def worker_results(job_id: str, request: Request, x_worker_secret: str = Header(default="")):
    _check_worker(x_worker_secret)
    body = await request.json()
    _s3_put_json(f"results/{job_id}/reels.json", body.get("reels", {}))
    return {"ok": True}
