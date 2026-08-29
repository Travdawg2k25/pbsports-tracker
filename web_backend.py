# web_backend.py — Purple Box Sports | Web Backend
# =============================================================================
# End-to-end product flow:
#
#   1. Sign up / buy a package      → POST /api/signup           (grants credits)
#   2. Upload a game video          → POST /api/games            (creates a job,
#                                                                  scans players)
#   3. See detected players         → GET  /api/games/<id>/players
#   4. Select player(s) & analyze   → POST /api/games/<id>/analyze (spends 1 credit)
#   5. Poll job status              → GET  /api/games/<id>
#   6. Download stats               → GET  /api/games/<id>/stats
#   7. Download highlight reel      → GET  /api/games/<id>/players/<pid>/reel
#
# Authentication is a simple API key (returned at signup) sent as the
# `X-API-Key` header. This is an MVP — swap for real auth / a DB before
# production, but the flow and storage layout are production-shaped.
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify, send_file, abort

logger = logging.getLogger("PurpleBox.WebBackend")
logging.basicConfig(level=logging.INFO)

# ── Storage layout ────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.environ.get("PB_DATA_DIR", "data")).resolve()
UPLOAD_DIR  = DATA_DIR / "uploads"
JOBS_DIR    = DATA_DIR / "jobs"          # per-game outputs (stats, reels)
USERS_FILE  = DATA_DIR / "users.json"
GAMES_FILE  = DATA_DIR / "games.json"

for d in (DATA_DIR, UPLOAD_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_UPLOAD_BYTES = int(os.environ.get("PB_MAX_UPLOAD_MB", "2048")) * 1024 * 1024

# ── Package definitions (credits = games that can be analyzed) ────────────────
PACKAGES = {
    "starter":  {"games": 5,  "price_usd": 49},
    "team":     {"games": 15, "price_usd": 129},
    "season":   {"games": 40, "price_usd": 299},
}


# =============================================================================
# TINY JSON-BACKED STORE  (thread-safe)
# =============================================================================
class JsonStore:
    """Minimal thread-safe JSON persistence for users and games."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)
        else:
            self._data = {}
            self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        tmp.replace(self.path)

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            v = self._data.get(key)
            return dict(v) if isinstance(v, dict) else v

    def put(self, key: str, value: Dict) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def update(self, key: str, **changes) -> Optional[Dict]:
        with self._lock:
            v = self._data.get(key)
            if v is None:
                return None
            v.update(changes)
            self._flush()
            return dict(v)

    def all(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._data)


users = JsonStore(USERS_FILE)
games = JsonStore(GAMES_FILE)


# =============================================================================
# ANALYSIS WORKER  (runs the heavy pipeline in a background thread)
# =============================================================================
# Lazy imports so the server starts fast and doesn't require torch until a
# job actually runs.
_scanner_lock = threading.Lock()
_scanner = None


def _get_scanner():
    """Load the PlayerScanner once and reuse it (model load is expensive)."""
    global _scanner
    with _scanner_lock:
        if _scanner is None:
            from player_select import PlayerScanner
            _scanner = PlayerScanner(
                model_path=os.environ.get("PB_PLAYER_MODEL", "yolov8n.pt"),
                device=os.environ.get("PB_DEVICE", "cpu"),
                scan_seconds=float(os.environ.get("PB_SCAN_SECONDS", "20")),
            )
        return _scanner


def _appearance_signature(crop) -> Optional[List[float]]:
    """
    Compute a normalized HSV color histogram from a player crop.
    Returns a flat list of floats (JSON-serializable) or None.
    This is the stable cross-pass identity: a player's jersey/skin/short
    colors are consistent whether they're tracked in the scan pass or the
    analysis pass, even though the tracker assigns different numeric IDs.
    """
    try:
        import cv2
        import numpy as np
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        # Focus on the torso band so the jersey dominates the signature.
        h, w = crop.shape[:2]
        torso = crop[int(h * 0.15):int(h * 0.55), int(w * 0.15):int(w * 0.85)]
        if torso.size == 0:
            torso = crop
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return [float(x) for x in hist.flatten()]
    except Exception:
        return None


def _appearance_distance(sig_a: List[float], sig_b: List[float]) -> float:
    """Correlation-based similarity → distance in [0,2]; lower is more similar."""
    import numpy as np
    a = np.array(sig_a, dtype="float32")
    b = np.array(sig_b, dtype="float32")
    import cv2
    corr = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)  # 1 = identical
    return 1.0 - corr


def _scan_players_job(game_id: str, video_path: str) -> None:
    """Background: scan the uploaded video for selectable players."""
    try:
        games.update(game_id, status="scanning")
        scanner = _get_scanner()
        players = scanner.scan(video_path)

        job_dir = JOBS_DIR / game_id
        job_dir.mkdir(parents=True, exist_ok=True)

        import cv2
        player_list = []
        for p in players:
            thumb_name = f"player_{p.track_id:03d}.jpg"
            cv2.imwrite(str(job_dir / thumb_name), p.to_thumbnail((150, 200)))
            # Appearance signature (HSV color histogram of the player's crop).
            # This is the stable identity we use to re-find the player during
            # analysis, since tracker IDs differ between the scan pass and the
            # analysis pass.
            sig = _appearance_signature(p.crop)
            player_list.append({
                "track_id":    p.track_id,
                "appearances": p.appearances,
                "confidence":  round(p.confidence, 3),
                "thumbnail":   f"/api/games/{game_id}/players/{p.track_id}/thumb",
                "bbox":        list(p.bbox),
                "appearance":  sig,   # list[float] or None
            })

        games.update(
            game_id,
            status="players_ready",
            players=player_list,
            players_found=len(player_list),
        )
        logger.info("Scan done: game=%s players=%d", game_id, len(player_list))
    except Exception as e:
        logger.exception("Scan failed for game %s", game_id)
        games.update(game_id, status="error", error=str(e))


def _analyze_job(game_id: str, video_path: str, selected_ids: List[int]) -> None:
    """Background: run full pipeline, write stats + per-player highlight reels."""
    try:
        games.update(game_id, status="analyzing", progress=0.0)

        job_dir = JOBS_DIR / game_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Load the appearance signatures of the selected players (captured at
        # scan time). We match analysis-pass tracks against these because the
        # scan and analysis use different tracker id spaces.
        g = games.get(game_id) or {}
        sel_sigs = {}   # scan_track_id -> signature
        for p in g.get("players", []):
            if p.get("track_id") in selected_ids and p.get("appearance"):
                sel_sigs[p["track_id"]] = p["appearance"]
        APPEARANCE_MATCH_MAX = 0.35   # max distance to count as "this player"

        from pipeline import AnalysisPipeline
        cfg = {
            "device":       os.environ.get("PB_DEVICE", "cpu"),
            "player_model": os.environ.get("PB_PLAYER_MODEL", "yolov8n.pt"),
            "ball_model":   os.environ.get("PB_BALL_MODEL", "basketball_rim_best.pt"),
            "fps":          30.0,
        }
        pipeline = AnalysisPipeline(cfg)

        import cv2
        cap = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Per-selected-player highlight reels.
        # A reel captures a window of frames around any "action moment" the
        # player is part of. An action moment is EITHER:
        #   (a) an event whose actor is this player, OR
        #   (b) this player's track being near the ball (active in the play).
        # This guarantees a selected player gets a meaningful reel even if the
        # event-actor attribution misses them.
        reels: Dict[int, cv2.VideoWriter] = {}
        reel_paths: Dict[int, str] = {}
        pre_buf   = int(fps * 3.0)   # lead-in frames kept in a rolling buffer
        post_hold = int(fps * 2.0)   # keep recording this many frames after a moment
        recent_frames: List[Any] = []
        record_until: Dict[int, int] = {pid: -1 for pid in selected_ids}
        NEAR_BALL_PX = 140.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        max_frames = int(os.environ.get("PB_MAX_ANALYZE_FRAMES", "0"))  # 0 = all

        def _ensure_reel(pid: int):
            if pid not in reels:
                rp = str(job_dir / f"reel_player_{pid}.mp4")
                reel_paths[pid] = rp
                reels[pid] = cv2.VideoWriter(rp, fourcc, fps, (width, height))
                for fbuf in recent_frames:      # flush lead-in
                    reels[pid].write(fbuf)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            res = pipeline.process_frame(frame, timestamp=frame_idx / fps)
            annotated = res["annotated_frame"]
            ball_pos  = res.get("ball_position")

            recent_frames.append(annotated)
            if len(recent_frames) > pre_buf:
                recent_frames.pop(0)

            # Match each active analysis-pass track to a selected player by
            # appearance. matched_sel[selected_id] = analysis_track object.
            matched_sel = {}
            if sel_sigs and pipeline.player_tracker:
                for t in pipeline.player_tracker.active_tracks:
                    x1, y1, x2, y2 = (int(v) for v in t.bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    if x2 - x1 < 8 or y2 - y1 < 16:
                        continue
                    tsig = _appearance_signature(frame[y1:y2, x1:x2])
                    if tsig is None:
                        continue
                    for sid, ssig in sel_sigs.items():
                        d = _appearance_distance(ssig, tsig)
                        if d <= APPEARANCE_MATCH_MAX:
                            # keep the closest track for this selected player
                            prev = matched_sel.get(sid)
                            if prev is None or d < prev[1]:
                                matched_sel[sid] = (t, d)

            # (a) an event fired near a matched player's position → their moment
            for ev in res["events"]:
                bp = ev.get("ball_position") or ball_pos
                if not bp:
                    continue
                for sid, (t, _d) in matched_sel.items():
                    cx, cy = t.center
                    if ((cx - bp[0]) ** 2 + (cy - bp[1]) ** 2) ** 0.5 <= NEAR_BALL_PX * 1.5:
                        record_until[sid] = frame_idx + post_hold

            # (b) matched player is near the ball → active in the play
            if ball_pos:
                for sid, (t, _d) in matched_sel.items():
                    cx, cy = t.center
                    if ((cx - ball_pos[0]) ** 2 + (cy - ball_pos[1]) ** 2) ** 0.5 <= NEAR_BALL_PX:
                        record_until[sid] = frame_idx + post_hold

            # Write to any reel whose window is currently open
            for pid, until in record_until.items():
                if frame_idx <= until:
                    _ensure_reel(pid)
                    reels[pid].write(annotated)

            if frame_idx % 60 == 0:
                games.update(game_id, progress=round(frame_idx / total, 3))

        cap.release()
        for w in reels.values():
            w.release()

        # ── Write stats ────────────────────────────────────────────────────
        summary = pipeline.get_stats()
        stats_path = job_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Per-selected-player slice of the stats
        selected_stats = {}
        for key, pl in summary.get("players", {}).items():
            selected_stats[key] = pl
        with open(job_dir / "player_stats.json", "w") as f:
            json.dump(selected_stats, f, indent=2, default=str)

        games.update(
            game_id,
            status="complete",
            progress=1.0,
            reels={str(pid): f"/api/games/{game_id}/players/{pid}/reel"
                   for pid in reel_paths},
            stats_url=f"/api/games/{game_id}/stats",
        )
        logger.info("Analysis complete: game=%s reels=%d", game_id, len(reel_paths))
    except Exception as e:
        logger.exception("Analysis failed for game %s", game_id)
        games.update(game_id, status="error", error=str(e))


# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _auth() -> Dict:
    """Resolve the current user from the X-API-Key header, or 401."""
    key = request.headers.get("X-API-Key", "")
    for uid, u in users.all().items():
        if u.get("api_key") == key:
            return {"id": uid, **u}
    abort(401, "Invalid or missing API key")


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "packages": PACKAGES})


# ── 1. Sign up / buy a package ─────────────────────────────────────────────────
@app.post("/api/signup")
def signup():
    body = request.get_json(silent=True) or {}
    email   = body.get("email", "").strip().lower()
    package = body.get("package", "starter")
    if not email:
        return jsonify({"error": "email required"}), 400
    if package not in PACKAGES:
        return jsonify({"error": f"unknown package; choose {list(PACKAGES)}"}), 400

    user_id = str(uuid.uuid4())
    api_key = "pb_" + secrets.token_urlsafe(24)
    credits = PACKAGES[package]["games"]
    users.put(user_id, {
        "email":    email,
        "package":  package,
        "credits":  credits,
        "api_key":  api_key,
        "created":  time.time(),
    })
    return jsonify({
        "user_id":  user_id,
        "api_key":  api_key,       # client stores this and sends as X-API-Key
        "package":  package,
        "credits":  credits,
        "message":  f"Signed up on '{package}' — {credits} game credits.",
    }), 201


@app.get("/api/account")
def account():
    u = _auth()
    return jsonify({
        "email":   u["email"],
        "package": u["package"],
        "credits": u["credits"],
    })


# ── 2. Upload a game video → scan players ───────────────────────────────────────
@app.post("/api/games")
def upload_game():
    u = _auth()
    if u["credits"] <= 0:
        return jsonify({"error": "no game credits remaining — buy a package"}), 402
    if "file" not in request.files:
        return jsonify({"error": "send a video as multipart 'file'"}), 400
    file = request.files["file"]
    if not file.filename or not _allowed(file.filename):
        return jsonify({"error": f"allowed types: {sorted(ALLOWED_EXT)}"}), 400

    game_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    video_path = str(UPLOAD_DIR / f"{game_id}.{ext}")
    file.save(video_path)

    games.put(game_id, {
        "user_id":  u["id"],
        "status":   "uploaded",
        "video":    video_path,
        "created":  time.time(),
        "players":  [],
    })

    # Kick off player scan in the background
    threading.Thread(target=_scan_players_job, args=(game_id, video_path),
                     daemon=True).start()

    return jsonify({
        "game_id": game_id,
        "status":  "scanning",
        "poll":    f"/api/games/{game_id}",
    }), 202


# ── 3. See detected players (for selection UI) ──────────────────────────────────
@app.get("/api/games/<game_id>/players")
def list_players(game_id):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    return jsonify({
        "status":  g["status"],
        "players": g.get("players", []),
    })


@app.get("/api/games/<game_id>/players/<int:pid>/thumb")
def player_thumb(game_id, pid):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    p = JOBS_DIR / game_id / f"player_{pid:03d}.jpg"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/jpeg")


# ── 4. Select player(s) & analyze (spends 1 credit) ─────────────────────────────
@app.post("/api/games/<game_id>/analyze")
def analyze_game(game_id):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    if g["status"] not in ("players_ready", "complete", "error"):
        return jsonify({"error": f"not ready to analyze (status={g['status']})"}), 409

    body = request.get_json(silent=True) or {}
    selected = body.get("player_ids", [])
    if not isinstance(selected, list) or not selected:
        return jsonify({"error": "provide player_ids: [track_id, ...]"}), 400
    selected = [int(x) for x in selected]

    # Spend one credit per game analyzed (not per player)
    if u["credits"] <= 0:
        return jsonify({"error": "no game credits remaining"}), 402
    users.update(u["id"], credits=u["credits"] - 1)

    games.update(game_id, status="queued", selected_players=selected,
                 progress=0.0)
    threading.Thread(target=_analyze_job,
                     args=(game_id, g["video"], selected),
                     daemon=True).start()

    return jsonify({
        "game_id": game_id,
        "status":  "queued",
        "selected_players": selected,
        "credits_remaining": u["credits"] - 1,
        "poll": f"/api/games/{game_id}",
    }), 202


# ── 5. Poll job status ──────────────────────────────────────────────────────────
@app.get("/api/games/<game_id>")
def game_status(game_id):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    return jsonify({
        "game_id":  game_id,
        "status":   g["status"],
        "progress": g.get("progress", 0.0),
        "players_found": g.get("players_found"),
        "selected_players": g.get("selected_players"),
        "reels":    g.get("reels"),
        "stats_url": g.get("stats_url"),
        "error":    g.get("error"),
    })


# ── 6. Download stats ───────────────────────────────────────────────────────────
@app.get("/api/games/<game_id>/stats")
def get_stats(game_id):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    p = JOBS_DIR / game_id / "stats.json"
    if not p.exists():
        return jsonify({"error": "stats not ready"}), 404
    with open(p) as f:
        return jsonify(json.load(f))


# ── 7. Download a player's highlight reel ───────────────────────────────────────
@app.get("/api/games/<game_id>/players/<int:pid>/reel")
def get_reel(game_id, pid):
    u = _auth()
    g = games.get(game_id)
    if not g or g["user_id"] != u["id"]:
        abort(404)
    p = JOBS_DIR / game_id / f"reel_player_{pid}.mp4"
    if not p.exists():
        return jsonify({"error": "reel not ready for this player"}), 404
    return send_file(str(p), mimetype="video/mp4", as_attachment=True,
                     download_name=f"highlight_player_{pid}.mp4")


if __name__ == "__main__":
    print("=" * 60)
    print("Purple Box Sports — Web Backend")
    print("=" * 60)
    print("Flow: signup → upload → select player → analyze → stats + reel")
    print("Packages:", ", ".join(f"{k} ({v['games']} games)" for k, v in PACKAGES.items()))
    print("Listening on http://0.0.0.0:8000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
