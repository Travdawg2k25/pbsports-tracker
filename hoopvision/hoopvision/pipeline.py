"""Stage orchestration.

Stages write JSON into the run directory and are individually resumable: re-running with
``--from events`` reuses the existing tracks, which matters because detection on a full
game is hours of GPU time and the event heuristics need tuning against a hand-scored clip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

from .audio import load_whistles, stage_audio
from .boxscore import aggregate, load_roster
from .config import Config
from .court import Calibration
from .events import build_events
from .jersey import JerseyReader, resolve_identity
from .pose import estimate_poses, frames_near, load_poses, save_poses
from .review import apply_edits, ensure_ids, load_edits, review_queue
from .stabilize import Motion, displacement, estimate_motion
from .teams import assign_teams, separation, track_colour
from .types import (
    Event,
    Identity,
    Track,
    VideoMeta,
    dump_json,
    load_json,
    tracks_from_json,
)
from .video import FrameSeeker, iter_frames, probe

log = logging.getLogger("hoopvision")

STAGES = ["motion", "track", "teams", "jersey", "pose", "audio", "events", "boxscore"]

# Below this the camera is static enough that warping every frame is noise, not signal.
STATIC_CAMERA_PX = 8.0


def run(
    video: str | Path,
    out_dir: str | Path,
    calibration: str | Path | None = None,
    config: Config | None = None,
    roster: str | Path | None = None,
    start_stage: str = "motion",
    max_frames: int | None = None,
    stabilize: bool = True,
) -> dict:
    cfg = config or Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    dump_json(meta, out / "video.json")
    begin = STAGES.index(start_stage)

    def _timed(name: str, fn):
        t0 = time.time()
        fn()
        log.info("stage %s took %.1fs", name, time.time() - t0)

    if begin <= 0 and stabilize:
        _timed("motion", lambda: stage_motion(video, out, cfg, meta, max_frames))
    calib = load_calibration(calibration, out) if calibration else None

    if begin <= 1:
        _timed("track", lambda: stage_track(video, out, cfg, calib, meta, max_frames))
    if begin <= 2:
        _timed("teams", lambda: stage_teams(video, out, cfg))
    if begin <= 3:
        _timed("jersey", lambda: stage_jersey(video, out, cfg))
    if begin <= 4 and cfg.pose.enabled and calib is not None:
        _timed("pose", lambda: stage_pose(video, out, cfg, calib, meta))
    if begin <= 5 and cfg.audio.enabled:
        _timed("audio", lambda: stage_audio(video, out, cfg.audio))
    if begin <= 6:
        _timed("events", lambda: stage_events(out, cfg, calib, meta))
    return stage_boxscore(out, meta, roster, cfg.events.review_threshold)


def stage_motion(
    video: str | Path, out: Path, cfg: Config, meta: VideoMeta, max_frames: int | None
) -> Motion:
    end = None if max_frames is None else max_frames - 1
    # Motion is CPU feature-matching and dominates runtime; estimate it every
    # motion_stride frames (camera drift is smooth enough to interpolate between).
    motion = estimate_motion(video, stride=max(cfg.frame_stride, cfg.motion_stride), end=end)
    moved = displacement(motion, meta.width, meta.height)
    log.info("camera motion: %.0f px peak frame-centre displacement", moved)
    if moved < STATIC_CAMERA_PX:
        (out / "motion.json").unlink(missing_ok=True)
        log.info("camera is static; using the calibration frame directly")
    else:
        motion.save(out / "motion.json")
    return motion


def load_calibration(path: str | Path, out: Path) -> Calibration:
    """Calibration for the run, with camera motion attached when the camera moved."""
    calib = Calibration.load(path)
    motion_file = Path(out) / "motion.json"
    if motion_file.exists():
        calib.motion = Motion.load(motion_file)
    return calib


def stage_track(
    video: str | Path,
    out: Path,
    cfg: Config,
    calib: Calibration | None,
    meta: VideoMeta,
    max_frames: int | None,
) -> None:
    from .detect import Detector
    from .track import BallTracker, ByteTracker

    # A moving camera invalidates the fixed court box, so the ball is searched for
    # across the whole frame instead.
    search_box = calib.court_pixel_box() if calib and calib.motion is None else None
    detector = Detector(cfg.detection, search_box)
    players = ByteTracker(cfg.track, cls="player")
    ball = BallTracker()

    end = None if max_frames is None else max_frames - 1
    t0 = time.time()
    processed = 0
    for idx, frame in iter_frames(video, stride=cfg.frame_stride, end=end):
        dets = detector.detect(frame)
        players.update([d for d in dets if d.cls == "player"], idx)
        ball.update([d for d in dets if d.cls == "ball"], idx)
        processed += 1
        if processed % 50 == 0:
            rate = processed / (time.time() - t0)
            log.info("tracked %d frames (%.2f fps)", processed, rate)

    dump_json(players.tracks(), out / "tracks.json")
    dump_json(ball.track(), out / "ball.json")
    log.info("tracking done: %d player tracks, %d ball frames", len(players.tracks()),
             len(ball.track().frames))


def stage_teams(video: str | Path, out: Path, cfg: Config) -> None:
    tracks = tracks_from_json(load_json(out / "tracks.json"))
    colours = {}
    with FrameSeeker(video) as seeker:
        for t in tracks:
            colours[t.track_id] = track_colour(seeker, t, cfg.jersey)
    labels = assign_teams(colours)
    dump_json(
        {
            "teams": {str(k): v for k, v in labels.items()},
            "kit_separation": round(separation(colours), 2),
        },
        out / "teams.json",
    )


def stage_jersey(video: str | Path, out: Path, cfg: Config) -> None:
    tracks = tracks_from_json(load_json(out / "tracks.json"))
    teams = {int(k): v for k, v in load_json(out / "teams.json")["teams"].items()}
    reader = JerseyReader(cfg.jersey, gpu=cfg.detection.device != "cpu")
    identities: list[Identity] = []
    with FrameSeeker(video) as seeker:
        for t in tracks:
            votes = reader.read_track(seeker, t)
            identities.append(
                resolve_identity(t.track_id, votes, teams.get(t.track_id), cfg.jersey)
            )
    dump_json(identities, out / "identities.json")
    named = sum(1 for i in identities if i.jersey)
    log.info("jersey OCR: %d/%d tracks identified", named, len(identities))


def _event_inputs(out: Path):
    tracks = tracks_from_json(load_json(out / "tracks.json"))
    ball = tracks_from_json([load_json(out / "ball.json")])[0]
    identities = {
        i["track_id"]: (i["team"], i["jersey"]) for i in load_json(out / "identities.json")
    }
    return tracks, ball, identities


def stage_pose(
    video: str | Path, out: Path, cfg: Config, calib: Calibration, meta: VideoMeta
) -> None:
    """Keypoints for the players involved in candidate events (shots, blocks, changes)."""
    tracks, ball, identities = _event_inputs(out)
    rough = build_events(tracks, ball, calib, identities, meta.fps, cfg.events)
    centres = [e.frame for e in rough if e.kind != "possession"]
    involved = {e.track_id for e in rough if e.track_id is not None}
    for e in rough:
        involved.update(c["track_id"] for c in e.detail.get("candidates", []))
    wanted = frames_near(centres, int(cfg.pose.window_s * meta.fps), cfg.pose.stride)
    poses = estimate_poses(
        video, [t for t in tracks if t.track_id in involved] or tracks, cfg.pose,
        device=getattr(cfg, "device", "cpu"), frames_of_interest=wanted,
    )
    save_poses(poses, out / "pose.json")
    log.info("pose: %d tracks with keypoints over %d frames", len(poses), len(wanted))


def stage_events(out: Path, cfg: Config, calib: Calibration | None, meta: VideoMeta) -> None:
    if calib is None:
        raise ValueError("event detection needs a calibration file (hoopvision calibrate)")
    tracks, ball, identities = _event_inputs(out)
    poses = load_poses(out / "pose.json")
    whistles = load_whistles(out / "whistles.json")
    events = build_events(
        tracks, ball, calib, identities, meta.fps, cfg.events, whistles=whistles, poses=poses
    )
    if cfg.event_model and Path(cfg.event_model).exists():
        from .ml.event_model import EventRescorer

        rescorer = EventRescorer(cfg.event_model, device=getattr(cfg, "device", "cpu"))
        events = rescorer.rescore(events, ball, tracks, identities, calib, meta.fps, poses)
    dump_json(events, out / "events.json")
    log.info("events: %d (%d whistles, %d posed tracks)", len(events), len(whistles), len(poses))


def stage_boxscore(
    out: Path, meta: VideoMeta, roster: str | Path | None, review_threshold: float = 0.6
) -> dict:
    """Box score from the AI events with every reviewer edit applied on top."""
    ai_events = ensure_ids([Event(**e) for e in load_json(out / "events.json")])
    edits = load_edits(out)
    events = apply_edits(ai_events, edits, meta.fps)
    box = aggregate(events, meta.fps, load_roster(roster))
    box["video"] = asdict(meta) | {"duration_s": meta.duration_s}
    queue = review_queue(events, review_threshold)
    box["review"] = {
        "edits": len(edits),
        "pending": len(queue),
        "status": "verified" if not queue else ("in_review" if edits else "ai_draft"),
    }
    dump_json(box, out / "boxscore.json")
    return box


def load_tracks(out: Path) -> list[Track]:
    return tracks_from_json(load_json(Path(out) / "tracks.json"))
