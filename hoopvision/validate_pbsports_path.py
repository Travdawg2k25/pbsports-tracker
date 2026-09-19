"""End-to-end check of the pbsports-integration analyze path, no API/DB needed.

Runs the exact chain hoopvision_worker.do_analyze uses:
    auto-rim -> pipeline.run -> to_pbsports_stats -> H.264 reel

against a local video, and prints/saves the pbsports stats.json plus reel info so we can
confirm the shape and that a browser-playable clip is produced. Meant to run on the GPU
box inside the HoopVision venv:

    .venv/bin/python validate_pbsports_path.py game_ft.mp4 \
        --rim-model /opt/pbsports/basketball_rim_best.pt --device cuda:0 \
        --jersey 23 --name "Test Player" --out runs/pbtest
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from hoopvision.autorim import AutoRimConfig, detect_rims
from hoopvision.boxscore import player_clips
from hoopvision.config import Config
from hoopvision.pbsports_adapter import to_pbsports_stats
from hoopvision.pipeline import run as hv_run
from hoopvision.types import Event, load_json
from hoopvision.video import FrameSeeker, probe


def gather_clip_frames(video, clips, fps):
    wanted = sorted(
        (max(0, int(c["start_s"] * fps)), max(0, int(c["end_s"] * fps))) for c in clips
    )
    frames = []
    with FrameSeeker(video) as seeker:
        for s, e in wanted:
            for fidx in range(s, e + 1):
                img = seeker.get(fidx)
                if img is not None:
                    frames.append(img)
    return frames


def write_h264(frames, path, fps, size):
    w, h = size
    if not frames:
        return False, "no frames"
    tmp = Path(str(path) + ".mp4v.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        tmp.replace(path)
        return path.exists(), "mp4v only (no ffmpeg)"
    res = subprocess.run(
        [ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)],
        capture_output=True, text=True,
    )
    tmp.unlink(missing_ok=True)
    return (res.returncode == 0 and path.exists()), ("h264" if res.returncode == 0 else res.stderr[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rim-model", default="/opt/pbsports/basketball_rim_best.pt")
    ap.add_argument("--player-model", default="yolov8x.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--court-length-ft", type=float, default=84.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--jersey", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs/pbtest")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(args.video)
    print(f"video: {meta.width}x{meta.height} {meta.frame_count}f @ {meta.fps:.2f}fps")

    print("== auto-rim ==")
    calib = detect_rims(
        args.video, AutoRimConfig(model=args.rim_model, device=args.device),
        court_length_ft=args.court_length_ft,
    )
    if calib is None:
        print("RESULT: auto-rim found no stable rim (would need manual calibration)")
        return
    print(f"rim boxes: { {k: [round(v) for v in b] for k, b in calib.rim_boxes.items()} }")
    print(f"distance_reliable: {calib.distance_reliable}")
    calib_path = out / "calibration.json"
    calib.save(calib_path)

    print("== pipeline.run ==")
    cfg = Config()
    cfg.detection.device = args.device
    cfg.detection.half = args.device != "cpu"
    cfg.detection.model = args.player_model
    cfg.detection.ball_model = args.rim_model
    boxscore = hv_run(
        video=args.video, out_dir=str(out / "run"), calibration=str(calib_path),
        config=cfg, max_frames=args.max_frames,
    )

    focus_key = boxscore["players"][0]["player"] if boxscore["players"] else None
    print(f"focus player: {focus_key}")

    print("== to_pbsports_stats ==")
    stats = to_pbsports_stats(
        boxscore, game_id="validate-local", distance_reliable=calib.distance_reliable,
        focus_player_key=focus_key, jersey_number=args.jersey, player_name=args.name,
        duration_sec=meta.duration_s, total_frames=meta.frame_count,
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str)[:1500])

    print("== H.264 reel ==")
    if focus_key:
        events = [Event(**e) for e in load_json(out / "run" / "events.json")]
        clips = player_clips(events, focus_key, meta.fps, cfg.events)
        print(f"clip windows: {len(clips)}")
        frames = gather_clip_frames(args.video, clips, meta.fps)
        reel = out / "reel_focus.mp4"
        ok, how = write_h264(frames, reel, meta.fps, (meta.width, meta.height))
        size = reel.stat().st_size if reel.exists() else 0
        print(f"reel written={ok} codec={how} bytes={size}")
        # Confirm the codec is really H.264/avc1
        if ok and shutil.which("ffprobe"):
            pr = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(reel)],
                capture_output=True, text=True,
            )
            print(f"ffprobe codec: {pr.stdout.strip()}")

    print("\nDONE — stats + reel written to", out)


if __name__ == "__main__":
    main()
