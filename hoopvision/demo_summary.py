"""Print a clean, demo-ready summary of a HoopVision run.

Pulls the show-worthy facts out of a run directory into one readable block: what the
system did automatically (rim detection, players tracked, events, reel) plus the
box-score shape. Meant for walking a stakeholder through progress WITHOUT overselling
stat accuracy on hard footage.

    .venv/bin/python demo_summary.py runs/demo
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def load(p):
    return json.load(open(p))


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo")
    stats_path = run / "stats.json"
    inner = run / "run"

    print("=" * 62)
    print("  HoopVision — automatic single-camera basketball analysis")
    print("=" * 62)

    calib = load(run / "calibration.json") if (run / "calibration.json").exists() else {}
    rims = calib.get("rim_boxes", {})
    print("\n1. AUTOMATIC COURT SETUP (no manual calibration by the parent)")
    if rims:
        for side, b in rims.items():
            print(f"   - detected {side} rim at pixels {[round(v) for v in b]}")
    else:
        print("   - (no rim file found)")

    if (inner / "tracks.json").exists():
        tracks = load(inner / "tracks.json")
        ball = load(inner / "ball.json")
        print("\n2. AUTOMATIC TRACKING")
        print(f"   - players tracked across the clip: {len(tracks)}")
        print(f"   - frames the ball was detected: {len(ball.get('frames', []))}")

    if (inner / "events.json").exists():
        import collections

        evs = load(inner / "events.json")
        kinds = collections.Counter(e["kind"] for e in evs)
        print("\n3. EVENTS DETECTED (the raw material for the box score)")
        for k, n in kinds.most_common():
            print(f"   - {k}: {n}")

    if stats_path.exists():
        s = load(stats_path)
        print("\n4. BOX SCORE (AI draft — parent verifies/edits)")
        print(f"   - players in box score: {len(s['players'])}")
        print(f"   - team score (detected): {s.get('score')}")
        print(f"   - distance-reliable (2v3, FT geometry): {s.get('distance_reliable')}")
        sb = s.get("scoreboard")
        if sb and sb.get("available"):
            print(f"   - scoreboard truth: {sb['scoreboard']} | unaccounted: {sb['unaccounted']}")
        # show the stat CATEGORIES available per player (the product surface)
        if s["players"]:
            p = next(iter(s["players"].values()))
            cats = ["points", "fg_makes", "fg_attempts", "fg3_makes", "ft_makes",
                    "total_rebounds", "assists", "steals", "blocks", "turnovers"]
            print("   - stat categories per player: " + ", ".join(cats))
            print(f"   - best-effort flags (verify-me): {p.get('best_effort')}")

    reel = next(iter(run.glob("reel*.mp4")), None)
    print("\n5. HIGHLIGHT REEL")
    if reel and reel.stat().st_size > 0:
        codec = "?"
        try:
            import shutil

            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                codec = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(reel)],
                    capture_output=True, text=True,
                ).stdout.strip()
        except Exception:
            pass
        mb = reel.stat().st_size / 1e6
        print(f"   - {reel.name}: {mb:.1f} MB, codec={codec} (avc1/h264 = browser-playable)")
    else:
        print("   - (no reel produced)")

    print("\n" + "=" * 62)
    print("  Honest note: stat ACCURACY depends on footage. This clip is a")
    print("  wide broadcast angle (hardest case). Parent-filmed footage (close,")
    print("  steady, hoop in frame) is what makes points/FG% reliable.")
    print("=" * 62)


if __name__ == "__main__":
    main()
