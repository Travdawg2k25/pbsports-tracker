"""Quick test: process 500 frames and report events detected."""
import sys
import time
sys.path.insert(0, '.')

from pipeline import AnalysisPipeline
import cv2

VIDEO = "Game5.mp4"
MAX_FRAMES = 500

cfg = {
    "device": "cpu",
    "player_model": "yolov8n.pt",
    "ball_model": "basketball_rim_best.pt",
    "draw_trails": False,
    "draw_zones": False,
    "fps": 30.0,
    "display_width": 1280,
    "display_height": 720,
}

print("Loading pipeline...")
pipeline = AnalysisPipeline(cfg)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open {VIDEO}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO} | {fps:.1f} fps | {total} frames")
print(f"Processing first {MAX_FRAMES} frames...\n")

all_events = []
dead_count = 0
t_start = time.time()

for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    timestamp = i / fps
    result = pipeline.process_frame(frame, timestamp=timestamp)
    
    if result["is_dead_ball"]:
        dead_count += 1
    
    if result["events"]:
        for ev in result["events"]:
            all_events.append(ev)
            print(f"  [{i:4d}] EVENT: {ev.get('type', '?'):<20} player={ev.get('player', '')} label={ev.get('label', '')}")
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        print(f"  --- Frame {i+1}/{MAX_FRAMES} | {(i+1)/elapsed:.1f} fps | events={len(all_events)} | dead={dead_count} ---")

cap.release()
elapsed = time.time() - t_start

print(f"\n{'='*60}")
print(f"RESULTS ({MAX_FRAMES} frames in {elapsed:.1f}s @ {MAX_FRAMES/elapsed:.1f} fps):")
print(f"  Total events:     {len(all_events)}")
print(f"  Dead ball frames: {dead_count}/{MAX_FRAMES} ({100*dead_count/MAX_FRAMES:.1f}%)")
print(f"  Live frames:      {MAX_FRAMES - dead_count}")

if all_events:
    from collections import Counter
    counts = Counter(ev.get("type", "?") for ev in all_events)
    print(f"\n  Event breakdown:")
    for etype, count in counts.most_common():
        print(f"    {etype:<20}: {count}")
else:
    print("\n  ⚠ NO EVENTS DETECTED")
print(f"{'='*60}")
