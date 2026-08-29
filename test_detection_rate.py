"""Quick diagnostic: how often does the ball model detect the basketball?"""
import cv2
import sys
sys.path.insert(0, '.')
from pipeline import YOLODetector

VIDEO = "Game5.mp4"
MAX_FRAMES = 300

ball_detector = YOLODetector(
    model_path="basketball_rim_best.pt",
    device="cpu",
    conf=0.35,
    target_classes=["basketball", "ball", "rim"],
)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open {VIDEO}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO} | {fps:.1f} fps | {total} frames")
print(f"Testing first {MAX_FRAMES} frames...\n")

ball_detected = 0
rim_detected = 0
consecutive_miss = 0
max_consecutive_miss = 0

for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    dets = ball_detector.detect(frame)
    
    has_ball = any(d.class_name in ("basketball", "ball") for d in dets)
    has_rim = any(d.class_name == "rim" for d in dets)
    
    if has_ball:
        ball_detected += 1
        if consecutive_miss > max_consecutive_miss:
            max_consecutive_miss = consecutive_miss
        consecutive_miss = 0
    else:
        consecutive_miss += 1
    
    if has_rim:
        rim_detected += 1
    
    if (i + 1) % 50 == 0:
        print(f"  Frame {i+1}: ball={ball_detected}/{i+1} ({100*ball_detected/(i+1):.0f}%) rim={rim_detected}/{i+1} ({100*rim_detected/(i+1):.0f}%)")

cap.release()

if consecutive_miss > max_consecutive_miss:
    max_consecutive_miss = consecutive_miss

print(f"\n{'='*50}")
print(f"RESULTS ({MAX_FRAMES} frames):")
print(f"  Ball detected: {ball_detected}/{MAX_FRAMES} ({100*ball_detected/MAX_FRAMES:.1f}%)")
print(f"  Rim detected:  {rim_detected}/{MAX_FRAMES} ({100*rim_detected/MAX_FRAMES:.1f}%)")
print(f"  Max consecutive ball miss: {max_consecutive_miss} frames ({max_consecutive_miss/fps:.1f}s)")
print(f"\n  With threshold=15:  dead ball would trigger after {15/fps:.1f}s miss")
print(f"  With threshold=90:  dead ball would trigger after {90/fps:.1f}s miss")
print(f"  With threshold=180: dead ball would trigger after {180/fps:.1f}s miss")
