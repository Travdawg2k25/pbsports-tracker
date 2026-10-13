"""Player pose: keypoints per track, and the shooting / contest signals derived from them.

Boxes say where a player is; keypoints say what they are doing. A jump shot has both
wrists above the head with the elbows up, a contest has an arm raised toward the ball,
and neither looks like a dribble or a pass. Those two signals are what lets the event
layer tell a shot from a ball that merely rose near the rim, and a block from a
defender who was only standing in the lane.

Keypoints follow the 17-point COCO layout the Ultralytics pose models emit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PoseConfig
from .track import iou_matrix
from .types import Track, dump_json, load_json

log = logging.getLogger("hoopvision.pose")

NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class PoseTrack:
    """Keypoints for one player track: frame -> 17 x (x, y, conf)."""

    track_id: int
    frames: dict[int, list[list[float]]] = field(default_factory=dict)

    def nearest(self, frame: int, max_gap: int) -> np.ndarray | None:
        if not self.frames:
            return None
        if frame in self.frames:
            return np.asarray(self.frames[frame], dtype=float)
        best = min(self.frames, key=lambda f: abs(f - frame))
        if abs(best - frame) > max_gap:
            return None
        return np.asarray(self.frames[best], dtype=float)


def _visible(kp: np.ndarray, idx: int, min_conf: float) -> bool:
    return kp.shape[0] > idx and kp[idx, 2] >= min_conf


def shooting_score(kp: np.ndarray | None, min_conf: float = 0.3) -> float:
    """0..1: how much the pose looks like a shot release (hands up, elbows up).

    Image y grows downward, so "above" is a smaller y. Scaled by torso length so it does
    not depend on how far the player is from the camera.
    """
    if kp is None:
        return 0.0
    need = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if not all(_visible(kp, i, min_conf) for i in need):
        return 0.0
    shoulder_y = (kp[L_SHOULDER, 1] + kp[R_SHOULDER, 1]) / 2
    hip_y = (kp[L_HIP, 1] + kp[R_HIP, 1]) / 2
    torso = max(hip_y - shoulder_y, 1e-3)
    head_y = kp[NOSE, 1] if _visible(kp, NOSE, min_conf) else shoulder_y - 0.35 * torso

    wrists = [kp[i, 1] for i in (L_WRIST, R_WRIST) if _visible(kp, i, min_conf)]
    elbows = [kp[i, 1] for i in (L_ELBOW, R_ELBOW) if _visible(kp, i, min_conf)]
    if not wrists:
        return 0.0
    # Highest wrist above the head, in torso lengths: 0 at the head, 1 at half a torso above.
    wrist_lift = np.clip((head_y - min(wrists)) / (0.5 * torso), 0.0, 1.0)
    elbow_lift = (
        np.clip((shoulder_y - min(elbows)) / (0.3 * torso), 0.0, 1.0) if elbows else 0.0
    )
    both_hands = 1.0 if len(wrists) == 2 and max(wrists) < shoulder_y else 0.0
    return float(round(0.55 * wrist_lift + 0.3 * elbow_lift + 0.15 * both_hands, 3))


def contest_score(kp: np.ndarray | None, min_conf: float = 0.3) -> float:
    """0..1: at least one arm raised above the shoulder — a defender contesting or blocking."""
    if kp is None or not all(_visible(kp, i, min_conf) for i in (L_SHOULDER, R_SHOULDER)):
        return 0.0
    shoulder_y = min(kp[L_SHOULDER, 1], kp[R_SHOULDER, 1])
    width = abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0]) or 1.0
    wrists = [kp[i, 1] for i in (L_WRIST, R_WRIST) if _visible(kp, i, min_conf)]
    if not wrists:
        return 0.0
    return float(round(np.clip((shoulder_y - min(wrists)) / (1.2 * width), 0.0, 1.0), 3))


def match_poses(
    pose_boxes: np.ndarray, track_boxes: dict[int, tuple], min_iou: float
) -> dict[int, int]:
    """Greedy IoU match of pose detections to the tracks present on a frame."""
    if len(pose_boxes) == 0 or not track_boxes:
        return {}
    ids = list(track_boxes)
    ious = iou_matrix(np.array([track_boxes[i] for i in ids], dtype=float), pose_boxes)
    out: dict[int, int] = {}
    used: set[int] = set()
    for flat in np.argsort(-ious, axis=None):
        r, c = np.unravel_index(flat, ious.shape)
        if ious[r, c] < min_iou:
            break
        if ids[r] in out or c in used:
            continue
        out[ids[r]] = int(c)
        used.add(int(c))
    return out


def estimate_poses(
    video: str | Path,
    tracks: list[Track],
    cfg: PoseConfig,
    device: str = "cpu",
    half: bool = False,
    frames_of_interest: set[int] | None = None,
) -> dict[int, PoseTrack]:
    """Run the pose model every ``cfg.stride`` frames and attach keypoints to tracks.

    ``frames_of_interest`` restricts inference to frames near candidate events, which is
    where pose matters and keeps a full game affordable.
    """
    from ultralytics import YOLO

    from .video import iter_frames

    model = YOLO(cfg.model)
    by_frame: dict[int, dict[int, tuple]] = {}
    for t in tracks:
        for tf in t.frames:
            by_frame.setdefault(tf.frame, {})[t.track_id] = tf.box
    out: dict[int, PoseTrack] = {t.track_id: PoseTrack(t.track_id) for t in tracks}
    done = 0
    for idx, frame in iter_frames(video, stride=cfg.stride):
        if frames_of_interest is not None and idx not in frames_of_interest:
            continue
        present = by_frame.get(idx)
        if not present:
            continue
        res = model.predict(
            frame, imgsz=cfg.imgsz, conf=cfg.conf, device=device, half=half, verbose=False
        )[0]
        if res.keypoints is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()  # n x 17 x 3
        for tid, j in match_poses(boxes, present, cfg.match_iou).items():
            out[tid].frames[idx] = np.round(kps[j], 1).tolist()
        done += 1
        if done % 200 == 0:
            log.info("pose: %d frames", done)
    return {k: v for k, v in out.items() if v.frames}


def frames_near(centres: list[int], radius: int, stride: int) -> set[int]:
    """Frames on the pose stride within ``radius`` of any centre frame."""
    out: set[int] = set()
    for c in centres:
        for f in range(max(0, c - radius), c + radius + 1):
            if f % stride == 0:
                out.add(f)
    return out


def save_poses(poses: dict[int, PoseTrack], path: str | Path) -> None:
    dump_json({str(k): {str(f): kp for f, kp in v.frames.items()} for k, v in poses.items()}, path)


def load_poses(path: str | Path) -> dict[int, PoseTrack]:
    if not Path(path).exists():
        return {}
    raw = load_json(path)
    return {
        int(k): PoseTrack(int(k), {int(f): kp for f, kp in v.items()}) for k, v in raw.items()
    }
