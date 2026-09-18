"""Team assignment from kit colour.

Per track we take the median torso colour over a sample of frames in Lab space (more
stable than RGB under gym lighting), then split all tracks into two clusters with k-means.
Referees and bench players fall out as low-silhouette members and are left unassigned.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import JerseyConfig
from .types import Track
from .video import FrameSeeker, crop


def torso_crop(img: np.ndarray, box, cfg: JerseyConfig) -> np.ndarray:
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    ty1 = y1 + cfg.torso_top * h
    ty2 = y1 + cfg.torso_bottom * h
    tx1 = x1 + cfg.torso_inset * w
    tx2 = x2 - cfg.torso_inset * w
    return crop(img, (tx1, ty1, tx2, ty2))


def track_colour(
    seeker: FrameSeeker, track: Track, cfg: JerseyConfig, samples: int = 8
) -> np.ndarray | None:
    frames = track.frames
    if not frames:
        return None
    step = max(1, len(frames) // samples)
    vals = []
    for tf in frames[::step][:samples]:
        img = seeker.get(tf.frame)
        if img is None:
            continue
        patch = torso_crop(img, tf.box, cfg)
        if patch.size == 0 or patch.shape[0] < 8:
            continue
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
        # Median over the patch ignores the arms and the background showing between legs.
        vals.append(np.median(lab.reshape(-1, 3), axis=0))
    if not vals:
        return None
    return np.median(np.stack(vals), axis=0)


def assign_teams(
    colours: dict[int, np.ndarray], home_label: str = "home", away_label: str = "away"
) -> dict[int, str | None]:
    """Two-means over torso colours. Returns track_id -> team label."""
    ids = [tid for tid, c in colours.items() if c is not None]
    if len(ids) < 2:
        return {tid: None for tid in colours}
    data = np.stack([colours[tid] for tid in ids]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    # Label the brighter kit "home" so runs are reproducible rather than cluster-order
    # dependent; the roster file can rename the teams.
    bright = int(np.argmax(centres[:, 0]))
    names = {bright: home_label, 1 - bright: away_label}
    out: dict[int, str | None] = {tid: None for tid in colours}
    for tid, lab in zip(ids, labels, strict=True):
        out[tid] = names[int(lab)]
    return out


def separation(colours: dict[int, np.ndarray]) -> float:
    """Ratio of between-cluster to within-cluster spread; low values mean similar kits."""
    vals = [c for c in colours.values() if c is not None]
    if len(vals) < 4:
        return 0.0
    data = np.stack(vals).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    compactness, _, centres = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    between = float(np.linalg.norm(centres[0] - centres[1]))
    within = float(np.sqrt(compactness / len(data)))
    return between / max(within, 1e-6)
