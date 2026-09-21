"""Automatic rim detection — calibration without a human.

The parent never marks court landmarks. Instead a fine-tuned rim detector
(``basketball_rim_best.pt`` in production) is run over a sample of frames; the stable
rim box on each side of the frame becomes the calibration's ``rim_boxes``.

This yields a *rim-only* :class:`~hoopvision.court.Calibration`: it has no image->court
homography, so :meth:`Calibration.has_homography` is ``False`` and
:attr:`Calibration.distance_reliable` is ``False``. Makes, points and FG% are computed
purely from the rim box and are reliable; court-space distances (three-pointers, shot
distance) fall back to an approximate pixel scale and are flagged best-effort for a human
to correct.

If the detector never finds a confident rim (bad angle, occlusion, no rim model), this
returns ``None`` and the caller decides whether to fall back to manual calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .court import Calibration
from .types import Box
from .video import iter_frames, probe

log = logging.getLogger("hoopvision.autorim")

# The rim model labels its two classes; a fine-tuned basketball model emits "rim".
RIM_CLASS_NAMES = ("rim", "hoop", "basket")


@dataclass
class AutoRimConfig:
    model: str = "basketball_rim_best.pt"
    device: str = "cpu"
    conf: float = 0.35
    # Sample this many frames spread across the clip; a rim is static, so a sparse
    # sample over the whole video is more robust than a dense burst at the start.
    sample_frames: int = 120
    # A side needs at least this many agreeing detections to be trusted.
    min_detections_per_side: int = 8
    # Long edge the frame is resized to for inference.
    imgsz: int = 1280


def _rim_detections(model, frame, cfg: AutoRimConfig) -> list[tuple[Box, float]]:
    """Rim boxes + confidence from one frame, or [] if the model finds none."""
    res = model.predict(
        frame, imgsz=cfg.imgsz, conf=cfg.conf, device=cfg.device, verbose=False
    )[0]
    names = res.names  # class_id -> name
    out: list[tuple[Box, float]] = []
    boxes = res.boxes
    if boxes is None:
        return out
    for xyxy, conf, cls in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
    ):
        name = str(names.get(int(cls), "")).lower() if isinstance(names, dict) else ""
        if name in RIM_CLASS_NAMES:
            out.append((tuple(xyxy), float(conf)))
    return out


def _median_box(boxes: list[Box]) -> Box:
    arr = np.array(boxes, dtype=np.float32)
    m = np.median(arr, axis=0)
    return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))


def detect_rims(
    video: str,
    cfg: AutoRimConfig | None = None,
    court_length_ft: float = 94.0,
) -> Calibration | None:
    """Scan the video and return a rim-only Calibration, or None if no stable rim.

    Rims are clustered into left/right by their centre x against the frame midline and
    each side's box is the median of its detections, which rejects the occasional false
    positive without needing a tracker.
    """
    cfg = cfg or AutoRimConfig()
    meta = probe(video)
    mid_x = meta.width / 2.0

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - import guard
        log.warning("ultralytics not available for auto-rim: %s", exc)
        return None

    try:
        model = YOLO(cfg.model)
    except Exception as exc:  # noqa: BLE001 - model file may be missing
        log.warning("could not load rim model %s: %s", cfg.model, exc)
        return None

    # Even stride across the whole clip.
    total = meta.frame_count or (cfg.sample_frames * 10)
    stride = max(1, total // cfg.sample_frames)

    left: list[Box] = []
    right: list[Box] = []
    scanned = 0
    for idx, frame in iter_frames(video, stride=stride):
        scanned += 1
        for box, _conf in _rim_detections(model, frame, cfg):
            cx = (box[0] + box[2]) / 2.0
            (left if cx < mid_x else right).append(box)
        if scanned >= cfg.sample_frames:
            break

    rim_boxes: dict[str, Box] = {}
    if len(left) >= cfg.min_detections_per_side:
        rim_boxes["left"] = _median_box(left)
    if len(right) >= cfg.min_detections_per_side:
        rim_boxes["right"] = _median_box(right)

    if not rim_boxes:
        log.warning(
            "auto-rim found no stable rim (left=%d right=%d over %d frames)",
            len(left), len(right), scanned,
        )
        return None

    log.info(
        "auto-rim: %s (left=%d right=%d detections over %d frames)",
        {k: [round(v) for v in b] for k, b in rim_boxes.items()},
        len(left), len(right), scanned,
    )
    return Calibration(rim_boxes=rim_boxes, court_length_ft=court_length_ft)
