"""Player and ball detection.

Two passes per frame:

* players — one inference on the whole frame downscaled to ``player_imgsz``;
* ball — overlapping native-resolution tiles restricted to the court area, because a
  basketball on a 4K sideline frame survives neither downscaling nor a low tile overlap.

Both passes use COCO classes from a stock YOLO checkpoint (``person``, ``sports ball``)
so the pipeline runs with no training. Point ``detection.model`` at a fine-tuned
basketball checkpoint to improve ball recall, which is the main accuracy lever.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .config import DetectionConfig
from .types import Box, Detection

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


class Detector:
    def __init__(self, cfg: DetectionConfig, court_mask_box: Box | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.player_model = YOLO(cfg.model)
        self.ball_model = (
            self.player_model if cfg.ball_model == cfg.model else YOLO(cfg.ball_model)
        )
        self.court_box = court_mask_box

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_players(frame) + self.detect_ball(frame)

    def detect_players(self, frame: np.ndarray) -> list[Detection]:
        res = self.player_model.predict(
            frame,
            imgsz=self.cfg.player_imgsz,
            conf=self.cfg.player_conf,
            classes=[COCO_PERSON],
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )[0]
        out = []
        for box, conf in zip(
            res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True
        ):
            out.append(Detection(box=tuple(box), conf=float(conf), cls="player"))
        return out

    def detect_ball(self, frame: np.ndarray) -> list[Detection]:
        if not self.cfg.ball_tiled:
            return self._detect_ball_single(frame)
        h, w = frame.shape[:2]
        region = self._search_region(w, h)
        best: Detection | None = None
        for tile, (ox, oy) in _tiles(frame, region, self.cfg.ball_tile, self.cfg.ball_tile_overlap):
            res = self.ball_model.predict(
                tile,
                imgsz=self.cfg.ball_tile,
                conf=self.cfg.ball_conf,
                classes=[self.cfg.ball_class],
                device=self.cfg.device,
                half=self.cfg.half,
                verbose=False,
            )[0]
            for box, conf in zip(
                res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True
            ):
                det = Detection(
                    box=(box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy),
                    conf=float(conf),
                    cls="ball",
                )
                if best is None or det.conf > best.conf:
                    best = det
        # There is exactly one game ball; keeping only the best candidate per frame keeps
        # the ball tracker from latching onto a ball rack on the sideline.
        return [best] if best else []

    def _detect_ball_single(self, frame: np.ndarray) -> list[Detection]:
        """One full-frame ball inference (no tiling) — the fast path for a ball-trained
        model. Keeps only the single highest-confidence ball, same as the tiled path."""
        res = self.ball_model.predict(
            frame,
            imgsz=self.cfg.ball_imgsz,
            conf=self.cfg.ball_conf,
            classes=[self.cfg.ball_class],
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )[0]
        best: Detection | None = None
        for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True):
            det = Detection(box=tuple(box), conf=float(conf), cls="ball")
            if best is None or det.conf > best.conf:
                best = det
        return [best] if best else []

    def _search_region(self, w: int, h: int) -> Box:
        if self.court_box is None:
            return (0.0, 0.0, float(w), float(h))
        m = self.cfg.ball_search_margin
        x1, y1, x2, y2 = self.court_box
        return (max(0.0, x1 - m), max(0.0, y1 - m), min(float(w), x2 + m), min(float(h), y2 + m))


def _tiles(
    frame: np.ndarray, region: Box, size: int, overlap: int
) -> Iterable[tuple[np.ndarray, tuple[int, int]]]:
    x1, y1, x2, y2 = (int(v) for v in region)
    step = max(1, size - overlap)
    for oy in range(y1, max(y1 + 1, y2 - overlap), step):
        for ox in range(x1, max(x1 + 1, x2 - overlap), step):
            tx2, ty2 = min(ox + size, x2), min(oy + size, y2)
            if tx2 - ox < 32 or ty2 - oy < 32:
                continue
            yield frame[oy:ty2, ox:tx2], (ox, oy)
