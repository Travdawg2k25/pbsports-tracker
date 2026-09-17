"""Frame access helpers.

A 4K game file is far too large to hold in memory, so every stage streams frames and the
stages that only need a few frames (jersey crops, clip thumbnails) seek instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .types import VideoMeta


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    meta = VideoMeta(
        path=str(path),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(
    path: str | Path, stride: int = 1, start: int = 0, end: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (end is not None and idx > end):
                break
            if (idx - start) % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


class FrameSeeker:
    """Random access to frames, reusing one capture handle."""

    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self._pos = -1
        self._last: np.ndarray | None = None

    def get(self, frame: int) -> np.ndarray | None:
        # Callers commonly ask for the same frame again (several tracks share it) and
        # then walk forward; both are far cheaper than seeking on a long GOP.
        if frame == self._pos and self._last is not None:
            return self._last
        if 0 < frame - self._pos <= 8:
            while self._pos < frame:
                ok, img = self.cap.read()
                self._pos += 1
                if not ok:
                    self._last = None
                    return None
            self._last = img
            return img
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self.cap.read()
        self._pos = frame if ok else -1
        self._last = img if ok else None
        return self._last

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> FrameSeeker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def crop(img: np.ndarray, box: tuple[float, float, float, float], pad: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(w, int(box[2]) + pad)
    y2 = min(h, int(box[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]
