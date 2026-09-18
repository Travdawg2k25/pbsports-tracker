"""Jersey number OCR and track-level identity resolution.

Per-frame OCR of a jersey number is unreliable: the number is only legible when the
player faces the camera, is not occluded, and is not mid-stride. The pipeline therefore
never trusts a single read. It samples frames across a track, keeps reads above a
confidence floor, and resolves the track's number by weighted vote — so one track needs
only a handful of good frames out of hundreds.
"""

from __future__ import annotations

import re
from collections import defaultdict

import cv2
import numpy as np

from .config import JerseyConfig
from .teams import torso_crop
from .types import Identity, Track
from .video import FrameSeeker

DIGITS = "0123456789"
_VALID = re.compile(r"^\d{1,2}$")


class JerseyReader:
    def __init__(self, cfg: JerseyConfig, gpu: bool = False):
        import easyocr

        self.cfg = cfg
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, patch: np.ndarray) -> list[tuple[str, float]]:
        """OCR a torso crop, returning (number, confidence) candidates."""
        if patch.size == 0 or patch.shape[0] < self.cfg.min_crop_height // 2:
            return []
        prepared = _prepare(patch, self.cfg.upscale_to)
        raw = self.reader.readtext(prepared, allowlist=DIGITS, detail=1, paragraph=False)
        out = []
        for _, text, conf in raw:
            text = text.strip().lstrip("0") or "0"
            if _VALID.match(text) and conf >= self.cfg.ocr_conf:
                out.append((text, float(conf)))
        return out

    def read_track(self, seeker: FrameSeeker, track: Track) -> dict[str, float]:
        """Accumulate weighted votes for a track's jersey number."""
        votes: dict[str, float] = defaultdict(float)
        for tf in track.frames[:: self.cfg.sample_every]:
            if (tf.box[3] - tf.box[1]) < self.cfg.min_crop_height:
                continue
            img = seeker.get(tf.frame)
            if img is None:
                continue
            for number, conf in self.read(torso_crop(img, tf.box, self.cfg)):
                votes[number] += conf
        return dict(votes)


def _prepare(patch: np.ndarray, target_h: int) -> np.ndarray:
    """Upscale, grey, and contrast-normalise a torso crop for OCR."""
    h, w = patch.shape[:2]
    if h < target_h:
        scale = target_h / h
        patch = cv2.resize(patch, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)


def resolve_identity(
    track_id: int, votes: dict[str, float], team: str | None, cfg: JerseyConfig
) -> Identity:
    """Pick the winning number for a track, or leave it unidentified."""
    counts = {k: int(round(v)) for k, v in votes.items()}
    if not votes:
        return Identity(track_id, team, None, 0.0, counts)
    total = sum(votes.values())
    number, score = max(votes.items(), key=lambda kv: kv[1])
    share = score / total if total else 0.0
    enough = score >= cfg.min_votes * cfg.ocr_conf and share >= cfg.vote_margin
    return Identity(
        track_id=track_id,
        team=team,
        jersey=number if enough else None,
        jersey_confidence=round(share if enough else 0.0, 3),
        votes=counts,
    )


def merge_identities(identities: list[Identity]) -> dict[str, list[int]]:
    """Group tracks that resolved to the same team+number.

    Tracking breaks whenever players cross or leave the frame, so one real player is
    normally spread over many tracks; the jersey number is what stitches them back
    together for the box score.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for ident in identities:
        if ident.jersey is None:
            continue
        groups[f"{ident.team or 'unknown'}:{ident.jersey}"].append(ident.track_id)
    return dict(groups)
