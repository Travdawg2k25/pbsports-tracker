"""Referee whistle detection from the game audio.

Fouls are the one stat vision barely sees: contact is brief, often occluded, and only
counts if an official calls it. The whistle is the call. It is a loud, nearly pure tone
in roughly the 2-4.5 kHz band that crowd noise and squeaking shoes do not sustain, so a
band-energy plus tonality test finds it cheaply. What happens next (free throws, an
inbound, the other team's ball) then decides what kind of whistle it was.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig
from .types import dump_json, load_json

log = logging.getLogger("hoopvision.audio")


@dataclass
class Whistle:
    start_s: float
    end_s: float
    confidence: float
    peak_hz: float


def load_audio(video: str | Path, sample_rate: int) -> np.ndarray | None:
    """Mono float32 samples via ffmpeg, or None when the file has no audio track."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            log.warning("no ffmpeg; whistle detection skipped")
            return None
    cmd = [ffmpeg, "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0 or not res.stdout:
        log.info("no audio track in %s", video)
        return None
    return np.frombuffer(res.stdout, dtype=np.float32)


def detect_whistles(samples: np.ndarray, sample_rate: int, cfg: AudioConfig) -> list[Whistle]:
    win = int(cfg.window_s * sample_rate)
    hop = max(1, win // 2)
    if samples is None or len(samples) < win:
        return []
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
    band = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
    n = 1 + (len(samples) - win) // hop
    flags = np.zeros(n, dtype=bool)
    scores = np.zeros(n)
    peaks = np.zeros(n)
    for i in range(n):
        seg = samples[i * hop : i * hop + win] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        total = spec.sum() + 1e-12
        band_spec = spec[band]
        ratio = band_spec.sum() / total
        # Tonality: how much of the band energy sits in its strongest few bins.
        top = np.sort(band_spec)[-cfg.tonal_bins :].sum() / (band_spec.sum() + 1e-12)
        loud = 10 * np.log10(np.mean(seg**2) + 1e-12)
        ok = ratio >= cfg.band_ratio and top >= cfg.tonality and loud >= cfg.min_db
        flags[i] = ok
        scores[i] = ratio * top
        peaks[i] = freqs[band][int(np.argmax(band_spec))] if band_spec.size else 0.0
    out: list[Whistle] = []
    i = 0
    hop_s = hop / sample_rate
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (flags[j + 1] or (j + 2 < n and flags[j + 2])):
            j += 1
        start, end = i * hop_s, j * hop_s + cfg.window_s
        if end - start >= cfg.min_duration_s:
            out.append(
                Whistle(
                    start_s=round(start, 2),
                    end_s=round(end, 2),
                    confidence=round(float(np.clip(scores[i : j + 1].mean() * 1.4, 0, 1)), 3),
                    peak_hz=round(float(np.median(peaks[i : j + 1])), 0),
                )
            )
        i = j + 1
    return out


def stage_audio(video: str | Path, out: Path, cfg: AudioConfig) -> list[Whistle]:
    samples = load_audio(video, cfg.sample_rate)
    whistles = detect_whistles(samples, cfg.sample_rate, cfg) if samples is not None else []
    dump_json(whistles, out / "whistles.json")
    log.info("audio: %d whistles", len(whistles))
    return whistles


def load_whistles(path: str | Path) -> list[Whistle]:
    if not Path(path).exists():
        return []
    return [Whistle(**w) for w in load_json(path)]
