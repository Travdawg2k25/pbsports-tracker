# =============================================================================
# team_classifier.py  —  Purple Box Sports
# Jersey-color-based team assignment for player tracks
# =============================================================================
#
# Assigns each player Track a stable team_id (0 or 1) by clustering the
# dominant torso color of every player into two groups.
#
# Design goals:
#   - No manual configuration. Learns the two team colors on the fly.
#   - Stable per-track: once a track_id is confidently assigned a team,
#     it stays (majority vote across all frames it's seen), so identity
#     doesn't flicker frame to frame.
#   - Cheap: samples a torso crop, reduces to a single HSV feature vector,
#     runs k-means (k=2) once enough samples are collected.
#
# Usage (per frame, after tracker.update):
#     team_clf.assign(frame, tracks)
# Each Track then has track.team_id set to 0 or 1 (or None until confident).
# =============================================================================

from __future__ import annotations

import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("PurpleBox.TeamClassifier")


class TeamClassifier:
    """
    Two-team classifier based on dominant jersey color.

    Call `assign(frame, tracks)` each frame. It:
      1. Extracts a torso-color feature for each player track.
      2. Accumulates features per track_id.
      3. Once `min_samples_to_fit` total samples exist, fits a 2-means
         model on the mean per-track colors to learn the two team colors.
      4. Assigns each track a team via majority vote of its own samples,
         which keeps the label stable across the whole clip.
    """

    def __init__(
        self,
        min_samples_to_fit: int = 40,
        refit_every:        int = 120,
        torso_top_frac:     float = 0.15,
        torso_bot_frac:     float = 0.45,
        torso_side_frac:    float = 0.20,
        sample_stride:      int = 3,
    ):
        self.min_samples_to_fit = min_samples_to_fit
        self.refit_every        = refit_every
        self.torso_top_frac     = torso_top_frac
        self.torso_bot_frac     = torso_bot_frac
        self.torso_side_frac    = torso_side_frac
        self.sample_stride      = sample_stride

        # Per-track accumulated color features (H, S, V)
        self._track_samples: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        # Per-track running team vote
        self._track_votes:   Dict[int, Counter] = defaultdict(Counter)
        # Final assigned team per track_id
        self._track_team:    Dict[int, int] = {}

        # Learned team color centroids (in feature space)
        self._centroids: Optional[np.ndarray] = None
        self._total_samples = 0
        self._frame_count   = 0
        self._fitted        = False

    # ── Public API ────────────────────────────────────────────────────────────

    def assign(self, frame: np.ndarray, tracks: List) -> None:
        """
        Extract colors, (re)fit if needed, and write team_id onto each track.
        """
        self._frame_count += 1

        # 1. Sample torso color for each player track (throttled per track)
        for t in tracks:
            if getattr(t, "class_name", "") not in ("player", "person"):
                continue
            if self._frame_count % self.sample_stride != 0 and t.track_id in self._track_samples:
                # Already have samples; only re-sample every few frames
                pass
            feat = self._extract_feature(frame, t.bbox)
            if feat is not None:
                self._track_samples[t.track_id].append(feat)
                self._total_samples += 1

        # 2. Fit / refit the 2-means model
        if (not self._fitted and self._total_samples >= self.min_samples_to_fit) \
                or (self._fitted and self._frame_count % self.refit_every == 0):
            self._fit()

        # 3. Assign team_id to every current track
        if self._fitted:
            for t in tracks:
                if getattr(t, "class_name", "") not in ("player", "person"):
                    continue
                team = self._classify_track(t.track_id)
                if team is not None:
                    t.team_id = team

    def get_team(self, track_id: int) -> Optional[int]:
        return self._track_team.get(track_id)

    def reset(self) -> None:
        self._track_samples.clear()
        self._track_votes.clear()
        self._track_team.clear()
        self._centroids     = None
        self._total_samples = 0
        self._frame_count   = 0
        self._fitted        = False

    # ── Internals ──────────────────────────────────────────────────────────────

    def _extract_feature(
        self,
        frame: np.ndarray,
        bbox:  Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Return the median (H, S, V) of the player's torso region, or None.
        The torso band avoids the head, arms, and legs so the jersey color
        dominates.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame))

        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 24:
            return None

        ty1 = y1 + int(bh * self.torso_top_frac)
        ty2 = y1 + int(bh * self.torso_bot_frac)
        tx1 = x1 + int(bw * self.torso_side_frac)
        tx2 = x2 - int(bw * self.torso_side_frac)

        if ty2 <= ty1 or tx2 <= tx1:
            return None

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Ignore very dark / very bright pixels (shadows, glare) which corrupt hue
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)
        h = hsv[:, :, 0].reshape(-1)
        mask = (v > 30) & (v < 245)
        if np.count_nonzero(mask) < 10:
            mask = np.ones_like(v, dtype=bool)

        # Hue is circular — encode as (cos, sin) to make clustering meaningful
        hf = h[mask].astype(np.float32) * (2.0 * np.pi / 180.0)
        return (
            float(np.median(np.cos(hf)) * 128 + 128),  # hue-cos  → 0..256
            float(np.median(s[mask])),                 # saturation
            float(np.median(v[mask])),                 # value/brightness
        )

    def _track_mean_feature(self, track_id: int) -> Optional[np.ndarray]:
        samples = self._track_samples.get(track_id)
        if not samples:
            return None
        return np.mean(np.array(samples, dtype=np.float32), axis=0)

    def _fit(self) -> None:
        """Fit 2-means on the per-track mean colors."""
        track_ids = [tid for tid, s in self._track_samples.items() if len(s) >= 2]
        if len(track_ids) < 2:
            return

        feats = np.array(
            [self._track_mean_feature(tid) for tid in track_ids],
            dtype=np.float32,
        )

        # k-means (k=2) via OpenCV
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                feats, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
        except cv2.error as e:
            logger.debug("kmeans failed: %s", e)
            return

        self._centroids = centers
        labels = labels.reshape(-1)

        # Record a vote for each track based on this fit
        for tid, lab in zip(track_ids, labels):
            self._track_votes[tid][int(lab)] += 1

        self._fitted = True

    def _classify_track(self, track_id: int) -> Optional[int]:
        """
        Return the majority-vote team for a track, refreshing the vote from
        the current centroids using the track's mean feature.
        """
        if self._centroids is None:
            return self._track_team.get(track_id)

        mean_feat = self._track_mean_feature(track_id)
        if mean_feat is not None:
            d0 = np.linalg.norm(mean_feat - self._centroids[0])
            d1 = np.linalg.norm(mean_feat - self._centroids[1])
            self._track_votes[track_id][0 if d0 <= d1 else 1] += 1

        votes = self._track_votes.get(track_id)
        if not votes:
            return self._track_team.get(track_id)

        team = votes.most_common(1)[0][0]
        self._track_team[track_id] = team
        return team
