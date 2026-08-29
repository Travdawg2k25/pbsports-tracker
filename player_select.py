# player_select.py — Purple Box Sports | Player Selection Module
# ================================================================
# Processes the first N seconds of video to detect and crop players,
# returning thumbnails for user selection.
# ================================================================

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("PurpleBox.PlayerSelect")

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from tracker import MultiObjectTracker, Detection
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# PLAYER DETECTION RESULT
# ═════════════════════════════════════════════════════════════════════════════
class DetectedPlayer:
    """Represents a player detected in the initial scan."""

    def __init__(
        self,
        track_id:   int,
        bbox:       Tuple[int, int, int, int],
        crop:       np.ndarray,
        frame_idx:  int,
        confidence: float,
        appearances: int = 1,
    ):
        self.track_id    = track_id
        self.bbox        = bbox         # (x1, y1, x2, y2)
        self.crop        = crop         # Best crop image
        self.frame_idx   = frame_idx    # Frame where best crop was captured
        self.confidence  = confidence
        self.appearances = appearances  # How many frames this player appeared in
        self._crops: List[np.ndarray] = [crop]

    def update(self, bbox: Tuple, crop: np.ndarray, confidence: float) -> None:
        """Update with a better crop if this one is larger/clearer."""
        self.appearances += 1
        # Keep the crop with the largest area (more pixels = clearer)
        curr_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        new_area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if new_area > curr_area and confidence > 0.4:
            self.bbox       = bbox
            self.crop       = crop
            self.confidence = confidence

    def to_thumbnail(self, size: Tuple[int, int] = (150, 200)) -> np.ndarray:
        """Resize crop to standard thumbnail size."""
        if self.crop is None or self.crop.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return cv2.resize(self.crop, size)

    def crop_base64(self, size: Tuple[int, int] = (150, 200)) -> str:
        """Return thumbnail as base64-encoded JPEG."""
        thumb = self.to_thumbnail(size)
        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id":    self.track_id,
            "bbox":        list(self.bbox),
            "confidence":  round(self.confidence, 3),
            "appearances": self.appearances,
            "thumbnail":   self.crop_base64(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# PLAYER SCANNER
# ═════════════════════════════════════════════════════════════════════════════
class PlayerScanner:
    """
    Scans the first N seconds of a video to detect and identify unique players.

    Returns a list of DetectedPlayer objects with crops suitable for
    user selection in the mobile app or web portal.
    """

    def __init__(
        self,
        model_path:     str   = "yolov8n.pt",
        device:         str   = "cpu",
        scan_seconds:   float = 15.0,
        conf_threshold: float = 0.40,
        min_appearances: int  = 5,
        min_height_px:  int   = 80,
    ):
        self.model_path      = model_path
        self.device          = device
        self.scan_seconds    = scan_seconds
        self.conf_threshold  = conf_threshold
        self.min_appearances = min_appearances
        self.min_height_px   = min_height_px

        self._model: Optional[Any] = None
        self._tracker: Optional[MultiObjectTracker] = None
        self._load()

    def _load(self) -> None:
        if not YOLO_AVAILABLE:
            logger.error("ultralytics not installed")
            return
        try:
            self._model = YOLO(self.model_path)
            logger.info("PlayerScanner: loaded %s on %s", self.model_path, self.device)
        except Exception as e:
            logger.error("Failed to load YOLO: %s", e)

        if TRACKER_AVAILABLE:
            self._tracker = MultiObjectTracker(max_age=30, max_distance=120)

    def scan(self, video_path: str) -> List[DetectedPlayer]:
        """
        Scan the first N seconds of video and return detected players.

        Returns list of DetectedPlayer sorted by appearances (most stable first).
        """
        if self._model is None:
            raise RuntimeError("YOLO model not loaded")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames  = int(self.scan_seconds * fps)
        frame_skip  = max(1, int(fps / 10))  # Process ~10 fps for speed

        logger.info(
            "Scanning %s | %.1f fps | scanning first %d frames (%.1fs)",
            video_path, fps, min(max_frames, total), self.scan_seconds,
        )

        players: Dict[int, DetectedPlayer] = {}
        frame_idx = 0

        try:
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames for speed
                if frame_idx % frame_skip != 0:
                    continue

                # Detect persons
                detections = self._detect_persons(frame, frame_idx)

                # Update tracker and get stable IDs
                if self._tracker and detections:
                    tracker_dets = [
                        Detection(
                            bbox=d["bbox"],
                            conf=d["conf"],
                            class_id=0,
                            class_name="person",
                            frame_id=frame_idx,
                        )
                        for d in detections
                    ]
                    tracked = self._tracker.update(tracker_dets)

                    for track in tracked:
                        tid = track.track_id
                        x1, y1, x2, y2 = track.bbox
                        h = y2 - y1

                        if h < self.min_height_px:
                            continue

                        # Crop player from frame
                        ix1 = max(0, int(x1))
                        iy1 = max(0, int(y1))
                        ix2 = min(frame.shape[1], int(x2))
                        iy2 = min(frame.shape[0], int(y2))
                        crop = frame[iy1:iy2, ix1:ix2].copy()

                        if crop.size == 0:
                            continue

                        bbox = (ix1, iy1, ix2, iy2)

                        if tid in players:
                            players[tid].update(bbox, crop, track.confidence if hasattr(track, 'confidence') else 0.5)
                        else:
                            players[tid] = DetectedPlayer(
                                track_id   = tid,
                                bbox       = bbox,
                                crop       = crop,
                                frame_idx  = frame_idx,
                                confidence = track.confidence if hasattr(track, 'confidence') else 0.5,
                            )

                elif detections:
                    # No tracker — use raw detections with index as ID
                    for i, d in enumerate(detections):
                        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
                        h = y2 - y1
                        if h < self.min_height_px:
                            continue
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size == 0:
                            continue
                        # Use a simple spatial hash as pseudo-ID
                        pseudo_id = i
                        if pseudo_id not in players:
                            players[pseudo_id] = DetectedPlayer(
                                track_id=pseudo_id,
                                bbox=(x1, y1, x2, y2),
                                crop=crop,
                                frame_idx=frame_idx,
                                confidence=d["conf"],
                            )

        finally:
            cap.release()

        # Filter by minimum appearances
        stable_players = [
            p for p in players.values()
            if p.appearances >= self.min_appearances
        ]

        # Sort by appearances (most stable players first)
        stable_players.sort(key=lambda p: p.appearances, reverse=True)

        logger.info(
            "Scan complete: %d total tracks, %d stable players (>=%d appearances)",
            len(players), len(stable_players), self.min_appearances,
        )

        return stable_players

    def scan_and_export(
        self,
        video_path: str,
        output_dir: str = "output/player_select",
    ) -> Dict[str, Any]:
        """
        Scan video and export results as JSON + thumbnail images.

        Returns a dict with player data suitable for API response.
        """
        players = self.scan(video_path)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        result = {
            "video":    video_path,
            "scanned_seconds": self.scan_seconds,
            "players_found":   len(players),
            "players": [],
        }

        for player in players:
            # Save thumbnail image
            thumb_path = out_path / f"player_{player.track_id:03d}.jpg"
            thumb = player.to_thumbnail((150, 200))
            cv2.imwrite(str(thumb_path), thumb)

            result["players"].append({
                "track_id":    player.track_id,
                "appearances": player.appearances,
                "confidence":  round(player.confidence, 3),
                "thumbnail":   str(thumb_path),
                "thumbnail_base64": player.crop_base64(),
            })

        # Save JSON
        json_path = out_path / "players.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("Exported %d player thumbnails to %s", len(players), output_dir)
        return result

    def _detect_persons(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """Run YOLO and return person detections."""
        results = self._model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # 0 = person in COCO
                    continue
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "bbox": tuple(xyxy),
                    "conf": conf,
                })

        return detections


# ═════════════════════════════════════════════════════════════════════════════
# FOCUSED ANALYZER — runs full pipeline focused on one player
# ═════════════════════════════════════════════════════════════════════════════
class FocusedAnalyzer:
    """
    Wraps the main PurpleBoxApp to run analysis focused on a single player.

    The selected player gets:
    - A highlighted bounding box (gold color, thicker)
    - Their own personal highlight reel
    - A dedicated stats output
    - Re-identification if track is lost (appearance matching)
    """

    def __init__(
        self,
        focus_track_id: int,
        jersey_number:  str = "",
        player_name:    str = "",
        config:         Optional[Dict] = None,
    ):
        self.focus_track_id = focus_track_id
        self.jersey_number  = jersey_number
        self.player_name    = player_name
        self.config         = config or {}

        # Store appearance embedding for re-identification
        self._appearance_hist: Optional[np.ndarray] = None

    def capture_appearance(self, crop: np.ndarray) -> None:
        """
        Compute a color histogram embedding from the player crop.
        Used to re-identify the player if track is lost.
        """
        if crop is None or crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        self._appearance_hist = hist.flatten()

    def match_appearance(self, crop: np.ndarray, threshold: float = 0.6) -> float:
        """
        Compare a crop against the stored appearance.
        Returns similarity score (0-1). Higher = more similar.
        """
        if self._appearance_hist is None or crop is None or crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten()

        # Correlation comparison
        score = cv2.compareHist(
            self._appearance_hist.reshape(-1, 1).astype(np.float32),
            hist_flat.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        return max(0.0, score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_track_id": self.focus_track_id,
            "jersey_number":  self.jersey_number,
            "player_name":    self.player_name,
            "has_appearance":  self._appearance_hist is not None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Purple Box Sports — Player Detection & Selection",
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--scan-seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", default="output/player_select")
    parser.add_argument("--min-appearances", type=int, default=5)

    args = parser.parse_args()

    scanner = PlayerScanner(
        model_path=args.model,
        device=args.device,
        scan_seconds=args.scan_seconds,
        min_appearances=args.min_appearances,
    )

    result = scanner.scan_and_export(args.video, args.output_dir)

    print(f"\n{'═' * 50}")
    print(f"  PLAYER DETECTION RESULTS")
    print(f"{'═' * 50}")
    print(f"  Players found: {result['players_found']}")
    print(f"  Scan duration: {result['scanned_seconds']}s")
    print()

    for p in result["players"]:
        print(f"  Track #{p['track_id']:>3} | {p['appearances']:>3} frames | conf={p['confidence']:.2f} | {p['thumbnail']}")

    print(f"\n  Output: {args.output_dir}/players.json")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
