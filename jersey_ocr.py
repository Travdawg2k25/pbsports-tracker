"""
jersey_ocr.py — Full jersey number OCR module.
Reads jersey numbers from player crops using EasyOCR with
multi-stage preprocessing, confidence filtering, and majority-vote smoothing.
Purple Box Sports — Full Production Version
"""

import cv2
import numpy as np
import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed. Jersey OCR disabled.")
    print("    Install with: pip install easyocr")


class PreprocessPipeline:
    """
    Multi-stage image preprocessing pipeline for jersey number OCR.
    Tries multiple preprocessing strategies and picks the best result.
    """

    def __init__(self, scale: int = 3):
        self.scale = scale

    def run_all(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Return multiple preprocessed versions of the crop
        for multi-attempt OCR.
        """
        results = []

        # Upscale
        h, w = crop.shape[:2]
        if h < 10 or w < 5:
            return results

        up = cv2.resize(
            crop,
            (w * self.scale, h * self.scale),
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        # Strategy 1: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)

        # Strategy 2: Inverted CLAHE + Otsu
        _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results.append(otsu_inv)

        # Strategy 3: Adaptive threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        results.append(adaptive)

        # Strategy 4: Sharpened
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, sharp_thresh = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        results.append(sharp_thresh)

        # Strategy 5: Raw grayscale (for EasyOCR's internal processing)
        results.append(gray)

        return results


class JerseyOCR:
    """
    Full jersey number OCR module.

    Features:
    - Multi-strategy image preprocessing
    - Multi-attempt OCR with best-result selection
    - Confidence-weighted voting
    - Per-track majority vote smoothing
    - Persistent confirmed assignments
    - Automatic track cleanup on ID loss
    - Debug mode for visualization
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        smoothing_window: int = 25,
        min_votes_to_confirm: int = 6,
        use_gpu: bool = False,
        debug: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.min_votes_to_confirm = min_votes_to_confirm
        self.debug = debug

        # Per-track reading history: {track_id: [(number, confidence), ...]}
        self.vote_history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

        # Confirmed assignments: {track_id: jersey_number}
        self.confirmed: Dict[int, str] = {}

        # Preprocessing pipeline
        self.preprocessor = PreprocessPipeline(scale=3)

        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            print("🔍  Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(
                ["en"],
                gpu=use_gpu,
                verbose=False,
            )
            print("✅  EasyOCR ready.")
        else:
            self.reader = None

        # Stats
        self.total_reads: int = 0
        self.successful_reads: int = 0
        self.confirmed_count: int = 0

    # ------------------------------------------------------------------
    # Main read method
    # ------------------------------------------------------------------

    def read(
        self,
        frame: np.ndarray,
        bbox: list,
        track_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        Attempt to read a jersey number from a player bounding box.

        Args:
            frame:    Full video frame (BGR)
            bbox:     Player bounding box [x1, y1, x2, y2]
            track_id: Tracker ID for vote smoothing

        Returns:
            Jersey number string (e.g. "23") or None
        """
        # Return confirmed assignment immediately
        if track_id is not None and track_id in self.confirmed:
            return self.confirmed[track_id]

        if self.reader is None:
            return None

        # Crop jersey region
        crop = self._crop_jersey_region(frame, bbox)
        if crop is None:
            return None

        self.total_reads += 1

        # Multi-attempt OCR
        result = self._multi_attempt_ocr(crop)

        if result is not None:
            number, confidence = result
            self.successful_reads += 1

            if track_id is not None:
                self._add_vote(track_id, number, confidence)
                confirmed = self._get_confirmed(track_id)
                if confirmed:
                    return confirmed

            return number

        return None

    def read_batch(
        self,
        frame: np.ndarray,
        players: list,
    ) -> Dict[int, Optional[str]]:
        """
        Read jersey numbers for a batch of players.

        Args:
            frame:   Full video frame
            players: List of track dicts with 'id' and 'bbox'

        Returns:
            Dict mapping track_id → jersey_number or None
        """
        results = {}
        for player in players:
            track_id = player.get("id")
            bbox = player.get("bbox")
            if bbox is None:
                continue
            results[track_id] = self.read(frame, bbox, track_id)
        return results

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------

    def _crop_jersey_region(
        self,
        frame: np.ndarray,
        bbox: list,
        top_frac: float = 0.12,
        bot_frac: float = 0.52,
        side_pad_frac: float = 0.10,
    ) -> Optional[np.ndarray]:
        """
        Crop the torso region of a player bounding box.
        The jersey number is typically in the upper torso area.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame - 1, x2)
        y2 = min(h_frame - 1, y2)

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        if bbox_h < 50 or bbox_w < 25:
            return None

        crop_y1 = y1 + int(bbox_h * top_frac)
        crop_y2 = y1 + int(bbox_h * bot_frac)
        pad_x = int(bbox_w * side_pad_frac)
        crop_x1 = x1 + pad_x
        crop_x2 = x2 - pad_x

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
            return None

        return crop

    # ------------------------------------------------------------------
    # Multi-attempt OCR
    # ------------------------------------------------------------------

    def _multi_attempt_ocr(
        self,
        crop: np.ndarray,
    ) -> Optional[Tuple[str, float]]:
        """
        Run OCR on multiple preprocessed versions of the crop.
        Returns the highest-confidence valid result.
        """
        processed_versions = self.preprocessor.run_all(crop)
        if not processed_versions:
            return None

        best_result: Optional[Tuple[str, float]] = None
        best_conf = 0.0

        for processed in processed_versions:
            result = self._ocr_single(processed)
            if result is not None:
                number, conf = result
                if conf > best_conf:
                    best_conf = conf
                    best_result = (number, conf)

        return best_result

    def _ocr_single(
        self,
        image: np.ndarray,
    ) -> Optional[Tuple[str, float]]:
        """Run EasyOCR on a single preprocessed image."""
        try:
            results = self.reader.readtext(
                image,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            if self.debug:
                print(f"  OCR error: {e}")
            return None

        candidates = []
        for (_, text, conf) in results:
            if conf < self.confidence_threshold:
                continue
            cleaned = re.sub(r"[^0-9]", "", text).strip()
            if 1 <= len(cleaned) <= 2:
                candidates.append((cleaned, float(conf)))

        if not candidates:
            return None

        # Return highest confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Vote smoothing
    # ------------------------------------------------------------------

    def _add_vote(self, track_id: int, number: str, confidence: float):
        """Add a weighted vote for a jersey number reading."""
        history = self.vote_history[track_id]
        history.append((number, confidence))
        if len(history) > self.smoothing_window:
            history.pop(0)

    def _get_confirmed(self, track_id: int) -> Optional[str]:
        """
        Return a confirmed jersey number using confidence-weighted voting.
        Requires at least min_votes_to_confirm readings.
        """
        history = self.vote_history[track_id]
        if len(history) < self.min_votes_to_confirm:
            return None

        # Weighted vote count
        weighted: Dict[str, float] = defaultdict(float)
        for number, conf in history:
            weighted[number] += conf

        best = max(weighted, key=weighted.__getitem__)
        total_weight = sum(weighted.values())

        # Must represent a clear majority of total confidence weight. Raised
        # from 0.50 to 0.65 to stop premature confirmation on noisy readings
        # (was flip-flopping between similar numbers on low-res footage).
        if weighted[best] / max(total_weight, 1e-6) >= 0.65:
            if track_id not in self.confirmed:
                self.confirmed_count += 1
                if self.debug:
                    print(f"  ✅  Jersey confirmed: Track #{track_id} → #{best}")
            self.confirmed[track_id] = best
            return best

        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_confirmed(self) -> Dict[int, str]:
        """Return all confirmed jersey number assignments."""
        return dict(self.confirmed)

    def is_confirmed(self, track_id: int) -> bool:
        """Check if a track has a confirmed jersey number."""
        return track_id in self.confirmed

    def reset_track(self, track_id: int):
        """Clear vote history and confirmation for a lost track."""
        self.vote_history.pop(track_id, None)
        self.confirmed.pop(track_id, None)

    def reset_all(self):
        """Clear all history (use between games)."""
        self.vote_history.clear()
        self.confirmed.clear()
        self.total_reads = 0
        self.successful_reads = 0
        self.confirmed_count = 0

    def get_stats(self) -> dict:
        """Return OCR performance statistics."""
        success_rate = (
            round(self.successful_reads / self.total_reads, 3)
            if self.total_reads > 0 else 0.0
        )
        return {
            "total_reads": self.total_reads,
            "successful_reads": self.successful_reads,
            "success_rate": success_rate,
            "confirmed_jerseys": self.confirmed_count,
            "confirmed_assignments": self.get_all_confirmed(),
        }

    def draw_debug(
        self,
        frame: np.ndarray,
        players: list,
    ) -> np.ndarray:
        """
        Draw jersey number OCR results on the frame for debugging.
        """
        out = frame.copy()
        for player in players:
            track_id = player.get("id")
            bbox = player.get("bbox")
            if bbox is None:
                continue

            jersey = self.confirmed.get(track_id)
            if jersey:
                x1, y1 = int(bbox[0]), int(bbox[1])
                cv2.putText(
                    out,
                    f"#{jersey}",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 200),
                    2,
                )

        return out