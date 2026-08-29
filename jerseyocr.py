"""
jersey_ocr.py - Jersey number OCR for Purple Box Sports
Uses EasyOCR to read jersey numbers from player crops
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger('PurpleBoxSports.JerseyOCR')


# ============================================================
# JERSEY OCR
# ============================================================

class JerseyOCR:
    """
    Reads jersey numbers from player bounding box crops.

    Pipeline:
    1. Preprocess crop (contrast, resize, grayscale)
    2. Run EasyOCR
    3. Filter for numeric results
    4. Validate (1-3 digits, plausible jersey range)
    5. Cache results per track ID
    """

    def __init__(self, config: dict):
        self.config = config

        self.min_confidence = config.get('ocr_confidence', 0.6)
        self.cache_size     = config.get('ocr_cache_size', 10)

        # Per-player vote cache: track_id → Counter of number strings
        self.number_votes:   Dict[int, Counter] = defaultdict(Counter)
        self.confirmed:      Dict[int, dict]    = {}

        # Initialize EasyOCR
        self._init_ocr()

        logger.info("JerseyOCR initialized ✓")

    def _init_ocr(self):
        """Initialize EasyOCR reader"""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                ['en'],
                gpu=True,
                verbose=False
            )
            self.ocr_available = True
            logger.info("EasyOCR loaded (GPU)")
        except Exception as e:
            try:
                import easyocr
                self.reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    verbose=False
                )
                self.ocr_available = True
                logger.info("EasyOCR loaded (CPU)")
            except ImportError:
                logger.warning(
                    "EasyOCR not available. "
                    "Install with: pip install easyocr"
                )
                self.reader        = None
                self.ocr_available = False

    def read_jersey(
        self,
        crop: np.ndarray,
        track_id: Optional[int] = None
    ) -> Optional[dict]:
        """
        Read jersey number from a player crop.

        Args:
            crop:     BGR image of player bounding box
            track_id: Optional track ID for vote caching

        Returns:
            dict with 'number' and 'confidence', or None
        """
        if not self.ocr_available or self.reader is None:
            return None

        if crop is None or crop.size == 0:
            return None

        # Check confirmed cache
        if track_id is not None and track_id in self.confirmed:
            return self.confirmed[track_id]

        # Preprocess
        processed = self._preprocess(crop)
        if processed is None:
            return None

        # Run OCR
        try:
            results = self.reader.readtext(
                processed,
                allowlist='0123456789',
                detail=1,
                paragraph=False
            )
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None

        # Parse results
        best = self._parse_results(results)

        if best is None:
            return None

        # Vote caching
        if track_id is not None:
            self.number_votes[track_id][best['number']] += 1
            best = self._get_voted_result(track_id, best)

        return best

    def _preprocess(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess player crop for OCR.
        Focuses on the jersey number region (upper-middle of crop).
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]

        # Extract jersey number region (upper chest area)
        y1 = int(h * 0.15)
        y2 = int(h * 0.55)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)

        region = crop[y1:y2, x1:x2]

        if region.size == 0:
            return None

        # Upscale for better OCR
        scale  = max(1, 80 // max(region.shape[:2]))
        scaled = cv2.resize(
            region,
            None,
            fx=max(scale, 2),
            fy=max(scale, 2),
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced  = clahe.apply(gray)

        # Try both dark-on-light and light-on-dark
        _, thresh_dark  = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh_light = cv2.bitwise_not(thresh_dark)

        # Stack both versions for OCR
        combined = np.hstack([thresh_dark, thresh_light])

        return combined

    def _parse_results(self, results: list) -> Optional[dict]:
        """
        Parse EasyOCR results to extract jersey number.

        Args:
            results: List of (bbox, text, confidence) tuples

        Returns:
            dict with 'number' and 'confidence', or None
        """
        candidates = []

        for (bbox, text, conf) in results:
            # Clean text
            cleaned = ''.join(c for c in text if c.isdigit())

            if not cleaned:
                continue

            # Valid jersey number: 0-99 (1-2 digits), some allow 3
            if not (1 <= len(cleaned) <= 2):
                continue

            number = int(cleaned)

            # Plausible jersey range
            if not (0 <= number <= 99):
                continue

            if conf >= self.min_confidence:
                candidates.append({
                    'number':     cleaned,
                    'int_value':  number,
                    'confidence': float(conf),
                })

        if not candidates:
            return None

        # Return highest confidence
        best = max(candidates, key=lambda x: x['confidence'])
        return {
            'number':     best['number'],
            'confidence': best['confidence'],
        }

    def _get_voted_result(
        self,
        track_id: int,
        current: dict
    ) -> dict:
        """
        Use voting across frames to confirm jersey number.
        Returns most voted number once threshold is reached.
        """
        votes = self.number_votes[track_id]
        total_votes = sum(votes.values())

        if total_votes < 3:
            return current

        # Most common number
        top_number, top_count = votes.most_common(1)[0]

        # Confirm if dominant
        if top_count >= max(3, total_votes * 0.6):
            confirmed = {
                'number':     top_number,
                'confidence': min(0.99, top_count / total_votes),
                'vote_count': top_count,
            }
            self.confirmed[track_id] = confirmed
            return confirmed

        return current

    def get_confirmed_jerseys(self) -> Dict[int, str]:
        """Return all confirmed jersey numbers by track ID"""
        return {
            tid: info['number']
            for tid, info in self.confirmed.items()
        }

    def reset_player(self, track_id: int):
        """Reset vote cache for a player (e.g., track ID reassigned)"""
        self.number_votes.pop(track_id, None)
        self.confirmed.pop(track_id, None)