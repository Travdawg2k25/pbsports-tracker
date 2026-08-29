# pipeline.py — Purple Box Sports | Analysis Pipeline
# =====================================================
# Connects: YOLO detection → tracking → event detection → output
# =====================================================

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("PurpleBox.Pipeline")


def _point_in_poly(x: float, y: float, poly) -> bool:
    """Ray-casting point-in-polygon test. poly = [[x,y], ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — detection disabled")

try:
    from tracker import (
        MultiObjectTracker,
        BallTracker,
        RimTracker,
        Visualizer,
        Detection,
    )
    TRACKER_AVAILABLE = True
except ImportError as e:
    TRACKER_AVAILABLE = False
    logger.warning("tracker import failed: %s", e)

try:
    from basketball_events import BasketballEventDetector
    EVENTS_AVAILABLE = True
except ImportError as e:
    EVENTS_AVAILABLE = False
    logger.warning("basketball_events import failed: %s", e)

try:
    from stats_engine import StatsEngine
    STATS_AVAILABLE = True
except ImportError as e:
    STATS_AVAILABLE = False
    logger.warning("stats_engine import failed: %s", e)

try:
    from dead_ball_detector import DeadBallDetector
    DEADBALL_AVAILABLE = True
except ImportError as e:
    DEADBALL_AVAILABLE = False
    logger.warning("dead_ball_detector import failed: %s", e)

try:
    from jersey_ocr import JerseyOCR
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning("jersey_ocr import failed: %s", e)

try:
    from team_classifier import TeamClassifier
    TEAM_AVAILABLE = True
except ImportError as e:
    TEAM_AVAILABLE = False
    logger.warning("team_classifier import failed: %s", e)


# ═════════════════════════════════════════════════════════════════════════════
# DETECTION RESULT  (lightweight dataclass)
# ═════════════════════════════════════════════════════════════════════════════
class DetectionResult:
    __slots__ = ("bbox", "confidence", "class_name", "class_id",
                 "track_id", "jersey")

    def __init__(
        self,
        bbox:       List[float],
        confidence: float,
        class_name: str,
        class_id:   int = -1,
        track_id:   int = -1,
        jersey:     str = "",
    ):
        self.bbox       = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.class_id   = class_id
        self.track_id   = track_id
        self.jersey     = jersey

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox":       self.bbox,
            "confidence": round(self.confidence, 3),
            "class":      self.class_name,
            "class_id":   self.class_id,
            "track_id":   self.track_id,
            "jersey":     self.jersey,
        }

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


# ═════════════════════════════════════════════════════════════════════════════
# YOLO DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(
        self,
        model_path:     str,
        device:         str   = "cpu",
        conf:           float = 0.35,
        iou:            float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        self.model_path     = model_path
        self.device         = device
        self.conf           = conf
        self.iou            = iou
        self.target_classes = target_classes
        self._model:        Optional[Any]       = None
        self._class_names:  Dict[int, str]      = {}
        self._load()

    def _load(self) -> None:
        if not YOLO_AVAILABLE:
            logger.error("Cannot load YOLO — ultralytics missing")
            return
        try:
            self._model = YOLO(self.model_path)
            self._class_names = self._model.names
            logger.info(
                "Loaded YOLO: %s | classes=%s | device=%s",
                self.model_path,
                list(self._class_names.values()),
                self.device,
            )
        except Exception as e:
            logger.error("YOLO load failed (%s): %s", self.model_path, e)
            self._model = None

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self._model is None:
            return []
        try:
            results = self._model(
                frame,
                device  = self.device,
                conf    = self.conf,
                iou     = self.iou,
                verbose = False,
            )
        except Exception as e:
            logger.error("YOLO inference error: %s", e)
            return []

        detections: List[DetectionResult] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, f"cls_{cls_id}")
                conf     = float(box.conf[0])
                xyxy     = box.xyxy[0].tolist()

                if self.target_classes and cls_name not in self.target_classes:
                    continue

                detections.append(DetectionResult(
                    bbox       = [float(v) for v in xyxy],
                    confidence = conf,
                    class_name = cls_name,
                    class_id   = cls_id,
                ))
        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    def is_ready(self) -> bool:
        return self._model is not None


# ═════════════════════════════════════════════════════════════════════════════
# FRAME ANNOTATOR
# ═════════════════════════════════════════════════════════════════════════════
class FrameAnnotator:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    COLOR_MAP = {
        "person":     (180,   0, 180),
        "player":     (180,   0, 180),
        "basketball": (  0, 215, 255),
        "ball":       (  0, 215, 255),
        "rim":        (  0, 200,   0),
        "default":    (255, 255, 255),
    }

    # Event label colors
    EVENT_COLORS = {
        "score":   (  0, 215, 255),
        "rebound": (  0, 255,   0),
        "steal":   (255, 165,   0),
        "block":   (255,  50,  50),
        "assist":  (255, 255,   0),
        "default": (255, 255, 255),
    }

    def __init__(self, draw_trails: bool = True, draw_zones: bool = True):
        self.draw_trails = draw_trails
        self.draw_zones  = draw_zones
        self._trails:    Dict[int, List[Tuple[int, int]]] = {}
        self._max_trail  = 30

    def annotate(
        self,
        frame:      np.ndarray,
        detections: List[DetectionResult],
        ball_pos:   Optional[Tuple[int, int]] = None,
        rim_pos:    Optional[Tuple[int, int]] = None,
        events:     Optional[List[Dict]]      = None,
    ) -> np.ndarray:
        out = frame.copy()

        if self.draw_trails and ball_pos:
            self._update_trail(-1, ball_pos)
            self._draw_trail(out, -1, (0, 215, 255))

        if rim_pos:
            rx, ry = int(rim_pos[0]), int(rim_pos[1])
            cv2.circle(out, (rx, ry), 18, (0, 200, 0), 2)
            cv2.putText(out, "RIM", (rx - 15, ry - 22),
                        self.FONT, 0.45, (0, 200, 0), 1)

        for det in detections:
            self._draw_detection(out, det)

        if events:
            self._draw_event_flashes(out, events)

        return out

    # ── private helpers ───────────────────────────────────────────────────────
    def _draw_detection(self, frame: np.ndarray, det: DetectionResult) -> None:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = self.COLOR_MAP.get(det.class_name, self.COLOR_MAP["default"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = []
        if det.track_id >= 0:
            parts.append(f"#{det.track_id}")
        if det.jersey:
            parts.append(f"J{det.jersey}")
        parts.append(f"{det.class_name} {det.confidence:.2f}")
        label = " ".join(parts)

        label_y = max(y1 - 6, 14)
        (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.45, 1)
        cv2.rectangle(frame,
                      (x1, label_y - th - 4),
                      (x1 + tw + 4, label_y + 2),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y),
                    self.FONT, 0.45, (0, 0, 0), 1)

        if self.draw_trails and det.track_id >= 0:
            self._update_trail(det.track_id, det.center)
            self._draw_trail(frame, det.track_id, color)

    def _update_trail(self, tid: int, pos: Tuple[int, int]) -> None:
        if tid not in self._trails:
            self._trails[tid] = []
        trail = self._trails[tid]
        trail.append(pos)
        if len(trail) > self._max_trail:
            trail.pop(0)

    def _draw_trail(
        self,
        frame: np.ndarray,
        tid:   int,
        color: Tuple[int, int, int],
    ) -> None:
        trail = self._trails.get(tid, [])
        if len(trail) < 2:
            return
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            c     = tuple(int(v * alpha) for v in color)
            pt1   = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2   = (int(trail[i][0]),     int(trail[i][1]))
            cv2.line(frame, pt1, pt2, c, max(1, int(alpha * 3)))

    def _draw_event_flashes(
        self,
        frame:  np.ndarray,
        events: List[Dict],
    ) -> None:
        h, w   = frame.shape[:2]
        y_base = 60
        for i, event in enumerate(events[-5:]):   # show last 5 events max
            etype  = event.get("type", "event").lower()
            label  = event.get("label", etype.upper())
            color  = self.EVENT_COLORS.get(etype, self.EVENT_COLORS["default"])
            y_pos  = y_base + i * 36

            (tw, th), _ = cv2.getTextSize(
                label, self.FONT_BOLD, 0.9, 2)
            # Semi-transparent background pill
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (w - tw - 24, y_pos - th - 6),
                          (w - 8,       y_pos + 6),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, label,
                        (w - tw - 16, y_pos),
                        self.FONT_BOLD, 0.9, color, 2)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE  (main orchestrator)
# ═════════════════════════════════════════════════════════════════════════════
class AnalysisPipeline:
    """
    Orchestrates:
      YOLO detection → tracking → event detection →
      stats → dead-ball filtering → annotated output
    """

    def __init__(self, config: Dict[str, Any]):
        self.config   = config
        self.device   = config.get("device", "cpu")
        self._frame_count  = 0
        self._fps          = config.get("fps", 30.0)
        self._recent_events: List[Dict] = []

        # ── Detectors ────────────────────────────────────────────────────────
        player_model = config.get("player_model", "yolov8n.pt")
        ball_model   = config.get("ball_model",   "basketball_rim_best.pt")

        self.player_detector = YOLODetector(
            model_path     = player_model,
            device         = self.device,
            conf           = config.get("player_conf", 0.35),
            target_classes = ["person", "player"],
        )
        # Low ball_conf so we catch nearly every ball; BallTracker's
        # trajectory gating rejects the resulting false positives.
        self.ball_detector = YOLODetector(
            model_path     = ball_model,
            device         = self.device,
            conf           = config.get("ball_conf", 0.10),
            target_classes = config.get(
                "ball_classes", ["basketball", "ball", "rim"]),
        )

        # ── Trackers ─────────────────────────────────────────────────────────
        if TRACKER_AVAILABLE:
            self.player_tracker = MultiObjectTracker(
                max_age=config.get("max_disappeared", 30),
                max_distance=config.get("max_distance",    120),
            )
            self.ball_tracker = BallTracker(
                trajectory_len = config.get("ball_history", 60),
            )
            self.rim_tracker = RimTracker()
        else:
            self.player_tracker = None
            self.ball_tracker   = None
            self.rim_tracker    = None

        # ── Event detector ────────────────────────────────────────────────────
        self.event_detector = (
            BasketballEventDetector(
                frame_width=config.get("display_width", 1280),
                frame_height=config.get("display_height", 720),
            )
            if EVENTS_AVAILABLE else None
        )

        # ── Stats engine ──────────────────────────────────────────────────────
        self.stats_engine = (
            StatsEngine() if STATS_AVAILABLE else None
        )

        # ── Dead ball detector ────────────────────────────────────────────────
        self.dead_ball_detector = (
            DeadBallDetector(
                fps=self._fps,
                missing_frames_threshold=int(self._fps * 3),   # 3 seconds missing
                stationary_threshold_px=4.0,
                stationary_frames=int(self._fps * 4),          # 4 seconds stationary
                out_of_bounds_margin=50,
                min_dead_duration_sec=2.0,
            )
            if DEADBALL_AVAILABLE else None
        )

        # ── Jersey OCR ────────────────────────────────────────────────────────
        self.jersey_ocr = (
            JerseyOCR() if OCR_AVAILABLE else None
        )

        # ── Team classifier (jersey-color 2-means) ───────────────────────────
        self.team_classifier = (
            TeamClassifier() if TEAM_AVAILABLE else None
        )

        # ── Annotator ─────────────────────────────────────────────────────────
        self.annotator = FrameAnnotator(
            draw_trails = config.get("draw_trails", True),
            draw_zones  = config.get("draw_zones",  True),
        )

        logger.info(
            "AnalysisPipeline ready | device=%s | "
            "tracker=%s | events=%s | stats=%s | deadball=%s | ocr=%s | team=%s",
            self.device,
            TRACKER_AVAILABLE,
            EVENTS_AVAILABLE,
            STATS_AVAILABLE,
            DEADBALL_AVAILABLE,
            OCR_AVAILABLE,
            TEAM_AVAILABLE,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PLAYER FILTERING
    # ─────────────────────────────────────────────────────────────────────────
    def _filter_players(self, dets, frame_h, frame_w):
        """
        Drop person detections that are unlikely to be on-court players.

        Heuristics (calibration-free):
          - Minimum box height as a fraction of frame height. Distant
            crowd/bench people are small; on-court players are tall.
          - Optional court polygon (config 'court_polygon' = [[x,y],...] in
            pixels): keep only detections whose feet fall inside it.
        Both are configurable; defaults are permissive so we don't drop real
        players on wide-angle footage.
        """
        min_h_frac = self.config.get("player_min_height_frac", 0.12)
        max_h_frac = self.config.get("player_max_height_frac", 0.95)
        poly = self.config.get("court_polygon")  # optional list of [x,y]

        min_h = frame_h * min_h_frac
        max_h = frame_h * max_h_frac

        kept = []
        for d in dets:
            if d.height < min_h or d.height > max_h:
                continue
            if poly:
                fx = (d.bbox[0] + d.bbox[2]) / 2.0
                fy = d.bbox[3]  # feet
                if not _point_in_poly(fx, fy, poly):
                    continue
            kept.append(d)
        return kept

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(
        self,
        frame:       np.ndarray,
        timestamp:   float = 0.0,
        is_dead_ball: bool = False,
    ) -> Dict[str, Any]:
        """
        Process one video frame.

        Returns a dict with:
          - annotated_frame  : np.ndarray
          - detections       : list of dicts
          - ball_position    : (x, y) or None
          - rim_position     : (x, y) or None
          - events           : list of event dicts
          - is_dead_ball     : bool
          - frame_index      : int
          - timestamp        : float
        """
        self._frame_count += 1
        result: Dict[str, Any] = {
            "frame_index":    self._frame_count,
            "timestamp":      timestamp,
            "detections":     [],
            "ball_position":  None,
            "rim_position":   None,
            "events":         [],
            "is_dead_ball":   is_dead_ball,
            "annotated_frame": frame,
        }

        # ── 1. Detect ─────────────────────────────────────────────────────────
        player_dets = self.player_detector.detect(frame)
        ball_dets   = self.ball_detector.detect(frame)

        # ── 1b. Filter non-players (refs, coaches, crowd) ─────────────────────
        # A generic person detector also picks up sideline/bench/crowd people.
        # Those fragment tracking and pollute team + appearance matching. We
        # keep only detections whose box height is a plausible fraction of the
        # frame (on-court players appear larger than distant crowd) and, if a
        # court polygon is configured, whose feet fall inside it.
        player_dets = self._filter_players(player_dets, frame.shape[0], frame.shape[1])

        all_dets    = player_dets + ball_dets

        # ── 2. Separate ball / rim detections ─────────────────────────────────
        ball_det_list = [
            d for d in ball_dets
            if d.class_name in ("basketball", "ball")
        ]
        rim_det_list  = [
            d for d in ball_dets
            if d.class_name == "rim"
        ]

        # ── 3. Update trackers ────────────────────────────────────────────────
        ball_pos: Optional[Tuple[int, int]] = None
        rim_pos:  Optional[Tuple[int, int]] = None
        tracked: List = []

        if self.player_tracker:
            # Convert DetectionResult → Detection for the tracker
            tracker_dets = [
                Detection(
                    bbox=tuple(d.bbox),
                    conf=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    frame_id=self._frame_count,
                )
                for d in player_dets
            ]
            tracked = self.player_tracker.update(tracker_dets)

            # ── Team assignment (jersey color) ────────────────────────────────
            # Runs on the tracker's Track objects so team_id persists per
            # track_id and flows into the event engine (steals, passes,
            # turnovers, team scoreboard all depend on it).
            if self.team_classifier:
                self.team_classifier.assign(frame, tracked)

            # Assign track IDs (and learned team) back to original detections
            for track in tracked:
                # Find the closest original detection to this track
                for det in player_dets:
                    if det.track_id < 0:  # not yet assigned
                        tx, ty = track.center
                        dx, dy = det.center
                        if abs(tx - dx) < 20 and abs(ty - dy) < 20:
                            det.track_id = track.track_id
                            break

        if self.ball_tracker and ball_det_list:
            ball_tracker_dets = [
                Detection(
                    bbox=tuple(d.bbox),
                    conf=d.confidence,
                    class_id=d.class_id,
                    class_name="ball",
                    frame_id=self._frame_count,
                )
                for d in ball_det_list
            ]
            ball_pos = self.ball_tracker.update(ball_tracker_dets)
        elif ball_det_list:
            ball_pos = ball_det_list[0].center

        if self.rim_tracker and rim_det_list:
            rim_tracker_dets = [
                Detection(
                    bbox=tuple(d.bbox),
                    conf=d.confidence,
                    class_id=d.class_id,
                    class_name="rim",
                    frame_id=self._frame_count,
                )
                for d in rim_det_list
            ]
            rim_result = self.rim_tracker.update(rim_tracker_dets)
            if rim_result:
                rim_pos = ((rim_result[0] + rim_result[2]) / 2, (rim_result[1] + rim_result[3]) / 2)
        elif rim_det_list:
            rim_pos = rim_det_list[0].center

        result["ball_position"] = ball_pos
        result["rim_position"]  = rim_pos

        # ── 4. Jersey OCR → persistent Track identity ─────────────────────────
        # OCR runs on the tracker's persistent Track objects (not the raw
        # detections) and writes the confirmed jersey back onto the Track via
        # track.jersey_number. This is what the event engine and stats read,
        # so a player's stats stay attached to their jersey instead of a
        # volatile track_id.
        #
        # We read every `ocr_interval` frames per track. EasyOCR is expensive,
        # so we cap how many tracks we OCR per frame and round-robin through
        # unconfirmed tracks. Confirmed jerseys return instantly (cached), so
        # once a number is locked in it costs nothing.
        if self.jersey_ocr and tracked:
            ocr_interval = self.config.get("ocr_every_n_frames", 12)
            max_ocr_per_frame = self.config.get("ocr_max_per_frame", 3)
            if self._frame_count % ocr_interval == 0:
                # Prioritise tracks without a confirmed jersey
                unconfirmed = [
                    t for t in tracked
                    if getattr(t, "class_name", "") in ("player", "person")
                    and not self.jersey_ocr.is_confirmed(t.track_id)
                ]
                # Round-robin so we don't always OCR the same first N tracks
                start = (self._frame_count // ocr_interval) % max(len(unconfirmed), 1)
                ordered = unconfirmed[start:] + unconfirmed[:start]
                for t in ordered[:max_ocr_per_frame]:
                    x1, y1, x2, y2 = (int(v) for v in t.bbox)
                    jersey = self.jersey_ocr.read(
                        frame, [x1, y1, x2, y2], track_id=t.track_id
                    )
                    if jersey:
                        t.jersey_number = jersey

            # Always propagate any confirmed jersey onto the Track (cheap,
            # covers tracks confirmed on a previous frame) and mirror it to
            # the matching detection for annotation.
            for t in tracked:
                conf_j = self.jersey_ocr.confirmed.get(t.track_id)
                if conf_j:
                    t.jersey_number = conf_j
                    for det in player_dets:
                        if det.track_id == t.track_id:
                            det.jersey = conf_j
                            break

            # Clean up OCR history for tracks that no longer exist so memory
            # and vote history don't leak across a long game.
            live_ids = {t.track_id for t in tracked}
            stale = [tid for tid in self.jersey_ocr.confirmed if tid not in live_ids]
            # Keep confirmed jerseys around briefly; only drop vote history for
            # tracks that have fully disappeared.
            for tid in list(self.jersey_ocr.vote_history.keys()):
                if tid not in live_ids and tid not in self.jersey_ocr.confirmed:
                    self.jersey_ocr.vote_history.pop(tid, None)

        # ── 5. Dead ball check ────────────────────────────────────────────────
        # NOTE: Disabled standalone dead ball detector — ball detection rate
        # is too low (~14%) which causes constant false dead-ball states.
        # The event engine has its own game-state-based dead ball logic.
        is_dead = False
        result["is_dead_ball"] = False

        # ── 6. Event detection (skip during dead ball) ────────────────────────
        new_events: List[Dict] = []
        if self.event_detector and not is_dead:
            # Get tracked player objects for event engine
            player_tracks = self.player_tracker.active_tracks if self.player_tracker else []
            
            raw_events = self.event_detector.process_frame(
                frame_id=self._frame_count,
                ball_center=ball_pos,
                rim_center=rim_pos,
                player_tracks=player_tracks,
            )
            new_events = [{"type": e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type),
                           "frame": e.frame_id,
                           "player": getattr(e, 'primary_track_id', '') or '',
                           "jersey": getattr(e, 'primary_jersey', '') or '',
                           "team_id": getattr(e, 'primary_team_id', None),
                           "ball_position": getattr(e, 'ball_position', None),
                           "court_zone": getattr(e, 'court_zone', None),
                           "label": getattr(e, 'description', '') or str(getattr(e, 'event_type', ''))} 
                          for e in raw_events] if raw_events else []
            
            if new_events:
                self._recent_events.extend(new_events)
                self._recent_events = self._recent_events[-20:]

                # Push to stats engine
                if self.stats_engine:
                    for ev in new_events:
                        self.stats_engine.record_event(ev)

        result["events"] = new_events
        result["detections"] = [d.to_dict() for d in all_dets]

        # ── 7. Annotate frame ─────────────────────────────────────────────────
        annotated = self.annotator.annotate(
            frame      = frame,
            detections = all_dets,
            ball_pos   = ball_pos,
            rim_pos    = rim_pos,
            events     = self._recent_events,
        )

        # Dead ball overlay
        if is_dead:
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
            cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
            cv2.putText(
                annotated, "DEAD BALL",
                (w // 2 - 80, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (100, 100, 255), 2,
            )

        result["annotated_frame"] = annotated
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """Return current game stats from the stats engine."""
        if self.stats_engine:
            return self.stats_engine.get_summary()
        return {}

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Return the last *n* events."""
        return self._recent_events[-n:]

    def reset(self) -> None:
        """Reset all state for a new game/clip."""
        self._frame_count   = 0
        self._recent_events = []
        if self.player_tracker:
            self.player_tracker.reset()
        if self.ball_tracker:
            self.ball_tracker.reset()
        if self.event_detector:
            self.event_detector.reset()
        if self.stats_engine:
            self.stats_engine.reset()
        if self.dead_ball_detector:
            self.dead_ball_detector.reset()
        if self.team_classifier:
            self.team_classifier.reset()
        if self.jersey_ocr:
            self.jersey_ocr.reset_all()
        logger.info("Pipeline reset")

    def cleanup(self) -> None:
        """Release resources."""
        self.reset()
        logger.info("Pipeline cleaned up")

    # ─────────────────────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path:  str,
        output_path: Optional[str] = None,
        show:        bool = False,
        skip_dead:   bool = True,
    ) -> Dict[str, Any]:
        """
        Process an entire video file end-to-end.

        Returns a summary dict with stats and event list.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = fps

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_events:   List[Dict] = []
        frames_written = 0
        t_start        = time.time()

        logger.info(
            "Processing video: %s | %dx%d @ %.1f fps | %d frames",
            video_path, width, height, fps, total,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                result    = self.process_frame(frame, timestamp=timestamp)

                if result["events"]:
                    all_events.extend(result["events"])

                # Skip dead-ball frames from output video
                if skip_dead and result["is_dead_ball"]:
                    continue

                if writer:
                    writer.write(result["annotated_frame"])
                    frames_written += 1

                if show:
                    cv2.imshow("Purple Box Sports", result["annotated_frame"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Progress log every 300 frames
                if self._frame_count % 300 == 0:
                    elapsed = time.time() - t_start
                    pct     = (self._frame_count / total * 100) if total else 0
                    logger.info(
                        "  %.1f%% | frame %d/%d | %.1f fps processed",
                        pct, self._frame_count, total,
                        self._frame_count / max(elapsed, 0.001),
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        summary = {
            "video":          video_path,
            "frames_total":   self._frame_count,
            "frames_written": frames_written,
            "duration_s":     round(elapsed, 2),
            "events":         all_events,
            "stats":          self.get_stats(),
        }
        logger.info(
            "Done: %d frames in %.1fs | %d events | %d frames written",
            self._frame_count, elapsed, len(all_events), frames_written,
        )
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = {
        "device":       "cpu",
        "player_model": "yolov8n.pt",
        "ball_model":   "basketball_rim_best.pt",
        "draw_trails":  True,
        "fps":          30.0,
    }

    pipeline = AnalysisPipeline(cfg)

    # Smoke test with a blank frame
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    out   = pipeline.process_frame(dummy, timestamp=0.0)
    print("process_frame OK — keys:", list(out.keys()))
    print("Stats:", pipeline.get_stats())