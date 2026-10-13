"""Fine-tune the detection models on your own footage (runs on the GPU server).

Wraps Ultralytics training so the ball/rim model, the player model and the pose model are
all trained the same way from a YOLO ``data.yaml`` (e.g. a Roboflow export such as the
``basketball-and-rim`` set). Phone footage differs enough from broadcast and stock COCO
images that fine-tuning on frames from your own gyms is the biggest detection win.
"""

from __future__ import annotations

from pathlib import Path


def train_detector(data: str | Path, model: str = "yolov8s.pt", epochs: int = 100,
                   imgsz: int = 1280, batch: int = 16, device: str = "0",
                   project: str | Path = "models/train", name: str = "detector") -> dict:
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
               project=str(project), name=name, patience=25, cos_lr=True,
               mosaic=1.0, degrees=3.0, translate=0.1, scale=0.5, fliplr=0.5,
               hsv_v=0.4, plots=True)
    metrics = yolo.val(data=str(data), imgsz=imgsz, device=device)
    best = Path(project) / name / "weights" / "best.pt"
    return {
        "weights": str(best),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }


def export_frames(video: str | Path, out_dir: str | Path, every_s: float = 2.0) -> int:
    """Dump frames from a game to label (upload the folder to Roboflow, label, export)."""
    import cv2

    from ..video import iter_frames, probe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    n = 0
    for idx, frame in iter_frames(video, stride=max(1, int(every_s * meta.fps))):
        cv2.imwrite(str(out / f"{Path(video).stem}_{idx:07d}.jpg"), frame)
        n += 1
    return n
