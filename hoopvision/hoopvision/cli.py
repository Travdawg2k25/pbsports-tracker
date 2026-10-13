"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import uvicorn

from .config import Config
from .pipeline import STAGES, run
from .video import FrameSeeker, probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hoopvision", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cal = sub.add_parser("calibrate", help="mark court landmarks and rims for a camera position")
    cal.add_argument("video")
    cal.add_argument("--out", default="calibration.json")
    cal.add_argument("--frame", type=int, default=0, help="frame to calibrate on")
    cal.add_argument("--port", type=int, default=8001)
    cal.add_argument("--host", default="127.0.0.1")

    ana = sub.add_parser("analyze", help="run the analysis pipeline over a game video")
    ana.add_argument("video")
    ana.add_argument("--out", default="runs/game")
    ana.add_argument("--calibration")
    ana.add_argument("--config")
    ana.add_argument("--roster")
    ana.add_argument("--from", dest="start_stage", default="motion", choices=STAGES)
    ana.add_argument("--max-frames", type=int)
    ana.add_argument(
        "--no-stabilize",
        dest="stabilize",
        action="store_false",
        help="skip camera motion estimation (a genuinely locked-off camera)",
    )
    ana.add_argument("--device", help="override inference device, e.g. cuda:0")

    srv = sub.add_parser("serve", help="browse the box score")
    srv.add_argument("run_dir")
    srv.add_argument("--video")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--host", default="127.0.0.1")

    ds = sub.add_parser("build-dataset", help="training examples from reviewed runs")
    ds.add_argument("runs", nargs="+", help="run directories with edits.json")
    ds.add_argument("--out", default="models/events.npz")

    te = sub.add_parser("train-events", help="train the temporal event model (GPU server)")
    te.add_argument("dataset", help="npz from build-dataset")
    te.add_argument("--out", default="models/events.pt")
    te.add_argument("--epochs", type=int, default=40)
    te.add_argument("--device", default="cpu")

    td = sub.add_parser("train-detector", help="fine-tune a YOLO ball/rim/player/pose model")
    td.add_argument("data", help="YOLO data.yaml, e.g. a Roboflow export")
    td.add_argument("--model", default="yolov8s.pt")
    td.add_argument("--epochs", type=int, default=100)
    td.add_argument("--imgsz", type=int, default=1280)
    td.add_argument("--batch", type=int, default=16)
    td.add_argument("--device", default="0")
    td.add_argument("--name", default="detector")

    fr = sub.add_parser("export-frames", help="dump frames from a game for labelling")
    fr.add_argument("video")
    fr.add_argument("--out", default="frames")
    fr.add_argument("--every", type=float, default=2.0, help="seconds between frames")

    ev = sub.add_parser("evaluate", help="compare a run's box score to a hand-kept one")
    ev.add_argument("run_dir")
    ev.add_argument("truth", help="CSV: team,jersey,pts,fgm,fga,...,pf")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "calibrate":
        return _calibrate(args)
    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "serve":
        return _serve(args)
    return _ml(args)


def _ml(args: argparse.Namespace) -> int:
    import json

    if args.cmd == "build-dataset":
        from .ml.dataset import build_dataset

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result = build_dataset([Path(r) for r in args.runs], Path(args.out))
    elif args.cmd == "train-events":
        from .ml.event_model import train

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result = train(Path(args.dataset), Path(args.out), args.epochs, device=args.device)
    elif args.cmd == "train-detector":
        from .ml.train_detector import train_detector

        result = train_detector(args.data, args.model, args.epochs, args.imgsz, args.batch,
                                args.device, name=args.name)
    elif args.cmd == "export-frames":
        from .ml.train_detector import export_frames

        result = {"frames": export_frames(args.video, args.out, args.every)}
    else:
        from .ml.evaluate import evaluate, load_truth

        box = json.loads((Path(args.run_dir) / "boxscore.json").read_text())
        result = evaluate(box, load_truth(args.truth))
    print(json.dumps(result, indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from .api import create_calibration_app

    meta = probe(args.video)
    frame_path = Path(args.out).with_suffix(".frame.jpg")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    with FrameSeeker(args.video) as seeker:
        img = seeker.get(args.frame)
    if img is None:
        raise SystemExit(f"could not read frame {args.frame} of {args.video}")
    cv2.imwrite(str(frame_path), img)
    print(f"{meta.width}x{meta.height} @ {meta.fps:.2f} fps — open http://{args.host}:{args.port}")
    uvicorn.run(create_calibration_app(frame_path, args.out), host=args.host, port=args.port)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.device:
        cfg.detection.device = args.device
        cfg.detection.half = args.device != "cpu"
    box = run(
        video=args.video,
        out_dir=args.out,
        calibration=args.calibration,
        config=cfg,
        roster=args.roster,
        start_stage=args.start_stage,
        max_frames=args.max_frames,
        stabilize=args.stabilize,
    )
    for p in box["players"][:15]:
        print(
            f"{(p['team'] or '?'):>7} #{p['jersey'] or '--':<3} "
            f"{p['points']:>3} pts  {p['fgm']}/{p['fga']} fg  "
            f"{p['rebounds']} reb  {p['assists']} ast"
        )
    print(f"\nwrote {Path(args.out) / 'boxscore.json'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .api import create_app

    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.run_dir, args.video), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
