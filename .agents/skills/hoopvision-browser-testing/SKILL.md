---
name: hoopvision-browser-testing
description: Exercise hoopvision calibration and review UIs locally using generated tracks and a synthetic video without expensive inference.
---

# Hoopvision browser testing

## Devin Secrets Needed
None for the local FastAPI applications.

## Setup
Use an existing Python environment with the package dependencies. For the package nested in pbsports-tracker, commands run from `hoopvision/`, not the repository root.

Generate a fresh temporary run using Python:
```python
from pathlib import Path
import json
from hoopvision import synthetic
from hoopvision.config import Config
from hoopvision.pipeline import stage_events, stage_boxscore
from hoopvision.types import VideoMeta
run = Path("/tmp/hoopvision-browser-run")
run.mkdir(exist_ok=True)
synthetic.generate(run)
meta = VideoMeta(**json.loads((run / "video.json").read_text()))
stage_events(run, Config(), synthetic.calibration(), meta)
stage_boxscore(run, meta, roster=None)
```

Start separate long-running processes:
```sh
hoopvision calibrate /tmp/hoopvision-browser-run/synthetic.mp4 --out /tmp/hoopvision-browser-run/ui-calibration.json --port 8765
hoopvision serve /tmp/hoopvision-browser-run --port 8766
```
Navigate to each localhost root. No login or real detection/OCR is needed.

## Browser playback compatibility
Ensure `ffmpeg` with libx264 is on PATH before generating fixtures. Current synthetic.generate automatically converts its OpenCV output to H.264 in place; freshly generated synthetic.mp4 should play directly without an override. Inspect the generated codec and test the original file first. When testing older revisions or diagnosing an unavailable encoder, if Chromium reports unsupported streams for MPEG-4 Part 2, preserve and report the original failure before isolating with a browser-compatible copy:
```sh
ffmpeg -i /tmp/hoopvision-browser-run/synthetic.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart /tmp/hoopvision-browser-run/synthetic-h264.mp4
hoopvision serve /tmp/hoopvision-browser-run --video /tmp/hoopvision-browser-run/synthetic-h264.mp4 --port 8767
```
Do not silently replace the original fixture or describe transcoded playback as native fixture playback.

## Server restarts after changes
Regenerate fixtures and restart both server processes after backend changes. Terminating a tool shell may leave its Python child listening: use `ss -ltnp` and `ps -fp <pid>` to identify only your own old server processes, terminate those PIDs, and confirm the replacement servers start. Use a fresh fixture directory to avoid confusing saved calibration evidence or codec overrides from previous runs.

## Useful assertions
- Court choices use identical landmark names but different coordinates. Verify the new request query and list rebuild, not a name change.
- Synthetic frame is 1280x720; native court coordinates map with `synthetic.court_to_px`. Clicking uses image natural-width scaling, so compare saved pixels with scaled mouse coordinates.
- Select one end's lane-baseline near/far and FT-line near/far explicitly, skipping full-court landmarks. Save with four; Undo and reject three.
- Check completed rim-box Undo as well as ordinary point Undo.
- Compare displayed stats against the actual generated boxscore.json.
- Clip labels show event timestamps; the seek target is `start_s`, which includes a lead-in. Observe `seeking`, `seeked`, `playing`, and advancing `currentTime` with read-only listeners while clicking rows in the UI.
- Compare duration to frame_count/fps; confirm it is a finite number on screen.
