# hoopvision

Basketball video analysis for a **single static 4K camera**: player and ball tracking,
jersey-number OCR, event detection, and a per-player box score you can browse in a
small web UI (pick a player, get their stats and the clips behind every stat).

Self-contained and installable on its own (`pip install -e ./hoopvision`); it imports
nothing from the modules at the repository root and changes none of them. It covers the
same ground as `tracker.py` / `basketball_events.py` / `stats_engine.py` but offline and
stage-by-stage rather than streaming, with a court homography, camera-motion
compensation and a regression suite. Consolidating the two — most usefully by moving
camera-motion compensation and the court homography into the live pipeline — is a
follow-up, not part of this drop.

## Pipeline

```
video -> detect (players + ball) -> track -> teams (kit colour) -> jersey OCR (track-level voting)
      -> court homography -> possession / shot / rebound events -> box score + clip index
      -> FastAPI + web UI
```

Each stage writes a JSON artifact under the run directory, so you can re-run a later
stage without redoing detection (which is the expensive part).

| Stage | Artifact | Module |
|---|---|---|
| camera motion | `motion.json` | `hoopvision/stabilize.py` |
| detection + tracking | `tracks.json` | `hoopvision/detect.py`, `hoopvision/track.py` |
| team assignment | `teams.json` | `hoopvision/teams.py` |
| jersey OCR | `identities.json` | `hoopvision/jersey.py` |
| events | `events.json` | `hoopvision/events.py` |
| box score | `boxscore.json` | `hoopvision/boxscore.py` |

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

GPU is strongly recommended. On CPU, 4K detection runs at roughly 0.2–1 fps; on a
modern GPU the same pipeline runs faster than real time at 1080p-downscaled input with
a high-resolution tile pass for the ball.

## Use

```bash
# 1. calibrate: mark the court corners and the two rims once per camera position
hoopvision calibrate game.mp4 --out runs/game/calibration.json

# 2. analyze
hoopvision analyze game.mp4 --calibration runs/game/calibration.json --out runs/game

# 3. browse
hoopvision serve runs/game            # http://localhost:8000
```

`runs/game/boxscore.json` holds the full box score; the UI lets you click a player and
see every counted event with a jump-to-timestamp clip.

### Footage that is not locked off

A calibration is only valid for the frame it was marked on. Before tracking, every
frame is matched back to that reference frame, so a camera that pans, drifts or zooms
still lands in the same court coordinates. Features are taken from the wood floor
only — walls, bleachers and the ceiling truss are a different plane and a homography
fitted to them shears the court. Pass `--no-stabilize` for a genuinely locked-off
camera.

When the operator pans, no single frame shows the whole court, so
`hoopvision.stabilize.mosaic()` stitches the clip into one reference-space image to
mark landmarks on.

## Accuracy notes

Everything downstream of tracking is heuristic and needs to be validated against a
hand-scored possession before you trust it:

- **Jersey OCR** only fires on frames where the number faces the camera; identity is
  resolved by majority vote over a whole track, which is why tracking continuity matters
  more than per-frame OCR accuracy.
- **Made vs missed** is decided from the ball trajectory relative to the calibrated rim
  box (entered from above, exited below, monotonic descent). Tip-ins and goaltends are
  unreliable.
- **Assists and turnovers** are inferred from possession changes and pass chains and are
  the least reliable numbers in the box score. They are reported with a confidence field.

### What has actually been validated

- The full pipeline reproduces a known box score on the synthetic game in
  `hoopvision/synthetic.py`, including with a panning camera (`tests/`).
- On a 27 s 1080p screen recording of a tightly panned broadcast, tracking (73 player
  tracks, 195 ball frames) and jersey OCR (13 tracks read, e.g. #5, #21, #25, #30) work,
  and motion compensation stitches a coherent court. **Events and the box score do not**:
  the pan never shows enough of the floor at once to mark court landmarks accurately, and
  the ball track is too fragmented at that zoom. The event layer is still only validated
  synthetically.
