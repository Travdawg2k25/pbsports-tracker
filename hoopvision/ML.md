# HoopVision ML environment

How the box score gets more accurate over time. The rules in `events.py` propose events;
human review corrects them; the corrections train models that replace the rules one
signal at a time.

```
phone video ─► motion ─► track ─► teams ─► jersey ─► pose ─► audio ─► events ─► boxscore
                                                    (keypoints) (whistles)  │ rules + model
                                                                            ▼
                         review UI (/review) ─► edits.json ─► box score recomputed
                                                     │
                                                     ▼
                     build-dataset ─► train-events ─► models/events.pt (HV_EVENT_MODEL)
```

## What each stat comes from

| Stat | Signal | Review |
|---|---|---|
| PTS, FGM/FGA, 3PM/3PA | ball through the calibrated rim; shooter's court position; shooting pose | low-confidence only |
| FTM/FTA | shot from inside the FT circle after a dead ball | low-confidence only |
| OREB/DREB | first possession after a miss | low-confidence only |
| AST | teammate possession just before a make | always |
| STL/TO | possession changes team without a shot | always |
| BLK | defender between shooter and rim + raised arm (pose) | always |
| PF | referee whistle (audio) during live play; fouler ranked by distance to the ball handler | always (pick from candidates) |

Every event carries `confidence` and a stable `event_id`. The box score has a `review`
block: `ai_draft` → `in_review` → `verified`.

## Review

```sh
hoopvision serve runs/game          # open /review
```

Confirm, change the player/stat, or delete. Edits are an append-only log in
`edits.json`; `events.json` (the AI output) is never modified, so the original detection
stays available for training. API: `GET /api/review`, `POST /api/events/{id}/edit`
(`{"op": "confirm"|"modify"|"delete", "changes": {...}}`), `POST /api/events`
(`{"event": {"kind": "foul", "frame": 1234, "player": "away:4"}}`).

The GPU worker uploads every run's artifacts to `s3://$PB_BUCKET/results/<job_id>/run/`
so a game can be pulled down and reviewed.

## Training on the AWS GPU server

```sh
pip install -e "./hoopvision[dev,worker]"     # with the CUDA torch build

# 1. Detection: fine-tune the ball/rim model on the Roboflow set (+ frames from your gyms)
hoopvision export-frames game.mp4 --out frames/game1 --every 2   # label in Roboflow
hoopvision train-detector Basketball-and-rim-1/data.yaml --model yolov8s.pt --name ball_rim
#    -> models/train/ball_rim/weights/best.pt ; set HV_BALL_MODEL / HV_RIM_MODEL to it
#    Same command fine-tunes players (yolov8s.pt) or pose (yolov8s-pose.pt) from a
#    pose-format data.yaml.

# 2. Events: once games are reviewed
aws s3 sync s3://$PB_BUCKET/results/<job_id>/run runs/<job_id>   # plus edits.json
hoopvision build-dataset runs/* --out models/events.npz
hoopvision train-events models/events.npz --out models/events.pt --device cuda:0
#    set HV_EVENT_MODEL=models/events.pt on the worker

# 3. Measure: against a hand-kept box score (team,jersey,pts,fgm,fga,...,pf)
hoopvision evaluate runs/<job_id> truth.csv
```

Split train/validation **by game**, not by frame: frames from one game are near
duplicates, so a per-frame split overstates accuracy. Keep 2–3 hand-scored games aside
that never go into training, and run `evaluate` on them after every model change.

## Worker settings

| Env | Default | |
|---|---|---|
| `HV_POSE` | `1` | pose keypoints near candidate events |
| `HV_POSE_MODEL` | `yolov8s-pose.pt` | downloaded on first use |
| `HV_AUDIO` | `1` | whistle detection for fouls |
| `HV_EVENT_MODEL` | unset | trained event model; rules only when unset |

## Honest limits

- Fouls from one phone are candidates, not facts: the whistle says *that* a foul was
  called, the geometry only ranks *who*. Expect the reviewer to pick the fouler.
- The event model needs reviewed games before it helps; with few examples keep it off.
- Accuracy targets have to be measured with `evaluate` on your footage; none of the
  numbers here are benchmarks.
