"""Human review: corrections layered over the AI events, never replacing them.

The AI writes ``events.json``; a reviewer's decisions go to ``edits.json`` as an ordered
log. The box score is always recomputed from AI events + edits, so a correction can be
undone, the original detection is never lost, and every reviewed event becomes a labelled
training example (see :mod:`hoopvision.ml.dataset`).

Edit ops:
  ``confirm``  {event_id}                    the AI got it right
  ``modify``   {event_id, changes: {...}}    e.g. a different player, kind, or value
  ``delete``   {event_id}                    a false positive
  ``add``      {event: {...}}                a stat the AI missed entirely
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .types import Event, dump_json, load_json

OPS = ("confirm", "modify", "delete", "add")
EDITABLE = {"kind", "player", "track_id", "value", "frame", "time_s", "detail"}

# Stats a single camera gets wrong often enough that every one is shown to a reviewer.
ALWAYS_REVIEW = {"foul", "block", "assist", "steal", "turnover"}


def ensure_ids(events: list[Event]) -> list[Event]:
    """Stable ids so edits can point at an event across recomputes."""
    seen: dict[str, int] = {}
    out = []
    for e in events:
        if e.event_id:
            out.append(e)
            continue
        base = f"{e.kind}-{e.frame}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(replace(e, event_id=base if n == 0 else f"{base}-{n}"))
    return out


def load_edits(run: Path) -> list[dict[str, Any]]:
    path = Path(run) / "edits.json"
    return load_json(path) if path.exists() else []


def record_edit(run: Path, edit: dict[str, Any], reviewer: str | None = None) -> dict[str, Any]:
    op = edit.get("op")
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    if op in ("confirm", "modify", "delete") and not edit.get("event_id"):
        raise ValueError(f"{op} needs event_id")
    if op == "modify":
        bad = set(edit.get("changes", {})) - EDITABLE
        if bad:
            raise ValueError(f"cannot modify {sorted(bad)}")
    if op == "add":
        ev = dict(edit.get("event") or {})
        if "kind" not in ev or "frame" not in ev:
            raise ValueError("add needs event.kind and event.frame")
        ev.setdefault("event_id", f"manual-{uuid.uuid4().hex[:8]}")
        edit = {**edit, "event": ev}
    entry = {**edit, "at": time.time(), "reviewer": reviewer}
    edits = load_edits(run)
    edits.append(entry)
    dump_json(edits, Path(run) / "edits.json")
    return entry


def apply_edits(events: list[Event], edits: list[dict[str, Any]], fps: float) -> list[Event]:
    """Final event list: AI events with every edit replayed in order."""
    by_id = {e.event_id: e for e in ensure_ids(events)}
    order = list(by_id)
    for ed in edits:
        op = ed["op"]
        if op == "add":
            raw = dict(ed["event"])
            raw.setdefault("time_s", raw["frame"] / fps)
            raw.setdefault("player", None)
            raw["confidence"] = 1.0
            raw["detail"] = {**raw.get("detail", {}), "source": "human", "reviewed": True}
            ev = Event(**{k: v for k, v in raw.items() if k in Event.__dataclass_fields__})
            by_id[ev.event_id] = ev
            order.append(ev.event_id)
            continue
        ev = by_id.get(ed["event_id"])
        if ev is None:
            continue
        if op == "delete":
            del by_id[ed["event_id"]]
        elif op == "confirm":
            by_id[ev.event_id] = replace(
                ev, confidence=1.0, detail={**ev.detail, "reviewed": True}
            )
        elif op == "modify":
            changes = dict(ed.get("changes", {}))
            if "frame" in changes and "time_s" not in changes:
                changes["time_s"] = changes["frame"] / fps
            detail = {**ev.detail, **changes.pop("detail", {}), "reviewed": True,
                      "ai_original": {k: getattr(ev, k) for k in ("kind", "player", "value")}}
            by_id[ev.event_id] = replace(ev, **changes, confidence=1.0, detail=detail)
    return sorted((by_id[i] for i in order if i in by_id), key=lambda e: e.frame)


def review_queue(events: list[Event], threshold: float) -> list[dict[str, Any]]:
    """Events a reviewer should look at, lowest confidence first."""
    out = []
    for e in ensure_ids(events):
        if e.kind == "possession" or e.detail.get("reviewed"):
            continue
        if e.kind in ALWAYS_REVIEW or e.confidence < threshold or e.player is None:
            out.append(asdict(e))
    return sorted(out, key=lambda e: e["confidence"])
