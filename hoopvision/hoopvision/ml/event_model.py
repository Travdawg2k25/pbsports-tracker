"""Temporal event classifier: re-scores and re-labels rule candidates.

A small 1-D conv + GRU over the feature window from :mod:`.features`. It is trained on
reviewed games and, at inference, replaces each candidate's rule confidence with the
model's probability for its kind (and relabels it when the model is confident the rules
picked the wrong kind, e.g. a "miss" that was really a pass). Candidates the model calls
``none`` are dropped. Without a trained model file the pipeline uses the rules alone.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..types import Event
from .features import FEATURES, KINDS, window

log = logging.getLogger("hoopvision.ml")


def _net(n_classes: int):
    import torch.nn as nn

    class EventNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(FEATURES, 64, 5, padding=2), nn.ReLU(),
                nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            )
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):  # x: B x T x F
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.gru(h)
            return self.head(out.mean(dim=1))

    return EventNet()


def train(dataset: Path, out: Path, epochs: int = 40, lr: float = 1e-3,
          device: str = "cpu", val_frac: float = 0.2, seed: int = 0) -> dict:
    import torch

    data = np.load(dataset)
    X, y = torch.tensor(data["X"]), torch.tensor(data["y"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    n_val = int(len(y) * val_frac)
    val, tr = perm[:n_val], perm[n_val:]
    model = _net(len(KINDS)).to(device)
    # Inverse-frequency weights: fouls and blocks are rare next to shots.
    counts = torch.bincount(y[tr], minlength=len(KINDS)).float().clamp(min=1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for idx in tr[torch.randperm(len(tr), generator=g)].split(64):
            opt.zero_grad()
            loss = loss_fn(model(X[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            log.info("epoch %d loss %.3f", ep + 1, float(loss))
    metrics = {"train_examples": int(len(tr)), "val_examples": int(n_val)}
    if n_val:
        model.eval()
        with torch.no_grad():
            pred = model(X[val].to(device)).argmax(1).cpu()
        metrics["val_accuracy"] = round(float((pred == y[val]).float().mean()), 3)
        metrics["val_per_kind"] = {
            KINDS[k]: round(float((pred[y[val] == k] == k).float().mean()), 3)
            for k in y[val].unique().tolist()
        }
    torch.save({"state": model.state_dict(), "kinds": KINDS, "metrics": metrics}, out)
    return metrics


class EventRescorer:
    def __init__(self, path: str | Path, device: str = "cpu", relabel_prob: float = 0.7):
        import torch

        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.kinds = ckpt["kinds"]
        self.model = _net(len(self.kinds)).to(device).eval()
        self.model.load_state_dict(ckpt["state"])
        self.device = device
        self.relabel_prob = relabel_prob

    def rescore(self, events: list[Event], ball, players, identities, calib, fps,
                poses=None) -> list[Event]:
        import torch

        cands = [e for e in events if e.kind != "possession" and e.kind in self.kinds]
        if not cands:
            return events
        X = np.stack([window(e.frame, e.track_id, ball, players, identities, calib, fps, poses)
                      for e in cands])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(X).to(self.device)), 1).cpu().numpy()
        new: dict[str, Event | None] = {}
        for ev, p in zip(cands, probs, strict=True):
            best = int(p.argmax())
            kind = ev.kind
            if p[best] >= self.relabel_prob and self.kinds[best] != kind:
                kind = self.kinds[best]
            if kind == "none":
                new[ev.event_id] = None
                continue
            detail = {**ev.detail, "rule_confidence": ev.confidence, "model": True}
            if kind != ev.kind:
                detail["rule_kind"] = ev.kind
            new[ev.event_id] = replace(
                ev, kind=kind, confidence=round(float(p[self.kinds.index(kind)]), 3),
                detail=detail,
            )
        out = []
        for e in events:
            if e.event_id in new:
                if new[e.event_id] is not None:
                    out.append(new[e.event_id])
            else:
                out.append(e)
        return out
