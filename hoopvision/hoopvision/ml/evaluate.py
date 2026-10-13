"""Accuracy against a hand-kept box score — the number that says whether a change helped.

Truth is a CSV with one row per player: ``team,jersey,pts,fgm,fga,tpm,tpa,ftm,fta,oreb,
dreb,ast,stl,blk,tov,pf`` (any subset of stat columns). Players are matched on team +
jersey. Reports, per stat, the mean absolute error per player, the share of players
exactly right, and the team-total error.
"""

from __future__ import annotations

import csv
from pathlib import Path

STAT_MAP = {
    "pts": "points", "fgm": "fgm", "fga": "fga", "tpm": "tpm", "tpa": "tpa", "ftm": "ftm",
    "fta": "fta", "oreb": "offensive_rebounds", "dreb": "defensive_rebounds",
    "reb": "rebounds", "ast": "assists", "stl": "steals", "blk": "blocks",
    "tov": "turnovers", "pf": "fouls",
}


def load_truth(path: str | Path) -> dict[str, dict[str, int]]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = f"{row['team'].strip()}:{row['jersey'].strip()}"
            out[key] = {k: int(v) for k, v in row.items() if k in STAT_MAP and v.strip()}
    return out


def evaluate(boxscore: dict, truth: dict[str, dict[str, int]]) -> dict:
    got = {p["player"]: p for p in boxscore.get("players", [])}
    stats = sorted({s for t in truth.values() for s in t})
    report: dict[str, dict] = {}
    for s in stats:
        field = STAT_MAP[s]
        errs, exact, tt, tp = [], 0, {}, {}
        for key, line in truth.items():
            if s not in line:
                continue
            pred = got.get(key, {}).get(field, 0)
            errs.append(abs(pred - line[s]))
            exact += pred == line[s]
            team = key.split(":")[0]
            tt[team] = tt.get(team, 0) + line[s]
            tp[team] = tp.get(team, 0) + pred
        report[s] = {
            "mae_per_player": round(sum(errs) / len(errs), 2) if errs else None,
            "exact_pct": round(exact / len(errs), 3) if errs else None,
            "team_error": {t: tp[t] - tt[t] for t in tt},
        }
    matched = sum(1 for k in truth if k in got)
    return {"players_in_truth": len(truth), "players_matched": matched, "stats": report}
