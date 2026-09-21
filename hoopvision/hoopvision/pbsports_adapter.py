"""Translate a HoopVision run into the pbsports web app's ``stats.json`` shape.

The pbsports frontend reads a fixed box-score schema (``stats_engine.Player.to_dict`` /
``StatsEngine.get_summary``). HoopVision produces its own ``boxscore.json``. This module
maps one onto the other so HoopVision can back the existing upload/review/download flow
without touching the routes, database, or S3 layout.

Two product rules are enforced here:

* **AI drafts, parent finalises.** The parent-entered jersey and name (stored on the job)
  are stamped onto the selected player, overriding OCR. Everything else is a draft the
  parent can edit.
* **Honesty about the single-camera ceiling.** On the rim-only path (auto-rim, no court
  homography) points and FG% are reliable, but three-pointers and shot distance are not.
  Those are surfaced under ``best_effort`` so the UI can flag them instead of presenting
  guesses as fact.
"""

from __future__ import annotations

from typing import Any

# Stats HoopVision still cannot observe from a single camera. Present as 0, not fabricated.
_UNSUPPORTED = ("passes", "fouls")


def _pct(makes: int, attempts: int) -> float | None:
    return round(makes / attempts, 3) if attempts else None


def _track_id_from_key(player_key: str) -> int:
    """HoopVision keys players as ``"team:jersey"``; pbsports rows key by track_id.

    We hash the stable player key into a non-negative int so the same player maps to the
    same row across runs. The real jersey/name still ride on the row for display.
    """
    return abs(hash(player_key)) % (10**8)


def _player_line_to_pbsports(
    line: dict[str, Any],
    distance_reliable: bool,
    jersey_override: str | None = None,
    name_override: str | None = None,
) -> dict[str, Any]:
    fgm, fga = line.get("fgm", 0), line.get("fga", 0)
    tpm, tpa = line.get("tpm", 0), line.get("tpa", 0)
    points = line.get("points", 0)

    # An un-OCR'd track carries a synthetic "t<id>" jersey so its events survive; that is
    # a placeholder, not a real number, so display it as unknown unless the parent typed one.
    ocr_jersey = line.get("jersey")
    if ocr_jersey and str(ocr_jersey).startswith("t") and str(ocr_jersey)[1:].isdigit():
        ocr_jersey = None

    # Effective FG% counts a three as 1.5 baskets; true shooting adds FTs (none here).
    efg = round((fgm + 0.5 * tpm) / fga, 3) if fga else None
    ts = round(points / (2 * fga), 3) if fga else None

    row: dict[str, Any] = {
        "track_id": _track_id_from_key(line["player"]),
        "jersey_number": jersey_override or ocr_jersey,
        "team_id": line.get("team"),
        "name": name_override or line.get("name"),
        # Scoring — reliable on the rim-only path
        "points": points,
        "fg_makes": fgm,
        "fg_attempts": fga,
        "fg_pct": line.get("fg_pct", _pct(fgm, fga)),
        # Three-pointers — best-effort without a homography (see best_effort below)
        "fg3_makes": tpm,
        "fg3_attempts": tpa,
        "fg3_pct": line.get("tp_pct", _pct(tpm, tpa)),
        "efg_pct": efg,
        "ts_pct": ts,
        # Rebounding
        "offensive_rebounds": line.get("offensive_rebounds", 0),
        "defensive_rebounds": line.get("defensive_rebounds", 0),
        "total_rebounds": line.get("rebounds", 0),
        # Playmaking — best-effort (built on approximate possessions when rim-only)
        "assists": line.get("assists", 0),
        "turnovers": line.get("turnovers", 0),
        "ast_to": (
            round(line.get("assists", 0) / line.get("turnovers", 1), 2)
            if line.get("turnovers")
            else None
        ),
        # Free throws (1 pt each) — detected from FT-line shots during dead-ball play;
        # best-effort and only when a homography makes the line geometry reliable.
        "ft_makes": line.get("ftm", 0),
        "ft_attempts": line.get("fta", 0),
        "ft_pct": line.get("ft_pct"),
        # Blocks — low-confidence single-camera candidate, for the parent to verify.
        "blocks": line.get("blocks", 0),
        # Defense / streaks / efficiency HoopVision does not measure — reported as 0/None
        "steals": line.get("steals", 0),
        "consecutive_makes": 0,
        "is_hot": False,
        "per_rating": None,
        "highlight_score": None,
        "zone_shooting": {},
        # Which numbers the parent should verify rather than trust.
        "best_effort": _best_effort_flags(distance_reliable),
    }
    for k in _UNSUPPORTED:
        row[k] = 0
    return row


def _best_effort_flags(distance_reliable: bool) -> dict[str, bool]:
    """Per-stat reliability. Points/FG% are always trustworthy; the rest depend on setup."""
    return {
        "points": False,
        "fg_pct": False,
        # Without court geometry we cannot tell a two from a three, or who passed to whom.
        "fg3": not distance_reliable,
        "assists": True,
        "turnovers": True,
        "steals": True,
        "rebounds": True,
        # FT detection needs the FT-line geometry (homography); blocks are a deliberately
        # low-confidence single-camera candidate. Both are always verify-me.
        "ft": True,
        "blocks": True,
    }


def to_pbsports_stats(
    boxscore: dict[str, Any],
    *,
    game_id: str,
    distance_reliable: bool,
    focus_player_key: str | None = None,
    jersey_number: str | None = None,
    player_name: str | None = None,
    duration_sec: float | None = None,
    total_frames: int | None = None,
) -> dict[str, Any]:
    """Build the pbsports ``stats.json`` dict from a HoopVision ``boxscore.json`` dict.

    ``focus_player_key`` is the HoopVision key ("team:jersey") of the player the parent
    selected; ``jersey_number``/``player_name`` are what the parent typed and override the
    OCR-derived values on that player only.
    """
    players_out: dict[str, dict[str, Any]] = {}
    teams_out: dict[str, dict[str, Any]] = {}

    for line in boxscore.get("players", []):
        is_focus = focus_player_key is not None and line.get("player") == focus_player_key
        row = _player_line_to_pbsports(
            line,
            distance_reliable,
            jersey_override=jersey_number if is_focus else None,
            name_override=player_name if is_focus else None,
        )
        players_out[str(row["track_id"])] = row

    for team_id, agg in boxscore.get("teams", {}).items():
        teams_out[str(team_id)] = {
            "team_id": team_id,
            "points": agg.get("points", 0),
            "fg_makes": agg.get("fgm", 0),
            "fg_attempts": agg.get("fga", 0),
            "rebounds": agg.get("rebounds", 0),
            "assists": agg.get("assists", 0),
        }

    video = boxscore.get("video", {})
    dur = duration_sec if duration_sec is not None else video.get("duration_s", 0.0)
    frames = total_frames if total_frames is not None else video.get("frame_count", 0)
    score = {tid: t["points"] for tid, t in teams_out.items()}

    return {
        "game_id": game_id,
        "duration_sec": round(dur, 2) if dur else 0.0,
        "total_frames": frames,
        "score": score,
        "teams": teams_out,
        "players": players_out,
        # Top-level honesty flag the UI can use to show a "verify these" banner.
        "distance_reliable": distance_reliable,
        "source": "hoopvision",
    }
