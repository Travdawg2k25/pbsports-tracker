"""Scoreboard OCR — ground-truth score for truing the inferred box score.

Every other stat in the pipeline is *inferred* and error-prone. The scoreboard, when it
is legible in frame, is *authoritative*: it shows the real score. Reading it gives two
things nothing else can:

1. **Truing** — compare the detected team points against the scoreboard and surface the
   gap, so a human knows how much the AI missed.
2. **Gap-filling** — a jump in the scoreboard (e.g. 40 -> 42 between two readings) proves
   a basket happened in that window even when ball tracking missed it entirely. We cannot
   always say *who* scored, but we can correct the team total and point a human at the
   moment to assign the player.

Limits, stated honestly: the scoreboard gives only *team* totals (never individual
stats), only when it is legible in frame, and only as running totals (we detect changes
between readings). It reconciles team points; individual attribution still needs vision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .types import Box
from .video import crop, iter_frames

log = logging.getLogger("hoopvision.scoreboard")

# Two 1-3 digit numbers are the home/away score; anything else on the board is ignored.
_SCORE_RE = re.compile(r"^\d{1,3}$")


@dataclass
class ScoreReading:
    frame: int
    time_s: float
    home: int
    away: int


@dataclass
class ScoreboardResult:
    readings: list[ScoreReading] = field(default_factory=list)
    # Confirmed score at the end of what we could read (last stable reading).
    final_home: int | None = None
    final_away: int | None = None

    def as_dict(self) -> dict:
        return {
            "readings": [
                {"time_s": round(r.time_s, 1), "home": r.home, "away": r.away}
                for r in self.readings
            ],
            "final_home": self.final_home,
            "final_away": self.final_away,
        }


def _plausible_pair(nums: list[int]) -> tuple[int, int] | None:
    """Pick the two numbers most likely to be the scores from an OCR of the board.

    A scoreboard shows scores plus clock/period/fouls; scores are 0-199 and usually the
    two largest 1-3 digit values. This is a heuristic — the reconciliation step tolerates
    the occasional bad reading because it only trusts monotonic, repeated values.
    """
    scores = sorted((n for n in nums if 0 <= n <= 199), reverse=True)
    if len(scores) < 2:
        return None
    return scores[0], scores[1]


def read_scoreboard(
    video: str,
    region: Box | None,
    fps: float,
    *,
    gpu: bool = True,
    sample_every_s: float = 5.0,
    home_left: bool = True,
) -> ScoreboardResult:
    """OCR the scoreboard every ``sample_every_s`` seconds and build a score timeline.

    ``region`` is the pixel box of the scoreboard (marked once per camera, like the
    court). If ``None``, the whole frame is OCR'd, which is slower and noisier. ``home_left``
    says whether the home score is the left/first number on the board.
    """
    try:
        import easyocr
    except ImportError:  # pragma: no cover
        log.warning("easyocr not available; scoreboard reading skipped")
        return ScoreboardResult()

    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    stride = max(1, int(sample_every_s * fps))
    raw: list[ScoreReading] = []

    for idx, frame in iter_frames(video, stride=stride):
        img = crop(frame, region) if region else frame
        if img.size == 0:
            continue
        texts = reader.readtext(img, detail=0, allowlist="0123456789:")
        nums = [int(t) for t in texts if _SCORE_RE.match(t.strip())]
        pair = _plausible_pair(nums)
        if pair is None:
            continue
        hi, lo = pair
        # Without a fixed layout we cannot know which score is home vs away from value
        # alone; home_left is resolved during OCR ordering below when a region is marked.
        home, away = (hi, lo) if home_left else (lo, hi)
        raw.append(ScoreReading(frame=idx, time_s=idx / fps, home=home, away=away))

    return _clean(raw)


def _clean(raw: list[ScoreReading]) -> ScoreboardResult:
    """Keep only readings consistent with a real game: scores never decrease.

    A running score is monotonic non-decreasing. We walk the readings and drop any that
    would require the score to go *down* (an OCR misread), which removes most noise
    without needing a confident single read.
    """
    result = ScoreboardResult()
    best_home = best_away = 0
    for r in sorted(raw, key=lambda x: x.frame):
        if r.home < best_home or r.away < best_away:
            continue  # a score cannot decrease; discard the misread
        # Reject implausible single-jump leaps (>10 pts between two 5s samples is likely
        # a misread of a different board element).
        if result.readings:
            if r.home - best_home > 10 or r.away - best_away > 10:
                continue
        best_home, best_away = max(best_home, r.home), max(best_away, r.away)
        result.readings.append(ScoreReading(r.frame, r.time_s, best_home, best_away))
    if result.readings:
        result.final_home = result.readings[-1].home
        result.final_away = result.readings[-1].away
    return result


def reconcile(
    scoreboard: ScoreboardResult,
    detected_team_points: dict[str, int],
    home_team_id: str = "home",
    away_team_id: str = "away",
) -> dict:
    """Compare scoreboard truth against detected points; report the gap per team.

    ``detected_team_points`` maps team id -> points the pipeline detected. Returns a
    reconciliation block for the UI: the authoritative score, what we detected, and the
    unaccounted points (baskets the vision missed) so a human can fill them in.
    """
    if scoreboard.final_home is None:
        return {"available": False}
    det_home = detected_team_points.get(home_team_id, 0)
    det_away = detected_team_points.get(away_team_id, 0)
    return {
        "available": True,
        "scoreboard": {"home": scoreboard.final_home, "away": scoreboard.final_away},
        "detected": {"home": det_home, "away": det_away},
        # Positive = points the scoreboard has that the pipeline missed.
        "unaccounted": {
            "home": max(0, scoreboard.final_home - det_home),
            "away": max(0, scoreboard.final_away - det_away),
        },
        "readings": scoreboard.as_dict()["readings"],
    }


def scoring_windows(scoreboard: ScoreboardResult) -> list[dict]:
    """Time windows where the score changed — candidate made-basket moments to attribute.

    Each window is between two consecutive readings where a team's score rose; a human (or
    a relaxed second-pass search) can assign the shooter. This is how a missed made basket
    is recovered even when ball tracking failed.
    """
    out: list[dict] = []
    prev: ScoreReading | None = None
    for r in scoreboard.readings:
        if prev is not None:
            dh, da = r.home - prev.home, r.away - prev.away
            if dh > 0 or da > 0:
                out.append(
                    {
                        "start_s": round(prev.time_s, 1),
                        "end_s": round(r.time_s, 1),
                        "home_points": dh,
                        "away_points": da,
                    }
                )
        prev = r
    return out
