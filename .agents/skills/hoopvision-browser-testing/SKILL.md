---
name: hoopvision-browser-testing
description: Suggested additions to the existing local synthetic hoopvision browser-testing skill for the event review workflow.
---

# Suggested additions to existing skill

## Devin Secrets Needed
None for the local FastAPI applications.

## Review fixture with fouls
Write whistles.json after synthetic.generate and before stage_events. Choose a
time inside a detected possession and not within 1.5 seconds after a made basket.
For the standard fixture, start_s=10/30 and end_s=10/30+0.4 with confidence=0.9
and peak_hz=3100 yields a reviewable foul. A whistle early in a later possession
can legitimately be excluded because of the preceding made basket.

## Review workflow evidence
- Preserve events.json bytes and boxscore.json before the first edit.
- Use /review; click its "box score" link in a second tab to inspect moved stats.
- Check both old and new player totals, team totals, and the clip lists.
  A correct stat total alone does not establish that event clips follow edits.
- Preserve edits.json at the confirm/modify/delete checkpoint before confirming
  remaining rows. Final edit count includes those later confirmations.
- Observe seeking/seeked/playing events with read-only browser listeners. Row
  clicks target max(0,event.time_s-3), then play automatically.
- Check the final verified header and empty-panel message after a full refresh.

## Startup command syntax
The CLI accepts positional paths:
`hoopvision serve RUN --port 8766`
and `hoopvision calibrate VIDEO --out CALIBRATION_JSON --port 8765`.
