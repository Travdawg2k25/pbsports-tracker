"""Learning layer: training data from reviewed games, the event model, and evaluation.

The geometric rules in :mod:`hoopvision.events` propose candidate events; the models here
learn from human-reviewed games to re-score and re-label those candidates. Every game a
reviewer corrects makes the next game more accurate.
"""
