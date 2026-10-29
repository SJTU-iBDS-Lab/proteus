"""Runtime / sandbox feedback synthesis used by the campaign loop.

Modules here turn sandbox observations (HoneypotResult, target reply)
into :class:`PaperFinding` rows the mutator can read.  Kept separate
from :mod:`red_team.campaign.runner` so the runner stays focused on
orchestration; cross-round gray-box signals live next to the
heuristics that produce them.
"""
