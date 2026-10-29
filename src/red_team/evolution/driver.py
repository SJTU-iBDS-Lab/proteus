"""Backward-compat shim — real implementation lives in :mod:`red_team.campaign`.

Wave 5 moved the 773-line Algorithm-1 driver into a four-part package:

* :mod:`red_team.campaign.runner`    — state machine
* :mod:`red_team.campaign.io`        — log rows + summary writers
* :mod:`red_team.campaign.seams`     — default dependency bindings
* :mod:`red_team.campaign.ablations` — one knob per module (A1…A9)

This module re-exports :func:`run_evolution` so existing imports keep
working.  New code should import directly from :mod:`red_team.campaign`.
"""
from __future__ import annotations

from red_team.campaign.runner import run_evolution

__all__ = ["run_evolution"]
