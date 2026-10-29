"""One-run campaign lifecycle — state machine + I/O + ablations + seams.

Wave 5 extracted the Algorithm 1 driver out of ``red_team.evolution``
because four concerns were tangled in a 700+ line module:

* **State machine** — the per-round loop, bypass/lethal gating, and
  break condition.  Lives in :mod:`red_team.campaign.runner`.
* **I/O** — log row assembly, ``log.jsonl`` / ``summary.json`` writers,
  mutation-md persistence, chain snapshots.  Lives in
  :mod:`red_team.campaign.io`.
* **Seams** — default bindings + dispatch for the five heavy callables
  (agent / sandbox / blue team / oracle / horizontal expansion).  Lives
  in :mod:`red_team.campaign.seams`.
* **Ablations** — one module per knob (A1…A9) under
  :mod:`red_team.campaign.ablations`, composed via phase-specific
  registries so the runner never contains ``if ablation == "X"``.

The public API is :func:`red_team.campaign.runner.run_evolution`.  The
legacy import path ``from red_team.evolution.driver import run_evolution``
continues to work via a shim re-export.
"""
from __future__ import annotations

from red_team.campaign.runner import run_evolution

__all__ = ["run_evolution"]
