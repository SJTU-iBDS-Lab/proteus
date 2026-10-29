"""A1 / A2 / A3 — ``disable_tool='<retrieval-tool-name>'``.

These flags are honoured by the agent runtime, not the driver.  The
driver's job is simply to forward ``ablation`` verbatim to
``agent_run``; this module documents the contract so the package's
public surface advertises all nine knobs in one place.

No runtime hook is registered — the flag never reaches a branch in the
runner.  Regression coverage lives in
``tests/integration/test_ablation_flags.py`` (A1/A2/A3).
"""
from __future__ import annotations

RECOGNISED_TOOLS = ("read_attack_log", "read_skill", "read_repo_context")
