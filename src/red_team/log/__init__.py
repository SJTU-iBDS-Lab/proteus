"""Red-team log module — DTO schemas, path conventions, and writers.

This is the sole owner of ``outputs/evolution/<run_id>/...`` layout and
the three JSONL / JSON log formats that the evolution driver, attack
runner, and agent loop emit. Wave 2 moved the attack_log writer / search
here from ``red_team.agent.attack_log``. Wave 3 added the campaign_log
and summary writers (previously inlined in ``red_team.evolution.driver``).
"""
from __future__ import annotations

from red_team.log import campaign_log, summary
from red_team.log.attack_log import (
    AttackLogEntry,
    append,
    read_all,
    row_to_document,
    search,
)
from red_team.log.paths import (
    attack_log_path,
    campaign_log_path,
    mutation_dir,
    run_root,
    summary_path,
)
from red_team.log.schema import (
    AttackLogRow,
    CampaignLogRow,
    SummaryRow,
)

__all__ = [
    # schema DTOs
    "CampaignLogRow",
    "SummaryRow",
    "AttackLogRow",
    "AttackLogEntry",
    # path helpers
    "run_root",
    "campaign_log_path",
    "attack_log_path",
    "summary_path",
    "mutation_dir",
    # attack_log writer / reader / search
    "append",
    "read_all",
    "row_to_document",
    "search",
    # W3 writers (campaign_log.jsonl + summary.json)
    "campaign_log",
    "summary",
]
