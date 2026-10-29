"""External attack baselines evaluated under a fixed candidate budget.

- :class:`SkillInjectBo5Line` — non-adaptive placement baseline
  (SKILL-INJECT Bo5-Line mechanism); no LLM, no observations.
- :class:`SkillJectTraceOnly` — trace-driven refinement (SkillJect
  mechanism); observes victim execution traces only, never defender state.
- :class:`VictimTrace` — the victim-only observation contract.
"""
from red_team.external_baselines.base import (
    CandidateGenerator,
    CandidateRecord,
    sha256_text,
)
from red_team.external_baselines.campaign import (
    SkillInjectCampaign,
    SkillJectCampaign,
)
from red_team.external_baselines.skill_inject import SkillInjectBo5Line
from red_team.external_baselines.skillject import SkillJectTraceOnly
from red_team.external_baselines.trace import VictimTrace

__all__ = [
    "CandidateGenerator",
    "CandidateRecord",
    "SkillInjectBo5Line",
    "SkillInjectCampaign",
    "SkillJectCampaign",
    "SkillJectTraceOnly",
    "VictimTrace",
    "sha256_text",
]
