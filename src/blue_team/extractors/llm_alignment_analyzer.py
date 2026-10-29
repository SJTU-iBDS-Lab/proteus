"""LLM-powered 4-class Alignment Matrix analysis.

SkillProbe Phase 2 — Semantic Consistency Detector.
Compares what SKILL.md declares vs what code actually implements,
classifying into: MATCH, OVER_DECLARATION, UNDER_DECLARATION, MIXED.

UNDER_DECLARATION (shadow functions) is the most dangerous:
code does things the documentation doesn't mention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.enums import SemanticAlignmentType
from core.models import SkillArtifact

from ..llm_client import chat_json, is_available

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a security auditor comparing a skill's DOCUMENTATION (SKILL.md) against \
its actual CODE to detect hidden or undeclared capabilities.

## 4-Class Alignment Matrix:

1. **MATCH** — Documentation accurately describes what the code does. \
No significant gaps in either direction.

2. **OVER_DECLARATION** — Documentation claims capabilities the code doesn't \
actually implement. (Docs promise more than code delivers — low risk, possibly \
incomplete or aspirational docs.)

3. **UNDER_DECLARATION** — Code implements capabilities NOT mentioned in docs. \
**This is the most dangerous class** — "shadow functions" that users don't know \
about. Examples:
   - Docs say "read notes" but code also sends data to an external server
   - Docs say "format text" but code also executes shell commands
   - Docs say "search files" but code also reads SSH keys or credentials
   - Docs say "generate reports" but code also modifies system configuration

4. **MIXED** — Both over-declaration and under-declaration exist.

## Analysis Steps:
1. Read SKILL.md carefully — list what capabilities it CLAIMS
2. Read the code — list what the code ACTUALLY does
3. Compare the two lists
4. Classify and provide evidence

## CRITICAL:
- Focus on SECURITY-RELEVANT undeclared capabilities (network, filesystem, shell, \
credentials, data exfiltration)
- Ignore minor differences (e.g., docs mention "fast" but code is slow)
- A skill that does EXACTLY what its docs say is MATCH, even if what it does is risky
- UNDER_DECLARATION requires the code to do something the docs DON'T mention

Output strict JSON:
{
  "alignment": "match" | "over_declaration" | "under_declaration" | "mixed",
  "declared_capabilities": ["what docs say the skill does"],
  "implemented_capabilities": ["what the code actually does"],
  "undeclared_capabilities": ["things code does but docs don't mention"],
  "over_declared_capabilities": ["things docs claim but code doesn't do"],
  "shadow_functions": ["security-relevant undeclared capabilities"],
  "reasoning": "brief explanation of the classification"
}"""


@dataclass(slots=True)
class LLMAlignmentResult:
    alignment: SemanticAlignmentType = SemanticAlignmentType.MATCH
    declared_capabilities: list[str] = field(default_factory=list)
    implemented_capabilities: list[str] = field(default_factory=list)
    undeclared_capabilities: list[str] = field(default_factory=list)
    over_declared_capabilities: list[str] = field(default_factory=list)
    shadow_functions: list[str] = field(default_factory=list)
    reasoning: str = ""
    used_llm: bool = False


_ALIGNMENT_MAP: dict[str, SemanticAlignmentType] = {
    "match": SemanticAlignmentType.MATCH,
    "over_declaration": SemanticAlignmentType.OVER_DECLARATION,
    "under_declaration": SemanticAlignmentType.UNDER_DECLARATION,
    "mixed": SemanticAlignmentType.MIXED,
}


def analyze_alignment(
    skill: SkillArtifact, *, layer: str = "alignment"
) -> LLMAlignmentResult:
    """Use LLM to classify the semantic alignment between docs and code."""
    if not is_available(layer):
        return LLMAlignmentResult()

    # Build context
    parts = [f"## SKILL.md\n{skill.skill_doc[:3000]}"]
    code_budget = 6000
    for path, content in skill.code_files.items():
        if code_budget <= 0:
            break
        snippet = content[:code_budget]
        parts.append(f"## {path}\n```\n{snippet}\n```")
        code_budget -= len(snippet)

    user_msg = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        result = chat_json(messages, layer=layer, temperature=0.1, timeout=90.0)
    except Exception as exc:
        logger.warning("LLM alignment analysis failed for %s: %s", skill.skill_id, exc)
        return LLMAlignmentResult()

    raw_alignment = str(result.get("alignment", "match")).lower().strip()
    alignment = _ALIGNMENT_MAP.get(raw_alignment, SemanticAlignmentType.MATCH)

    return LLMAlignmentResult(
        alignment=alignment,
        declared_capabilities=_str_list(result.get("declared_capabilities", [])),
        implemented_capabilities=_str_list(result.get("implemented_capabilities", [])),
        undeclared_capabilities=_str_list(result.get("undeclared_capabilities", [])),
        over_declared_capabilities=_str_list(result.get("over_declared_capabilities", [])),
        shadow_functions=_str_list(result.get("shadow_functions", [])),
        reasoning=str(result.get("reasoning", "")),
        used_llm=True,
    )


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
