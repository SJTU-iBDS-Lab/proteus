"""LLM-powered code capability extraction and risk tag refinement.

SkillProbe Phase 2 — Code Extractor with I/O Binding Analysis.
Uses LLM to analyze actual code behavior and assign precise risk tags,
replacing the noisy keyword-based approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.models import SkillArtifact

from ..llm_client import chat_json, is_available

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a security auditor analyzing an agent skill's code for risk assessment.

Given the skill's documentation (SKILL.md) and code files, perform I/O Binding Analysis:

1. Identify what the code ACTUALLY does (not what documentation says).
2. Assign precise source risk tags and sink sensitivity tags.

## Source Tags (what data the skill can ACCESS):
- "credentials": Code reads OTHER users' secrets (SSH keys, passwords, tokens from ~/.ssh, ~/.aws, /etc/passwd). NOT for reading its own API key from env/config.
- "local_files": Code reads ARBITRARY user files outside its own directory tree. NOT for reading its own config/data files.
- "unsanitized_external_string": Code receives external input and passes it to dangerous sinks without validation
- "executable_code_fragments": Code generates or receives executable code (eval, exec, code generation)
- "manipulative_semantics": Code can manipulate agent reasoning or override safety instructions

## Sink Tags (what the skill can DO with data):
- "network_egress": Code sends USER DATA or SENSITIVE DATA to external servers. NOT for calling its own API/service with its own API key.
- "filesystem_write": Code writes to locations OUTSIDE its own directory tree (e.g., ~/.bashrc, /etc/, other users' files)
- "shell_execution": Code executes shell commands or subprocesses (subprocess, os.system, exec)
- "llm_prompt_injection": Code constructs prompts that could override system instructions
- "sensitive_action_parameter": Code passes user-controlled data to dangerous system calls
- "persistent_state": Code modifies system-wide config, crontabs, or startup scripts
- "sensitive_action_trigger": Code triggers irreversible actions (delete, purchase, deploy)

## CRITICAL Rules:
- MOST skills legitimately read their own API keys from env vars and call their own API endpoints. This is NORMAL, not risky. Do NOT tag these as "credentials" or "network_egress".
- "credentials" means reading OTHER people's secrets, not the skill's own config.
- "network_egress" means sending user/sensitive data OUT, not calling the skill's own service.
- A typical skill pattern (read own API key → call own API → return result) should have ZERO source and ZERO sink tags.
- Only tag capabilities that represent ACTUAL security risk to the user.
- When in doubt, do NOT tag. Fewer precise tags >> many noisy tags.

Output strict JSON:
{
  "source_tags": ["tag1", "tag2"],
  "sink_tags": ["tag1"],
  "capabilities": [{"resource": "...", "access": "read/write/execute"}],
  "reasoning": "brief explanation"
}"""


@dataclass(slots=True)
class LLMExtractionResult:
    source_tags: list[str] = field(default_factory=list)
    sink_tags: list[str] = field(default_factory=list)
    capabilities: list[dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    used_llm: bool = False


def extract_capabilities(skill: SkillArtifact, *, layer: str = "alignment") -> LLMExtractionResult:
    """Use LLM to extract precise capabilities from skill code."""
    if not is_available(layer):
        return LLMExtractionResult()

    # Build context: SKILL.md + code files (truncated)
    parts = [f"## SKILL.md\n{skill.skill_doc[:3000]}"]
    code_budget = 8000
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
        logger.warning("LLM code extraction failed for %s: %s", skill.skill_id, exc)
        return LLMExtractionResult()

    return LLMExtractionResult(
        source_tags=_str_list(result.get("source_tags", [])),
        sink_tags=_str_list(result.get("sink_tags", [])),
        capabilities=list(result.get("capabilities", [])),
        reasoning=str(result.get("reasoning", "")),
        used_llm=True,
    )


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def refine_skill_tags(
    skill: SkillArtifact, *, layer: str = "alignment"
) -> tuple[list[str], list[str], LLMExtractionResult]:
    """Refine a skill's source/sink tags using LLM analysis.

    Returns (refined_source_tags, refined_sink_tags, llm_result).
    Falls back to original keyword tags if LLM unavailable.
    """
    llm_result = extract_capabilities(skill, layer=layer)
    if not llm_result.used_llm:
        return skill.source_tags, skill.sink_tags, llm_result

    # LLM tags replace keyword tags entirely when available
    return llm_result.source_tags, llm_result.sink_tags, llm_result
