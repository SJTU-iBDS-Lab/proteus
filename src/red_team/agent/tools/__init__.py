"""Red-team agent tool package.

Paper §4.2 ("Red Team: Self-Evolving Agent") describes a tool-use
agent that calls a bounded set of marketplace-internal tools to decide
how to mutate a :class:`~core.schemas.SkillVariant`.  This package
exposes **six** tool schemas (OpenAI function-calling format) plus a
synchronous :func:`_dispatch_tool` that the runtime uses to execute each
tool.

Design invariants (frozen):
  * :data:`TOOLS` holds exactly six schemas.  Names and order are::

        read_skill, list_skills, read_repo_context, read_attack_log,
        get_binary, submit_mutation

    (``get_binary`` was added alongside code-layer mutation so the
    red-team agent can stage attack primitives from a curated catalog
    into ``new_c.scripts`` instead of inlining shell in ``d``.)
  * There is **no** ``web_search`` tool.  Disallowing public-web
    evidence is a paper claim — enforced at the schema level here,
    not in the prompt.
  * ``submit_mutation`` is the only terminal tool.  Its ``dispatch``
    returns ``{"_terminal": True, "args": <defensive-copy>}`` so the
    runtime can hand the payload to the mutation validator.
  * Every tool result is ``json.dumps``-able and free-form text is
    truncated to :data:`_PREVIEW_LIMIT` characters with a ``'…'``
    marker so tool-message content stays bounded.

Backward compatibility note: this package replaces the legacy
``red_team.agent.tools`` *module*.  All previously-public symbols
(``TOOLS``, ``TOOL_NAMES``, ``_dispatch_tool``, and the ``_PREVIEW_*``
/ ``_VALID_*`` / ``_ENTRY_SCRIPT_*`` constants) are re-exported here
so callers of the monolith continue to work unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from red_team.agent.tools.base import (
    AgentTool,
    _BENIGN_SAMPLE_SIZE,
    _ENTRY_SCRIPT_SUFFIXES,
    _PREVIEW_LIMIT,
    _VALID_REPO_SECTIONS,
    _VALID_SKILL_SPLITS,
)
from red_team.agent.tools.get_binary import GetBinaryTool
from red_team.agent.tools.list_skills import ListSkillsTool
from red_team.agent.tools.read_attack_log import ReadAttackLogTool
from red_team.agent.tools.read_repo_context import ReadRepoContextTool
from red_team.agent.tools.read_skill import ReadSkillTool
from red_team.agent.tools.submit_mutation import SubmitMutationTool


# Registration: order MUST match the frozen ``TOOL_NAMES`` contract
# because prompt templates enumerate tools positionally for humans.
# Adding a tool silently at the wrong position changes the numbering
# shown to the LLM.
_TOOL_INSTANCES: tuple[AgentTool, ...] = (  # type: ignore[assignment]
    ReadSkillTool(),
    ListSkillsTool(),
    ReadRepoContextTool(),
    ReadAttackLogTool(),
    GetBinaryTool(),
    SubmitMutationTool(),
)


TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in _TOOL_INSTANCES)
TOOLS: tuple[dict[str, Any], ...] = tuple(t.schema for t in _TOOL_INSTANCES)

# Import-time guard: any refactor that silently drifts from the six-tool
# contract blows up loudly on first import rather than in a downstream
# test days later.
assert len(TOOLS) == 6, "TOOLS must hold exactly 6 schemas (red-team contract)"
assert TOOL_NAMES == (
    "read_skill",
    "list_skills",
    "read_repo_context",
    "read_attack_log",
    "get_binary",
    "submit_mutation",
), "TOOL_NAMES drifted from the frozen contract"

# Fast lookup for the dispatcher.
_TOOL_BY_NAME: dict[str, AgentTool] = {t.name: t for t in _TOOL_INSTANCES}


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    seed: Any,
    findings: Any,
    repo: Any,
    attack_log_path: Path | str | None,
) -> dict[str, Any]:
    """Execute a single tool call.

    The runtime calls this for every ``tool_calls[*]`` entry returned by
    the LLM.

    Parameters
    ----------
    name
        Exact tool name.  Unknown → :class:`KeyError` (programmer error,
        not an LLM mistake, so we fail loudly).
    args
        JSON-parsed ``function.arguments`` dict.
    seed, findings, repo
        Reserved for future rounds (e.g. auto-filling ``repo_id`` from
        ``repo.repo_id`` if the LLM omits it).  Tool behaviour is
        stateless with respect to these; they stay in the signature
        because the signature itself is public contract.
    attack_log_path
        Path to ``outputs/evolution/<run>/attack_log.jsonl`` or ``None``
        if memory is disabled / first round.

    Returns
    -------
    dict
        JSON-serialisable result.  Invalid *args* within a known tool
        yield ``{"error": ...}`` rather than raising so the LLM can
        self-correct on the next turn.  ``submit_mutation`` returns the
        sentinel ``{"_terminal": True, "args": <defensive-copy>}``.
    """
    del seed, findings, repo  # reserved — see docstring

    tool = _TOOL_BY_NAME.get(name)
    if tool is None:
        raise KeyError(f"unknown tool: {name!r}")

    # Only ``read_attack_log`` needs the log path; passing it as the
    # plugin's ``context`` argument keeps the plugin signature uniform
    # and avoids leaking per-tool specifics into the dispatcher.
    if tool.name == "read_attack_log":
        return tool.dispatch(args, attack_log_path)
    return tool.dispatch(args)


__all__ = [
    "TOOLS",
    "TOOL_NAMES",
    "_dispatch_tool",
    "AgentTool",
    "_PREVIEW_LIMIT",
    "_BENIGN_SAMPLE_SIZE",
    "_VALID_REPO_SECTIONS",
    "_VALID_SKILL_SPLITS",
    "_ENTRY_SCRIPT_SUFFIXES",
]
