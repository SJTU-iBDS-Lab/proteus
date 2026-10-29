"""Plugin base for the red-team agent tool package.

Each per-tool file in :mod:`red_team.agent.tools` implements a small
class that matches :class:`AgentTool` — a schema + name + terminal flag
+ ``dispatch`` method.  The package ``__init__`` registers the six
instances in the frozen order and re-exports :data:`TOOLS`,
:data:`TOOL_NAMES`, and :func:`_dispatch_tool` so callers of the
legacy monolithic module keep working.

The shared constants live here (and only here) so every plugin imports
from a single canonical place; tests pin their values because prompt
token accounting depends on the preview cap.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# Truncation cap for any free-form text field in tool results.  Mirrored
# back into the LLM context, so this directly bounds token cost per round.
_PREVIEW_LIMIT = 8000

# ``list_skills(category="benign")`` sample size — paper §4.2 asks for
# "a handful"; fixed at 20 for reproducibility.
_BENIGN_SAMPLE_SIZE = 20

# Accepted values for ``read_repo_context(section=...)``; anything else
# yields a structured error the LLM can self-correct on.
_VALID_REPO_SECTIONS = frozenset({"readme", "tree", "theme"})

# Accepted values for ``read_skill(split=...)``.
_VALID_SKILL_SPLITS = frozenset({"benign", "malicious"})

# Entry-script suffix preference — order matters because we return the
# first match so "the entry script" is stable across runs.
_ENTRY_SCRIPT_SUFFIXES = (".py", ".sh", ".js", ".ts")


@runtime_checkable
class AgentTool(Protocol):
    """Contract every tool plugin must satisfy.

    Attributes
    ----------
    schema
        OpenAI function-calling schema dict.  Exactly the shape an
        OpenAI-compatible client accepts.
    name
        Tool name (must match ``schema["function"]["name"]``).
    terminal
        True iff invoking this tool ends the agent loop; only
        ``submit_mutation`` sets it.

    Methods
    -------
    dispatch(args, context=None)
        Execute the tool.  ``submit_mutation`` returns the sentinel
        ``{"_terminal": True, "args": <defensive-copy>}`` so the runtime
        can hand the payload to the mutation builder rather than building
        the :class:`SkillVariant` here.
    """

    schema: dict
    name: str
    terminal: bool

    def dispatch(self, args: dict, context: Any = None) -> Any:
        ...
