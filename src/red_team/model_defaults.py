"""Default red-team model ids loaded from ``configs/models.yaml``.

Centralised so every public entry point (``run_evolution``, ``agent_run``,
``family_enumerate``, ``path_expand``, ``surface_expand``, mutation
strategies) shares one source of truth.

Usage in a function signature:

    from red_team.model_defaults import default_mutator_model

    def my_func(..., mutator_model: str = default_mutator_model(), ...):
        ...

The lookup is evaluated at module-import time (Python default-argument
semantics).  Callers that pass an explicit ``mutator_model=...`` skip
the lookup entirely.

Yaml shape (``configs/models.yaml``):

    red_team:
      mutator:    deepseek-v4-pro
      extractor:  deepseek-v4-pro
      evolver:    deepseek-v4-pro

**No silent fallbacks.**  If the yaml is missing the file, the
``red_team`` section, or a specific field, each helper raises
:class:`RuntimeError` so misconfiguration fails loud at import time
instead of silently routing to the wrong upstream.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "default_mutator_model",
    "default_extractor_model",
    "default_evolver_model",
]


# Walk up from this file (src/red_team/model_defaults.py) to the repo
# root that owns ``configs/``.  Two ``.parents`` hops: red_team → src.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_YAML = _PROJECT_ROOT / "configs" / "models.yaml"


@lru_cache(maxsize=1)
def _load_red_team_section() -> dict[str, Any]:
    """Return the ``red_team`` section of ``configs/models.yaml``.

    Cached for the process lifetime — the file is treated as static
    config; restart to pick up changes.  Raises if the file or section
    is missing — there is no sane silent fallback for "what model does
    the red team use".
    """
    if not _MODELS_YAML.is_file():
        raise RuntimeError(
            f"configs/models.yaml not found at {_MODELS_YAML} — "
            "create it with a `red_team:` section or pass --mutator-model."
        )
    import yaml  # type: ignore

    data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(
            f"configs/models.yaml is not a dict (got {type(data).__name__})"
        )
    section = data.get("red_team")
    if not isinstance(section, dict):
        raise RuntimeError(
            "configs/models.yaml is missing a `red_team:` section"
        )
    return section


def _required(field: str) -> str:
    """Look up ``red_team.<field>`` or raise."""
    section = _load_red_team_section()
    val = section.get(field)
    if not val:
        raise RuntimeError(
            f"configs/models.yaml red_team.{field} is missing or empty"
        )
    return str(val)


def default_mutator_model() -> str:
    """Return the configured red-team mutator model id (raise if unset)."""
    return _required("mutator")


def default_extractor_model() -> str:
    """Return the configured red-team extractor model id (raise if unset)."""
    return _required("extractor")


def default_evolver_model() -> str:
    """Return the configured red-team evolver model id (raise if unset)."""
    return _required("evolver")
