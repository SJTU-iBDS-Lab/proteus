"""Backward-compat shim — real implementations have moved.

This module used to own the poison-payload catalogue, prompt
building, and on-disk skill materialisation.  Each concern now lives
in its proper home:

- :data:`POISON_PAYLOADS`            → :mod:`red_team.payloads.doc`
- :func:`build_poisoned_prompt`      → :mod:`red_team.payloads.prompts`
- :func:`list_attack_types`          → :mod:`red_team.payloads`
- :func:`materialize_poisoned_skill` → :mod:`sandbox.materialize`
- :func:`materialize_poisoned_chain` → :mod:`sandbox.materialize`

``POISON_PAYLOADS``, ``build_poisoned_prompt`` and ``list_attack_types``
are re-exported verbatim (``is``-identical to their canonical homes).

``materialize_poisoned_skill`` / ``materialize_poisoned_chain`` are
wrapped by a thin **adapter** rather than re-exported directly.  The
sandbox implementations now take a pre-resolved ``payload_text``
keyword — the adapter accepts the legacy ``attack_type`` argument
(either positional or keyword), resolves it through
:data:`POISON_PAYLOADS`, and delegates.  This decouples the sandbox
layer from the red-team payload catalogue (no ``sandbox -> red_team``
import edge) while keeping every historical call site working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from red_team.payloads import list_attack_types
from red_team.payloads.doc import POISON_PAYLOADS
from red_team.payloads.prompts import build_poisoned_prompt
from sandbox.materialize import (
    materialize_poisoned_chain as _sandbox_materialize_chain,
    materialize_poisoned_skill as _sandbox_materialize_skill,
)

_DEFAULT_PAYLOAD_KEY = "data_exfiltration"


def _resolve_payload_text(
    attack_type: str | None, payload_text: str | None
) -> str:
    """Return the literal payload text the sandbox layer should append.

    Accepts exactly one of:
    - ``payload_text`` — already-resolved text, returned verbatim; or
    - ``attack_type`` — a POISON_PAYLOADS key.  Unknown keys fall back
      to the historical default (``data_exfiltration``) so behaviour
      matches the pre-adapter ``_resolve_payload`` exactly.

    Raises ``ValueError`` when neither is supplied — that is a caller
    bug and silently picking a default would mask it.
    """
    if payload_text is not None:
        return payload_text
    if attack_type is None:
        raise ValueError(
            "must supply either attack_type or payload_text"
        )
    key = attack_type.strip().lower()
    return POISON_PAYLOADS.get(key, POISON_PAYLOADS[_DEFAULT_PAYLOAD_KEY])


def materialize_poisoned_skill(
    src_skill_dir: Path,
    attack_type: str | None = None,
    dst_skill_dir: Path | None = None,
    *,
    payload_text: str | None = None,
    **kwargs: Any,
) -> Path:
    """Legacy-signature wrapper around
    :func:`sandbox.materialize.materialize_poisoned_skill`.

    Historical call shape::

        materialize_poisoned_skill(src_dir, attack_type, dst_dir)

    is still supported.  Modern callers should prefer::

        materialize_poisoned_skill(src_dir, dst_dir, payload_text=...)

    via the ``sandbox.materialize`` module directly.
    """
    if dst_skill_dir is None:
        raise TypeError(
            "materialize_poisoned_skill() missing required argument "
            "'dst_skill_dir'"
        )
    resolved = _resolve_payload_text(attack_type, payload_text)
    return _sandbox_materialize_skill(
        src_skill_dir, dst_skill_dir, payload_text=resolved, **kwargs
    )


def materialize_poisoned_chain(
    chain_skill_dirs: list[Path],
    attack_type: str | None = None,
    chain_root: Path | None = None,
    *,
    payload_text: str | None = None,
    poison_index: int = -1,
    **kwargs: Any,
) -> tuple[list[Path], Path]:
    """Legacy-signature wrapper around
    :func:`sandbox.materialize.materialize_poisoned_chain`.

    Historical call shape::

        materialize_poisoned_chain(chain_dirs, attack_type, chain_root,
                                   poison_index=...)

    is still supported.  Modern callers should prefer::

        materialize_poisoned_chain(chain_dirs, chain_root,
                                   payload_text=..., poison_index=...)

    via the ``sandbox.materialize`` module directly.
    """
    if chain_root is None:
        raise TypeError(
            "materialize_poisoned_chain() missing required argument "
            "'chain_root'"
        )
    resolved = _resolve_payload_text(attack_type, payload_text)
    return _sandbox_materialize_chain(
        chain_skill_dirs,
        chain_root,
        payload_text=resolved,
        poison_index=poison_index,
        **kwargs,
    )


__all__ = [
    "POISON_PAYLOADS",
    "build_poisoned_prompt",
    "list_attack_types",
    "materialize_poisoned_skill",
    "materialize_poisoned_chain",
]
