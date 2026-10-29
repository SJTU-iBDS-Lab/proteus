"""Copy red-team skill chains to disk and append a poison payload to one
of the chain's ``SKILL.md`` files.

Canonical home for ``materialize_poisoned_skill`` /
``materialize_poisoned_chain``.

Dependency direction
--------------------
This module deliberately does *not* import anything from
``red_team``.  It is a low-level file-copy + text-append utility; the
caller is responsible for resolving what text to append (the "poison
payload") and passes it in as ``payload_text``.  Historical callers
that still speak the ``attack_type`` vocabulary go through the
``red_team.skill_poisoner`` shim, which does the
``attack_type -> POISON_PAYLOADS[...]`` lookup before delegating here —
that keeps the ``sandbox -> red_team.payloads`` reverse edge out of the
import graph.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def _find_skill_md(skill_dir: Path) -> Path | None:
    """Locate the SKILL.md file inside a (possibly nested) skill directory."""
    direct = skill_dir / "SKILL.md"
    if direct.is_file():
        return direct
    nested = skill_dir / skill_dir.name / "SKILL.md"
    if nested.is_file():
        return nested
    for found in skill_dir.rglob("SKILL.md"):
        return found
    return None


def _append_payload(skill_md: Path, payload_text: str) -> None:
    """Append ``payload_text`` to ``skill_md``, guaranteeing that the
    original body ends with a newline first so the payload starts on a
    fresh line."""
    existing = skill_md.read_text(encoding="utf-8", errors="replace")
    if not existing.endswith("\n"):
        existing += "\n"
    skill_md.write_text(existing + payload_text, encoding="utf-8")


def _ensure_skill_frontmatter(skill_dir: Path, skill_name: str) -> None:
    """Inject minimal YAML frontmatter on the staged SKILL.md if missing.

    Some source skills in ``data/skills/clawhub/`` ship without the
    ``---\\nname:...\\ndescription:...\\n---`` header.  OpenClaw's skill
    registry parser only enumerates skills that begin with that block;
    a frontmatter-less SKILL.md exists on disk but is invisible to
    ``openclaw skills``.  Strong-alignment targets (gpt-5.4-mini) then
    refuse with "I don't see <skill> in the available skills list".

    The fix lives at the staging step (not the source) because the
    source files are owned by root and we don't want a destructive
    in-place edit.  Idempotent — only writes when the file does not
    already start with ``---``.
    """
    md = _find_skill_md(skill_dir)
    if md is None:
        return
    body = md.read_text(encoding="utf-8", errors="replace")
    if body.lstrip().startswith("---"):
        return
    # Pull the first markdown H1 as a description hint when present;
    # otherwise fall back to the directory name.
    first_line = (body.lstrip().splitlines() or [""])[0].lstrip("# ").strip()
    description = first_line or skill_name
    fm = (
        "---\n"
        f"name: {skill_name}\n"
        f"description: \"{description}\"\n"
        "---\n\n"
    )
    md.write_text(fm + body, encoding="utf-8")


def materialize_poisoned_skill(
    src_skill_dir: Path,
    dst_skill_dir: Path,
    *,
    payload_text: str,
) -> Path:
    """Copy ``src_skill_dir`` to ``dst_skill_dir`` and append
    ``payload_text`` to the copy's SKILL.md.

    Returns the path to the modified SKILL.md. Raises ``FileNotFoundError``
    if no SKILL.md exists in the source tree. The destination is removed
    and recreated if it already exists so the operation is idempotent.
    """
    if dst_skill_dir.exists():
        shutil.rmtree(dst_skill_dir)
    shutil.copytree(src_skill_dir, dst_skill_dir)

    skill_md = _find_skill_md(dst_skill_dir)
    if skill_md is None:
        raise FileNotFoundError(
            f"no SKILL.md found under {dst_skill_dir!s}"
        )

    _append_payload(skill_md, payload_text)
    return skill_md


def materialize_poisoned_chain(
    chain_skill_dirs: list[Path],
    chain_root: Path,
    *,
    payload_text: str,
    poison_index: int = -1,
) -> tuple[list[Path], Path]:
    """Copy every skill in a chain to ``chain_root`` and poison one of them.

    ``poison_index`` is the position (default ``-1`` = last = sink) whose
    SKILL.md gets ``payload_text`` appended. Returns
    ``(list of materialized skill dirs, path to the poisoned SKILL.md)``.
    """
    if not chain_skill_dirs:
        raise ValueError("empty chain")

    chain_root.mkdir(parents=True, exist_ok=True)
    dst_paths: list[Path] = []
    for src in chain_skill_dirs:
        dst = chain_root / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        dst_paths.append(dst)
        # Ensure each staged SKILL.md has YAML frontmatter so the
        # OpenClaw skill registry actually enumerates it on
        # ``openclaw skills``.  A frontmatter-less SKILL.md (raw
        # markdown) still exists on disk but is invisible to the
        # registry — strong-alignment targets (gpt-5.4-mini) refuse
        # because the user-named skill "doesn't appear in the
        # available skills list", and even subtle mutations are
        # blocked at the "I don't see this skill" stage.  Inject a
        # minimal frontmatter on the staged copy if the source is
        # missing it; the original source file is untouched.
        _ensure_skill_frontmatter(dst, src.name)

    if poison_index < 0:
        poison_index += len(dst_paths)
    poison_index = max(0, min(len(dst_paths) - 1, poison_index))

    poisoned_md = _find_skill_md(dst_paths[poison_index])
    if poisoned_md is None:
        raise FileNotFoundError(
            f"no SKILL.md found for poison target {dst_paths[poison_index]!s}"
        )
    _append_payload(poisoned_md, payload_text)

    return dst_paths, poisoned_md


__all__ = [
    "materialize_poisoned_skill",
    "materialize_poisoned_chain",
]
