"""Smoke-import test for the IMPL-14 evolution CLI.

Catches syntax / import regressions in ``scripts/red_team/run_evolution.py``
and confirms ``--help`` parses cleanly + required args are enforced.  We
avoid actually invoking ``main()`` because it would call the real
blue-team + sandbox + LLM stack.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "red_team" / "run_evolution.py"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script_module():
    """Load the script without polluting ``sys.modules`` with the real name."""
    spec = importlib.util.spec_from_file_location(
        "_claw_run_evolution_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_imports_cleanly() -> None:
    mod = _load_script_module()
    # Public surface the script exposes.
    assert callable(mod.main)
    assert callable(mod.build_parser)


def test_load_seed_records_accepts_frozen_json_manifest(tmp_path: Path) -> None:
    mod = _load_script_module()
    manifest = tmp_path / "extension.json"
    manifest.write_text(
        '{"manifest_name":"heldout","seeds":[{"seed_id":"seed-21"}]}',
        encoding="utf-8",
    )

    assert mod._load_seed_records(manifest) == [{"seed_id": "seed-21"}]


def test_resolve_repo_slice_falls_back_to_downloaded_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_script_module()
    skill = tmp_path / "downloaded-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Downloaded skill\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DEFAULT_SKILLS_ROOT", tmp_path)

    repo = mod._resolve_repo_slice([], "downloaded-skill")

    assert repo.repo_id == "downloaded-skill"
    assert repo.readme_content == "# Downloaded skill\n"
    assert "SKILL.md" in repo.file_tree


def test_help_does_not_raise() -> None:
    mod = _load_script_module()
    parser = mod.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_required_args_enforced() -> None:
    mod = _load_script_module()
    parser = mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --seed and --repo are required


def test_help_subprocess_exits_zero() -> None:
    """End-to-end: ``python scripts/red_team/run_evolution.py --help`` → 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"--help returned {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "--seed" in result.stdout
    assert "--repo" in result.stdout
    # Epilog must document ablation flag names.
    for flag in (
        "disable_tool",
        "tool_use_off",
        "disable_expansion",
        "search_space",
        "force_topology",
        "forbid_channel_edit",
        "feedback",
        "memory_enabled",
        "disable_family_enumeration",
    ):
        assert flag in result.stdout, f"ablation flag {flag!r} missing from --help"


def test_missing_required_args_subprocess_exits_two() -> None:
    """End-to-end: no required args → argparse exit code 2."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 2, (
        f"expected exit 2 from argparse, got {result.returncode}\n"
        f"stderr={result.stderr}"
    )
