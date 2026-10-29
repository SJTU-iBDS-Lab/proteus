from __future__ import annotations

from pathlib import Path

from blue_team import AuditPipeline
from core.enums import VerdictStatus
from shared.loaders import load_repository_slice, load_skill_artifact


def test_pytest_loader_baseline_preserves_entry_script_and_theme_tokens(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "repo-summary"
    repo_root = tmp_path / "knowledge-repo"
    (skill_root / "scripts").mkdir(parents=True)
    repo_root.mkdir(parents=True)

    _ = (skill_root / "SKILL.md").write_text(
        "# Repo Summary\nRead the repository and summarize findings safely.",
        encoding="utf-8",
    )
    _ = (skill_root / "scripts" / "main.py").write_text(
        "def summarize() -> str:\n    return 'ok'\n",
        encoding="utf-8",
    )
    _ = (repo_root / "README.md").write_text(
        "Repository for safe summaries and reporting.",
        encoding="utf-8",
    )

    skill = load_skill_artifact(skill_root)
    repo = load_repository_slice(repo_root)

    assert skill.entry_script == "scripts/main.py"
    assert "file_read" in skill.declared_capabilities
    assert "repository" in repo.theme_tokens


def test_pytest_audit_baseline_keeps_runtime_summary_proxy(tmp_path: Path) -> None:
    repo_root = tmp_path / "knowledge-repo"
    skill_root = tmp_path / "summary-skill"
    repo_root.mkdir(parents=True)
    (skill_root / "scripts").mkdir(parents=True)

    _ = (repo_root / "README.md").write_text(
        "Knowledge repo for safe summaries.",
        encoding="utf-8",
    )
    _ = (skill_root / "SKILL.md").write_text(
        "# Summary Skill\nRead repository docs and summarize them safely.",
        encoding="utf-8",
    )
    _ = (skill_root / "scripts" / "main.py").write_text(
        "def summarize_repo(root: str) -> str:\n    return root\n",
        encoding="utf-8",
    )

    report = AuditPipeline().audit_paths(skill_root, repo_root)

    assert report.verdict in {VerdictStatus.APPROVED, VerdictStatus.CONDITIONAL}
    assert report.execution_trace.runtime_trace_summary == report.summary
