"""High-level pipeline facade for path-based auditing."""

from pathlib import Path

from core.models import AuditReport
from shared.loaders import load_repository_slice, load_skill_artifact

from .ablation_config import BlueTeamAblationConfig
from .blue_orchestrator import BlueOrchestrator


class AuditPipeline:
    def __init__(self, config: BlueTeamAblationConfig | None = None) -> None:
        self._orchestrator: BlueOrchestrator = BlueOrchestrator(config=config)

    def audit_paths(self, skill_root: str | Path, repo_root: str | Path) -> AuditReport:
        skill = load_skill_artifact(skill_root)
        repo = load_repository_slice(repo_root)
        return self._orchestrator.audit(skill, repo)
