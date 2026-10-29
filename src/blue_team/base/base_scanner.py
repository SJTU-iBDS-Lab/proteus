"""Scanner protocol."""

from abc import ABC, abstractmethod
from typing import Any

from core.models import (
    RepositorySlice,
    SkillArtifact,
    StaticFinding,
    StructuredFinding,
)


class BaseScanner(ABC):
    @abstractmethod
    def scan(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: Any
    ) -> list[StaticFinding]:
        raise NotImplementedError

    def scan_structured(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: Any
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        return self.scan(skill, repo, **kwargs), []
