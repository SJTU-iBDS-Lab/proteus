"""Probe protocol."""

from abc import ABC, abstractmethod
from typing import Any

from core.models import RepositorySlice, SkillArtifact


class BaseProbe(ABC):
    @abstractmethod
    def run(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: Any
    ) -> dict[str, Any]:
        raise NotImplementedError
