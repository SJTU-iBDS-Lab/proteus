"""Loader entrypoints for skill and repository fixtures."""

from .repo_loader import load_repository_slice
from .skill_loader import load_skill_artifact

__all__ = ["load_repository_slice", "load_skill_artifact"]
