"""Verdict protocol."""

from abc import ABC, abstractmethod
from typing import Any

from core.models import StaticFinding


class BaseVerdict(ABC):
    @abstractmethod
    def build(
        self,
        findings: list[StaticFinding],
        probe_outputs: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError
