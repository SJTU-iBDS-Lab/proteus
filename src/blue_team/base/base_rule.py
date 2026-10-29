"""Rule protocol."""

from abc import ABC, abstractmethod
from typing import Any


class BaseRule(ABC):
    rule_id: str
    severity: str

    @abstractmethod
    def match(self, extracted: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError
