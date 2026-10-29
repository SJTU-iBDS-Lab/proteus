"""Extractor protocol."""

from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, artifact: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
