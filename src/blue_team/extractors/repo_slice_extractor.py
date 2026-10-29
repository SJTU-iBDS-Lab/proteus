"""Extract repo context signals from a RepositorySlice."""

from core.models import RepositorySlice
from shared.utils import tokenize

from ..base import BaseExtractor


class RepoSliceExtractor(BaseExtractor):
    def extract(self, artifact: RepositorySlice, **kwargs: object) -> dict[str, object]:
        return {
            "theme_tokens": set(artifact.theme_tokens),
            "readme_tokens": set(tokenize(artifact.readme_text)),
            "file_inventory": list(artifact.file_inventory),
            "file_count": len(artifact.file_inventory),
        }
