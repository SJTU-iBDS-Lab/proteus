"""Load a repository slice for repo-aware scanning."""

from __future__ import annotations

from pathlib import Path

from core.constants import DEFAULT_LABEL_FILE, DEFAULT_README_FILES
from core.models import RepositorySlice
from shared.schemas import validate_repository_slice
from shared.utils import read_json_file, tokenize

_DEPENDENCY_FILENAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "poetry.lock",
    "uv.lock",
    "Pipfile",
    "Cargo.toml",
    "go.mod",
}


def _find_readme(root: Path) -> Path | None:
    for name in DEFAULT_README_FILES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def load_repository_slice(
    repo_root: str | Path, *, repo_id: str | None = None
) -> RepositorySlice:
    root = Path(repo_root)
    readme = _find_readme(root)
    readme_text = readme.read_text(encoding="utf-8") if readme else ""
    file_inventory = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    dependency_files = [
        path for path in file_inventory if Path(path).name in _DEPENDENCY_FILENAMES
    ]
    token_source = " ".join([readme_text, " ".join(file_inventory), root.name])
    labels_path = root / DEFAULT_LABEL_FILE
    labels = read_json_file(labels_path) if labels_path.exists() else {}
    repo = RepositorySlice(
        repo_id=repo_id or root.name,
        repo_root=str(root),
        readme_path=str(readme.relative_to(root)) if readme else None,
        readme_content=readme_text,
        file_tree=file_inventory,
        dependency_files=dependency_files,
        labels=labels,
        metadata={"file_count": len(file_inventory)},
        extracted_themes=sorted(set(tokenize(token_source))),
    )
    _ = validate_repository_slice(repo)
    return repo
