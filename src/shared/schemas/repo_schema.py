from __future__ import annotations

from core.models import RepositorySlice


def repo_to_record(repo: RepositorySlice) -> dict[str, object]:
    return repo.to_dict()


def repo_from_record(record: dict[str, object]) -> RepositorySlice:
    readme_path = record.get("readme_path")
    return RepositorySlice(
        repo_id=str(record["repo_id"]),
        repo_root=str(record["repo_root"]),
        readme_path=None if readme_path is None else str(readme_path),
        readme_content=str(record.get("readme_content", "")),
        file_tree=[str(item) for item in record.get("file_tree", [])],
        dependency_files=[str(item) for item in record.get("dependency_files", [])],
        labels=dict(record.get("labels", {})),
        metadata=dict(record.get("metadata", {})),
        extracted_themes=[str(item) for item in record.get("extracted_themes", [])],
    )
