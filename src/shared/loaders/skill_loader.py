"""Load a skill directory into a normalized SkillArtifact."""

from __future__ import annotations

from pathlib import Path

from core.constants import (
    CAPABILITY_KEYWORDS,
    DEFAULT_ALLOWED_SCRIPT_SUFFIXES,
    DEFAULT_LABEL_FILE,
    DEFAULT_MANIFEST_FILE,
    GUIDANCE_PATTERNS,
    NETWORK_MARKERS,
    SENSITIVE_PATH_MARKERS,
)
from core.models import SkillArtifact
from shared.schemas import validate_skill_artifact
from shared.utils import collect_keyword_hits, normalize_text, read_json_file


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            items.append(item)
    return items


def _infer_language(path: Path | None) -> str:
    if path is None:
        return "text"
    return {
        ".py": "python",
        ".sh": "shell",
        ".js": "javascript",
        ".ts": "typescript",
    }.get(path.suffix.lower(), "text")


def _detect_capabilities(text: str) -> list[str]:
    matches: list[str] = []
    normalized = normalize_text(text)
    for capability, keywords in CAPABILITY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(capability)
    return sorted(set(matches))


def _detect_guidance_cues(text: str) -> list[str]:
    cues: list[str] = []
    for rule_name, keywords in GUIDANCE_PATTERNS.items():
        if collect_keyword_hits(text, keywords):
            cues.append(rule_name)
    return sorted(cues)


def _detect_source_tags(
    declared_capabilities: list[str],
    code_capabilities: list[str],
    code_text: str,
    labels: dict[str, object],
) -> list[str]:
    tags = set(_string_list(labels.get("source_tags", [])))
    capabilities = set(declared_capabilities) | set(code_capabilities)
    lowered = normalize_text(code_text)
    if "file_read" in capabilities:
        tags.add("local_files")
    if "credential_access" in capabilities or any(
        marker in lowered for marker in SENSITIVE_PATH_MARKERS
    ):
        tags.add("credentials")
    return sorted(tags)


def _detect_sink_tags(
    declared_capabilities: list[str],
    code_capabilities: list[str],
    code_text: str,
    labels: dict[str, object],
) -> list[str]:
    tags = set(_string_list(labels.get("sink_tags", [])))
    capabilities = set(declared_capabilities) | set(code_capabilities)
    lowered = normalize_text(code_text)
    if "network_access" in capabilities or any(
        marker in lowered for marker in NETWORK_MARKERS
    ):
        tags.add("network_egress")
    if "file_write" in capabilities:
        tags.add("filesystem_write")
    if "shell_exec" in capabilities:
        tags.add("shell_execution")
    return sorted(tags)


def load_skill_artifact(
    skill_root: str | Path, *, source: str = "local"
) -> SkillArtifact:
    root = Path(skill_root)
    skill_md = root / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"missing SKILL.md in {root}")

    candidate_scripts = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_ALLOWED_SCRIPT_SUFFIXES
    )
    skill_text = skill_md.read_text(encoding="utf-8")
    code_files = {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in candidate_scripts
    }
    entry_scripts = list(code_files.keys())
    entry_script_path = candidate_scripts[0] if candidate_scripts else None
    code_text = "\n\n".join(code_files.values())
    declared_capabilities = _detect_capabilities(skill_text)
    code_capabilities = _detect_capabilities(code_text)
    guidance_cues = _detect_guidance_cues(skill_text)

    labels_path = root / DEFAULT_LABEL_FILE
    manifest_path = root / DEFAULT_MANIFEST_FILE
    labels = read_json_file(labels_path) if labels_path.exists() else {}
    manifest = read_json_file(manifest_path) if manifest_path.exists() else {}

    artifact = SkillArtifact(
        skill_id=root.name,
        skill_root=str(root),
        skill_doc_path="SKILL.md",
        entry_scripts=entry_scripts,
        skill_doc=skill_text,
        code_files=code_files,
        metadata={
            "source": source,
            "language": _infer_language(entry_script_path),
            "declared_capabilities": declared_capabilities,
            "code_capabilities": code_capabilities,
            "guidance_cues": guidance_cues,
        },
        labels=labels,
        permissions_declared=declared_capabilities,
        bootstrap_files=["SKILL.md"],
        manifest=manifest,
        capability_summary=", ".join(declared_capabilities),
        code_capability_summary=", ".join(code_capabilities),
        source_tags=_detect_source_tags(
            declared_capabilities, code_capabilities, code_text, labels
        ),
        sink_tags=_detect_sink_tags(
            declared_capabilities, code_capabilities, code_text, labels
        ),
    )
    _ = validate_skill_artifact(artifact)
    return artifact
