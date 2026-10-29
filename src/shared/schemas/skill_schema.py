from __future__ import annotations

from core.models import SkillArtifact


def skill_to_record(skill: SkillArtifact) -> dict[str, object]:
    return skill.to_dict()


def skill_from_record(record: dict[str, object]) -> SkillArtifact:
    return SkillArtifact(
        skill_id=str(record["skill_id"]),
        skill_root=str(record["skill_root"]),
        skill_doc_path=str(record["skill_doc_path"]),
        entry_scripts=[str(item) for item in record.get("entry_scripts", [])],
        skill_doc=str(record.get("skill_doc", "")),
        code_files={
            str(key): str(value)
            for key, value in dict(record.get("code_files", {})).items()
        },
        metadata=dict(record.get("metadata", {})),
        labels=dict(record.get("labels", {})),
        permissions_declared=[
            str(item) for item in record.get("permissions_declared", [])
        ],
        bootstrap_files=[str(item) for item in record.get("bootstrap_files", [])],
        manifest=dict(record.get("manifest", {})),
        capability_summary=str(record.get("capability_summary", "")),
        code_capability_summary=str(record.get("code_capability_summary", "")),
        source_tags=[str(item) for item in record.get("source_tags", [])],
        sink_tags=[str(item) for item in record.get("sink_tags", [])],
    )
