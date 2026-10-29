"""Tests for declared_capabilities tag enrichment fallback."""
from __future__ import annotations

from dataset.fingerprint_pipeline import enrich_tags_from_declared_capabilities
from core.models import SkillFingerprintRecord


def _make_record(
    *,
    source_tags: list[str] | None = None,
    sink_tags: list[str] | None = None,
    declared_capabilities: list[str] | None = None,
) -> SkillFingerprintRecord:
    meta: dict[str, object] = {}
    if declared_capabilities is not None:
        meta["declared_capabilities"] = declared_capabilities
    return SkillFingerprintRecord(
        skill_id="test-skill",
        skill_path="/tmp/test",
        schema_version="subgraph.v1",
        cluster_id="cluster-0001",
        category="batch",
        heat_tier="unknown",
        language="text",
        dependency_ecosystem=[],
        metadata=meta,
        source_tags=source_tags or [],
        sink_tags=sink_tags or [],
        selection_reason="fingerprint_extracted",
    )


def test_enriches_credential_access_and_shell_exec() -> None:
    rec = _make_record(
        declared_capabilities=["credential_access", "shell_exec"]
    )
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert "credentials" in enriched.source_tags
    assert "shell_execution" in enriched.sink_tags
    assert enriched.selection_reason == "declared_capability_fallback"


def test_enriches_network_access_as_dual() -> None:
    rec = _make_record(declared_capabilities=["network_access"])
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert "network_input" in enriched.source_tags
    assert "network_egress" in enriched.sink_tags


def test_enriches_file_read_and_file_write() -> None:
    rec = _make_record(
        declared_capabilities=["file_read", "file_write"]
    )
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert "local_files" in enriched.source_tags
    assert "filesystem_write" in enriched.sink_tags


def test_skips_when_ast_tags_present() -> None:
    rec = _make_record(
        source_tags=["credentials"],
        sink_tags=["network_egress"],
        declared_capabilities=["shell_exec"],
    )
    enriched = enrich_tags_from_declared_capabilities(rec)
    # Should NOT add shell_execution since AST tags are already present
    assert enriched.source_tags == ["credentials"]
    assert enriched.sink_tags == ["network_egress"]
    assert enriched.selection_reason == "fingerprint_extracted"


def test_skips_when_no_declared_capabilities() -> None:
    rec = _make_record(declared_capabilities=[])
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert enriched.source_tags == []
    assert enriched.sink_tags == []


def test_skips_when_metadata_missing_capabilities() -> None:
    rec = _make_record()
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert enriched.source_tags == []
    assert enriched.sink_tags == []


def test_skips_unknown_capabilities() -> None:
    rec = _make_record(
        declared_capabilities=["browser_automation", "email_send"]
    )
    enriched = enrich_tags_from_declared_capabilities(rec)
    assert enriched.source_tags == []
    assert enriched.sink_tags == []
    assert enriched.selection_reason == "fingerprint_extracted"
