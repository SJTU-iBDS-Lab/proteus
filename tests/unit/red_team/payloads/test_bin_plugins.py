"""Contract tests for every bin-layer payload plugin.

Each ``BIN_REGISTRY`` entry must wrap an existing
:data:`binary_catalog.BINARY_CATALOG` entry and render to a
:class:`PayloadArtifact` whose ``content`` is exactly the bash wrapper
produced by :func:`get_binary` and whose ``metadata`` preserves the
catalog fields the downstream sandbox runner consumes
(``filename``, ``params_used``, ``covers_objectives``, ...).

Go-dependent assertions are gated by ``requires_go`` so environments
without the toolchain skip gracefully — same pattern as the existing
``test_binary_catalog.py``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective, PayloadArtifact
from red_team.payloads.bin import REGISTRY as BIN_REGISTRY
from red_team.payloads.binary_catalog import BINARY_CATALOG


_GO_AVAILABLE = shutil.which("go") is not None
requires_go = pytest.mark.skipif(
    not _GO_AVAILABLE, reason="go toolchain unavailable"
)


EXPECTED_BIN_IDS = {
    "honeypot_beacon",
    "rc_installer",
    "privesc_grant",
    "content_tamperer",
    "file_shredder",
    "remote_stager",
}


@pytest.fixture()
def isolated_build_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLAW_BINARY_BUILD_ROOT", str(tmp_path))
    return tmp_path


def test_bin_registry_matches_binary_catalog_keys() -> None:
    assert set(BIN_REGISTRY) == EXPECTED_BIN_IDS
    assert set(BIN_REGISTRY) == set(BINARY_CATALOG)


@pytest.mark.parametrize("key", sorted(EXPECTED_BIN_IDS))
def test_each_bin_plugin_static_contract(key: str) -> None:
    plugin = BIN_REGISTRY[key]
    assert plugin.name == key
    assert plugin.layer == "bin"
    assert isinstance(plugin.objective, AttackObjective)


@requires_go
@pytest.mark.parametrize("key", sorted(EXPECTED_BIN_IDS))
def test_each_bin_plugin_renders_successfully(
    isolated_build_root: Path, key: str
) -> None:
    plugin = BIN_REGISTRY[key]
    artifact = plugin.render()

    assert isinstance(artifact, PayloadArtifact)
    assert artifact.name == key
    assert artifact.layer == "bin"
    assert artifact.objective is plugin.objective
    assert artifact.content, "bash wrapper content must be non-empty"
    assert artifact.content.startswith("#!/usr/bin/env bash")

    # Metadata must preserve the catalog-row fields a sandbox runner needs.
    md = artifact.metadata
    assert md["filename"] == BINARY_CATALOG[key].filename
    assert md["binary_id"] == key
    assert md["language"] == "go"
    assert "params_used" in md
    assert "covers_objectives" in md
    # Content moved out of metadata — prevents double-carrying the wrapper.
    assert "content" not in md


@requires_go
def test_bin_plugin_forwards_kwargs_to_get_binary(
    isolated_build_root: Path,
) -> None:
    plugin = BIN_REGISTRY["honeypot_beacon"]
    artifact = plugin.render(
        HONEYPOT_URL="http://10.0.0.42:4444/x",
        LABEL="tag with space",
    )
    # Simple values remain unquoted; values with whitespace are single-quoted.
    assert "export HONEYPOT_URL=http://10.0.0.42:4444/x" in artifact.content
    assert "export LABEL='tag with space'" in artifact.content
    assert artifact.metadata["params_used"]["HONEYPOT_URL"] == (
        "http://10.0.0.42:4444/x"
    )


def test_bin_objectives_cover_non_guidance_attack_space() -> None:
    """Every bin plugin must map to one of the 7 paper objectives, and
    guidance-injection stays doc-only by design."""
    objectives = {p.objective for p in BIN_REGISTRY.values()}
    assert AttackObjective.GUIDANCE_INJECTION not in objectives
    # Must cover at least these core objectives.
    assert AttackObjective.DATA_EXFILTRATION in objectives
    assert AttackObjective.PERSISTENCE in objectives
    assert AttackObjective.PRIVILEGE_ESCALATION in objectives
    assert AttackObjective.CLEANUP in objectives
    assert AttackObjective.SUPPLY_CHAIN_ABUSE in objectives
