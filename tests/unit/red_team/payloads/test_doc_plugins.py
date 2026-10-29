"""Contract tests for every doc-layer payload plugin.

We walk the whole ``red_team.payloads.doc.REGISTRY`` and verify that:

* Every key mirrors an entry in ``skill_poisoner.POISON_PAYLOADS`` (the
  legacy shim view of the same dict) and, after the W4 flip, each doc
  plugin owns its own literal payload text as ``_PAYLOAD_TEXT``.
* Every plugin renders to a :class:`PayloadArtifact` whose ``content``
  is byte-identical to the corresponding POISON_PAYLOADS entry.
* ``layer`` is always ``"doc"``.
* ``metadata`` records the upstream key and a source module path so
  reproducers can map a :class:`PayloadArtifact` back to the dict entry
  and the plugin module that produced it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective, PayloadArtifact
from red_team.skill_poisoner import POISON_PAYLOADS
from red_team.payloads.doc import REGISTRY as DOC_REGISTRY


def test_doc_registry_matches_poison_payload_keys() -> None:
    """Every POISON_PAYLOADS key has exactly one doc plugin, and vice versa."""
    assert set(DOC_REGISTRY) == set(POISON_PAYLOADS), (
        f"doc registry diverged from POISON_PAYLOADS:\n"
        f"  only in registry: {set(DOC_REGISTRY) - set(POISON_PAYLOADS)}\n"
        f"  only in POISON_PAYLOADS: {set(POISON_PAYLOADS) - set(DOC_REGISTRY)}"
    )


@pytest.mark.parametrize("key", sorted(POISON_PAYLOADS))
def test_each_doc_plugin_renders_faithfully(key: str) -> None:
    plugin = DOC_REGISTRY[key]

    # Static contract.
    assert plugin.name == key
    assert plugin.layer == "doc"
    assert isinstance(plugin.objective, AttackObjective)

    artifact = plugin.render()
    assert isinstance(artifact, PayloadArtifact)
    assert artifact.name == key
    assert artifact.layer == "doc"
    assert artifact.objective is plugin.objective
    # Byte-for-byte equality with the original seed dict.
    assert artifact.content == POISON_PAYLOADS[key]
    assert artifact.content, "content must be non-empty"

    # Metadata must let a consumer trace back to the seed dict and the
    # owning plugin module (source-of-truth after the W4 flip).
    assert artifact.metadata.get("key") == key
    assert artifact.metadata.get("source") == f"payloads.doc.{key}"


def test_doc_registry_has_no_duplicate_plugins() -> None:
    """Same instance must not be registered under two keys."""
    ids = {id(p) for p in DOC_REGISTRY.values()}
    assert len(ids) == len(DOC_REGISTRY)


def test_guidance_payloads_mapped_to_guidance_injection() -> None:
    """Guidance payloads have no explicit mapping in
    :meth:`AttackObjective.from_payload_key`; the plugins must still tag
    them correctly as ``GUIDANCE_INJECTION``."""
    for key in ("guidance_authority_claim", "guidance_autonomy_nudge"):
        assert DOC_REGISTRY[key].objective is AttackObjective.GUIDANCE_INJECTION
