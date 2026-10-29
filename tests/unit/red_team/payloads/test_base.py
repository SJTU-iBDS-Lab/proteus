"""Payload protocol contract.

The ``Payload`` protocol is the single shape every payload plugin (doc
or bin) must honor so ``registry.REGISTRY`` can mix them uniformly. We
keep the contract tiny (three attributes + ``render``) and verify it is
a ``runtime_checkable`` Protocol so downstream code can guard on it.
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
from red_team.payloads.base import Payload


def test_payload_protocol_is_runtime_checkable() -> None:
    # Plain objects missing the contract must NOT isinstance-match.
    class _Bare:
        pass

    assert not isinstance(_Bare(), Payload)


def test_minimal_payload_matches_protocol() -> None:
    """A tiny in-test stub that satisfies the contract must match."""

    class _Stub:
        name = "stub"
        layer = "doc"
        objective = AttackObjective.DATA_EXFILTRATION

        def render(self, **kwargs) -> PayloadArtifact:
            return PayloadArtifact(
                name=self.name,
                layer=self.layer,
                objective=self.objective,
                content="hello",
                metadata={},
            )

    stub = _Stub()
    assert isinstance(stub, Payload)
    art = stub.render()
    assert isinstance(art, PayloadArtifact)
    assert art.name == "stub"
    assert art.layer == "doc"


def test_doc_and_bin_registry_samples_match_protocol() -> None:
    """At least one real plugin from each of doc/ and bin/ must match."""
    from red_team.payloads.doc import REGISTRY as DOC_REGISTRY
    from red_team.payloads.bin import REGISTRY as BIN_REGISTRY

    assert DOC_REGISTRY, "doc REGISTRY must be non-empty"
    assert BIN_REGISTRY, "bin REGISTRY must be non-empty"

    doc_sample = next(iter(DOC_REGISTRY.values()))
    bin_sample = next(iter(BIN_REGISTRY.values()))

    assert isinstance(doc_sample, Payload)
    assert isinstance(bin_sample, Payload)
    assert doc_sample.layer == "doc"
    assert bin_sample.layer == "bin"
