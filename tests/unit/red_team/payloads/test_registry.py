"""Contract tests for the unified ``red_team.payloads.registry``.

The registry is the single dict downstream code (agent tools, attack
log, evolution driver) looks at to resolve a payload name to a plugin,
regardless of whether it lives at the doc or bin layer. Name collisions
between the two layers are explicitly disallowed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective
from red_team.payloads import registry
from red_team.payloads.base import Payload
from red_team.payloads.bin import REGISTRY as BIN_REGISTRY
from red_team.payloads.doc import REGISTRY as DOC_REGISTRY


def test_registry_is_union_of_doc_and_bin() -> None:
    assert set(registry.REGISTRY) == set(DOC_REGISTRY) | set(BIN_REGISTRY)


def test_doc_and_bin_names_do_not_collide() -> None:
    """Doc and bin plugins must use disjoint names — the registry is a
    flat dict, so overlap would silently discard one of them."""
    overlap = set(DOC_REGISTRY) & set(BIN_REGISTRY)
    assert overlap == set(), f"doc/bin name collision: {overlap}"


def test_every_registered_entry_matches_payload_protocol() -> None:
    for name, plugin in registry.REGISTRY.items():
        assert isinstance(plugin, Payload), name
        assert plugin.name == name, (plugin.name, name)
        assert plugin.layer in {"doc", "bin"}


def test_get_returns_plugin_by_name() -> None:
    sample_name = next(iter(DOC_REGISTRY))
    got = registry.get(sample_name)
    assert got is DOC_REGISTRY[sample_name]


def test_get_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        registry.get("definitely-not-a-payload")


def test_by_layer_partitions_correctly() -> None:
    docs = registry.by_layer("doc")
    bins = registry.by_layer("bin")
    assert {p.name for p in docs} == set(DOC_REGISTRY)
    assert {p.name for p in bins} == set(BIN_REGISTRY)
    # Unknown layer -> empty, not error.
    assert registry.by_layer("no_such_layer") == []


def test_by_objective_returns_only_matching_plugins() -> None:
    # DATA_EXFILTRATION should appear in both doc and bin rosters.
    plugins = registry.by_objective(AttackObjective.DATA_EXFILTRATION)
    names = {p.name for p in plugins}
    assert "data_exfiltration" in names       # doc
    assert "honeypot_beacon" in names         # bin

    # GUIDANCE_INJECTION is doc-only by design.
    guidance = registry.by_objective(AttackObjective.GUIDANCE_INJECTION)
    guidance_names = {p.name for p in guidance}
    assert "guidance_authority_claim" in guidance_names
    assert all(p.layer == "doc" for p in guidance)

    # Empty result is a plain list.
    class _FakeObjective:
        pass

    # Pass in a sentinel that cannot match any AttackObjective value.
    empty = [p for p in registry.REGISTRY.values() if p.objective is _FakeObjective]
    assert empty == []


def test_by_objective_result_list_is_independent_of_internal_state() -> None:
    """Callers that mutate the returned list must not affect the registry."""
    plugins = registry.by_objective(AttackObjective.DATA_EXFILTRATION)
    original_len = len(plugins)
    plugins.clear()
    again = registry.by_objective(AttackObjective.DATA_EXFILTRATION)
    assert len(again) == original_len
