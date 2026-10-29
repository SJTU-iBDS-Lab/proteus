"""Unit tests for :mod:`red_team.evolution.search_space`.

Covers Paper §4.3 / Algorithm 1:

* S1 mask = {mutable: [d], frozen: [o, tau, c, kappa, chain]}; only ``d`` may
  change between parent and child.
* S2 mask = {mutable: [c, kappa, d, chain], frozen: [o, tau]}; ``c`` and
  ``kappa`` may change while ``(o, tau)`` stay fixed.
* S3 mask = everything mutable, gated by :func:`shared_motif` (≥ 2 motif
  axes must overlap between parent and child).
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas.attack_objective import AttackObjective, Channel, Topology
from core.schemas.skill_variant import SkillVariant
from red_team.evolution.search_space import (
    InvariantViolation,
    SearchSpace,
    shared_motif,
)


def make_parent(**overrides) -> SkillVariant:
    """Canonical S1-test parent: single-skill PERSISTENCE attack."""
    base = dict(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={
            "entry_script": "run.sh",
            "deps": ["requests"],
            "declared_apis": ["http.get"],
        },
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="# original skill doc",
        chain=["skill-alpha"],
        metadata={
            "tool_chain_prefix": ["read_file", "exec"],
            "file_access": ["/home/user/.bashrc"],
        },
    )
    base.update(overrides)
    return SkillVariant(**base)


def clone_with(parent: SkillVariant, **overrides) -> SkillVariant:
    """Return a deep-copied clone of ``parent`` with requested field changes."""
    new_c = copy.deepcopy(parent.c)
    new_kappa = list(parent.kappa)
    new_chain = list(parent.chain)
    new_metadata = copy.deepcopy(parent.metadata)
    kwargs = dict(
        o=parent.o,
        tau=parent.tau,
        c=new_c,
        kappa=new_kappa,
        d=parent.d,
        chain=new_chain,
        metadata=new_metadata,
        poison_index=parent.poison_index,
    )
    kwargs.update(overrides)
    return SkillVariant(**kwargs)


# --------------------------------------------------------------------- mask


class MaskTests(unittest.TestCase):
    def test_s1_mask(self) -> None:
        m = SearchSpace("S1").mask()
        self.assertEqual(m["mutable"], ["d"])
        self.assertEqual(m["frozen"], ["o", "tau", "c", "kappa", "chain"])

    def test_s2_mask(self) -> None:
        m = SearchSpace("S2").mask()
        self.assertEqual(m["mutable"], ["c", "kappa", "d", "chain"])
        self.assertEqual(m["frozen"], ["o", "tau"])

    def test_s3_mask(self) -> None:
        m = SearchSpace("S3").mask()
        self.assertEqual(
            m["mutable"],
            ["o", "tau", "c", "kappa", "d", "chain"],
        )
        self.assertEqual(m["frozen"], [])

    def test_mask_returns_fresh_lists(self) -> None:
        space = SearchSpace("S1")
        m1 = space.mask()
        m1["mutable"].append("hacked")
        m2 = space.mask()
        self.assertEqual(m2["mutable"], ["d"])

    def test_invalid_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchSpace("S4")


# ----------------------------------------------------------------------- S1


class S1ValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("S1")

    def test_pass_change_only_d(self) -> None:
        child = clone_with(self.parent, d="# mutated doc — add benign polish")
        # Must not raise.
        self.space.validate(self.parent, child)

    def test_fail_change_entry_script(self) -> None:
        mutated_c = copy.deepcopy(self.parent.c)
        mutated_c["entry_script"] = "pwned.sh"
        child = clone_with(self.parent, c=mutated_c)
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)

    def test_fail_change_kappa_list(self) -> None:
        child = clone_with(self.parent, kappa=[Channel.NETWORK])
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)

    def test_fail_change_objective(self) -> None:
        child = clone_with(self.parent, o=AttackObjective.DATA_EXFILTRATION)
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)

    def test_fail_change_chain(self) -> None:
        child = clone_with(self.parent, chain=["skill-beta"])
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


# ----------------------------------------------------------------------- S2


class S2ValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("S2")

    def test_pass_change_c_and_kappa(self) -> None:
        mutated_c = copy.deepcopy(self.parent.c)
        mutated_c["entry_script"] = "alt_entry.sh"
        mutated_c["deps"] = ["httpx"]
        child = clone_with(
            self.parent,
            c=mutated_c,
            kappa=[Channel.NETWORK],
            d="# new path expansion doc",
        )
        self.space.validate(self.parent, child)  # no raise

    def test_fail_change_tau_pair(self) -> None:
        # Moving from SINGLE → PAIR requires chain length 2.
        child = clone_with(
            self.parent,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-beta"],
        )
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)

    def test_fail_change_objective(self) -> None:
        child = clone_with(self.parent, o=AttackObjective.CLEANUP)
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


# ----------------------------------------------------------------------- S3


class S3ValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("S3")

    def test_pass_change_o_and_tau_with_kappa_and_entry_script_motif(self) -> None:
        """S3 legitimate transfer: change o and tau, keep κ overlap + same entry_script.

        This exercises the metadata-less fallback: we strip
        ``tool_chain_prefix`` / ``file_access`` so ``shared_motif`` falls
        back to the (κ-overlap + entry_script) heuristic.
        """
        mutated_c = copy.deepcopy(self.parent.c)  # entry_script retained
        child = clone_with(
            self.parent,
            o=AttackObjective.DATA_EXFILTRATION,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-gamma"],
            # Keep at least one channel in common:
            kappa=[Channel.FILESYSTEM, Channel.NETWORK],
            c=mutated_c,
            metadata={},  # force fallback path
        )
        # Also clear parent metadata for this test so only fallback applies.
        parent_no_meta = clone_with(self.parent, metadata={})
        self.space.validate(parent_no_meta, child)  # no raise

    def test_pass_with_metadata_kappa_and_tool_chain_prefix(self) -> None:
        """Two axes via metadata: κ overlap + tool_chain_prefix overlap."""
        child = clone_with(
            self.parent,
            o=AttackObjective.SUPPLY_CHAIN_ABUSE,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-delta"],
            kappa=[Channel.SUBPROCESS, Channel.NETWORK],  # overlaps SUBPROCESS
            c={"entry_script": "totally_different.py"},
            metadata={
                # Shorter list is prefix of longer → prefix overlap holds.
                "tool_chain_prefix": ["read_file", "exec", "curl"],
                "file_access": ["/tmp/unrelated"],
            },
        )
        self.space.validate(self.parent, child)  # no raise

    def test_fail_no_motif_overlap(self) -> None:
        """Change everything with zero motif overlap → raise."""
        child = clone_with(
            self.parent,
            o=AttackObjective.GUIDANCE_INJECTION,
            tau=Topology.PAIR,
            chain=["skill-x", "skill-y"],
            kappa=[Channel.MEMORY],  # no overlap with parent {FS, SUBPROCESS}
            c={"entry_script": "brand_new.py"},
            d="# fully unrelated doc",
            metadata={
                "tool_chain_prefix": ["completely_different_tool"],
                "file_access": ["/var/log/unrelated.log"],
            },
        )
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


# ------------------------------------------------------------- shared_motif


class SharedMotifTests(unittest.TestCase):
    def test_fallback_same_entry_script_plus_kappa_overlap(self) -> None:
        parent = make_parent(metadata={})
        child = clone_with(parent, o=AttackObjective.CLEANUP, metadata={})
        # Fallback path: kappa overlap (1) + same entry_script (1) → 2.
        self.assertTrue(shared_motif(parent, child))

    def test_fallback_different_entry_script_only_kappa(self) -> None:
        parent = make_parent(metadata={})
        mutated_c = copy.deepcopy(parent.c)
        mutated_c["entry_script"] = "other.py"
        child = clone_with(parent, c=mutated_c, metadata={})
        # Fallback: only kappa (1) — insufficient.
        self.assertFalse(shared_motif(parent, child))

    def test_metadata_tool_chain_prefix_and_file_access(self) -> None:
        parent = make_parent()
        child = clone_with(
            parent,
            kappa=[Channel.MEMORY],  # no overlap with parent
            c={"entry_script": "nope.py"},
            metadata={
                "tool_chain_prefix": ["read_file", "exec", "curl"],
                "file_access": ["/home/user/.bashrc", "/etc/passwd"],
            },
        )
        # Axis 2 (tool_chain_prefix) + Axis 3 (file_access) → 2 axes.
        self.assertTrue(shared_motif(parent, child))

    def test_metadata_only_one_axis_fails(self) -> None:
        parent = make_parent()
        child = clone_with(
            parent,
            kappa=[Channel.MEMORY],
            c={"entry_script": "nope.py"},
            metadata={
                "tool_chain_prefix": ["something_else"],
                "file_access": ["/home/user/.bashrc"],  # only axis 3
            },
        )
        self.assertFalse(shared_motif(parent, child))


if __name__ == "__main__":
    unittest.main()
