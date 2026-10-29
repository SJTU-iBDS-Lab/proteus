"""Phase 0 seed corpus sanity tests.

These tests are split into two tiers:
  * Schema tests — always run; validate that every seed record parses
    into a valid :class:`SkillVariant` and points at a skill dir that
    exists on disk.
  * Lethality tests — require docker + OpenClaw; validate that seeds
    marked ``lethal_baseline.verified`` really are lethal under the
    current honeypot. Guarded by ``CLAW_DOCKER=1`` so they don't block
    CI by default.

The full Phase 0 deliverable is: every seed with
``lethal_baseline.verified = false`` must either become true after
running the lethality tier, or be removed from the corpus.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective, SkillVariant, Topology  # noqa: E402

SEED_PATH = ROOT / "data" / "seeds" / "malicious_seeds.jsonl"
TARGETS_PATH = ROOT / "data" / "targets" / "target_repos.jsonl"
SKILLS_ROOT = ROOT / "data" / "skills" / "clawhub" / "skills"

PAPER_CATEGORIES = {o.value for o in AttackObjective}


# --- Schema / corpus tier -------------------------------------------------

def _load_seeds() -> list[dict]:
    return [json.loads(l) for l in SEED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_targets() -> list[dict]:
    return [json.loads(l) for l in TARGETS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_seed_file_exists_and_has_ten_entries():
    seeds = _load_seeds()
    assert len(seeds) == 10, f"expected 10 seeds, got {len(seeds)}"


def test_targets_file_exists_and_has_ten_entries():
    targets = _load_targets()
    assert len(targets) == 10, f"expected 10 target repos, got {len(targets)}"


def test_seeds_cover_all_paper_categories():
    seeds = _load_seeds()
    cats = {s["metadata"]["paper_category"] for s in seeds}
    assert cats == PAPER_CATEGORIES, (
        f"missing categories: {PAPER_CATEGORIES - cats}; "
        f"unknown: {cats - PAPER_CATEGORIES}"
    )


def test_seed_ids_are_unique():
    seeds = _load_seeds()
    ids = [s["seed_id"] for s in seeds]
    assert len(ids) == len(set(ids)), "duplicate seed_id"


@pytest.mark.parametrize("record", _load_seeds(), ids=lambda r: r["seed_id"])
def test_seed_parses_as_skill_variant(record):
    # Every seed must satisfy SkillVariant schema (topology vs chain length
    # constraint, valid enum values).
    SkillVariant(
        o=record["o"],
        tau=record["tau"],
        c=record.get("c", {}),
        kappa=record.get("kappa", []),
        d=record.get("d", ""),
        poison_index=record.get("poison_index", -1),
        chain=record["chain"],
        metadata=record.get("metadata", {}),
    )


@pytest.mark.parametrize("record", _load_seeds(), ids=lambda r: r["seed_id"])
def test_seed_chain_skills_exist_on_disk(record):
    for skill_id in record["chain"]:
        p = SKILLS_ROOT / skill_id
        assert p.is_dir(), f"missing skill dir: {p}"


@pytest.mark.parametrize("record", _load_targets(), ids=lambda r: r["repo_id"])
def test_target_repo_has_minimum_context(record):
    assert record["readme_md"], f"{record['repo_id']} has empty readme"
    assert record["file_tree"], f"{record['repo_id']} has empty file tree"
    assert record["theme_summary"], f"{record['repo_id']} has empty theme summary"


# --- Lethality tier (docker required) -------------------------------------

def _needs_docker():
    return pytest.mark.skipif(
        os.environ.get("CLAW_DOCKER") != "1",
        reason="set CLAW_DOCKER=1 to run docker-dependent lethality tests",
    )


@_needs_docker()
@pytest.mark.parametrize(
    "record",
    [s for s in _load_seeds() if s["lethal_baseline"]["verified"]],
    ids=lambda r: r["seed_id"],
)
def test_verified_seed_is_lethal_in_docker(record):
    """Re-run each seed marked verified and confirm ``attack_succeeded``."""
    from sandbox.openclaw_chain_runner import run_poisoned_chain
    from sandbox.openclaw_runner import SandboxConfig

    cfg = SandboxConfig(container_name="skillrt-openclaw", timeout=180)
    skill_paths = [SKILLS_ROOT / s for s in record["chain"]]
    result = run_poisoned_chain(
        skill_paths, cfg,
        subgraph_id=f"phase0_seed:{record['seed_id']}",
        primary_attack_type=record["o"],
        poison_attack_type=record["payload_template"],
        poisoned_host_root=Path("data/skills/poisoned"),
    )
    assert result.attack_succeeded, (
        f"seed {record['seed_id']} not lethal — evidence: "
        f"{result.honeypot.evidence[:5]}"
    )
