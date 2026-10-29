"""Idempotent yaml-seed bootstrap (Medium plan §3 + §5).

Walks ``data/memory/seeds/{pos,chokepoint}/*.yaml`` and upserts each
file as a :class:`MemoryUnit` into the supplied :class:`Store`.

Idempotency is enforced via ``data/memory/seeds.lock`` — one JSON line
per committed file holding ``{path, sha256, unit_id}``.  A subsequent
run sees the same hash and skips the file.

The directory name supplies the ``category`` field (``pos`` →
``positive``; ``chokepoint`` → ``chokepoint``); the yaml itself must
declare the remaining MemoryUnit fields.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from red_team.log.memory.store import Store
from red_team.log.memory.unit import MemoryUnit

__all__ = ["load_yaml_seeds", "DEFAULT_SEEDS_ROOT", "DEFAULT_LOCK_PATH"]


DEFAULT_SEEDS_ROOT = Path("data/memory/seeds")
DEFAULT_LOCK_PATH = Path("data/memory/seeds.lock")

# Medium plan: only two on-disk categories.  The legacy `audit_chokepoint`
# directory is migrated by `git mv` to `chokepoint`.
_CATEGORY_DIRS: dict[str, str] = {
    "pos": "positive",
    "chokepoint": "chokepoint",
}


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _read_lock(lock_path: Path) -> dict[str, dict[str, str]]:
    if not lock_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["path"]] = rec
    return out


def _append_lock(lock_path: Path, record: dict[str, str]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _yaml_to_unit(
    raw: dict[str, Any], *, category: str, source_path: Path
) -> MemoryUnit:
    overrides = dict(raw)
    overrides.setdefault("category", category)
    overrides.setdefault("source", "seed")
    overrides.setdefault(
        "created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    overrides.setdefault("authored_by", f"yaml:{source_path.name}")
    overrides.setdefault("extractor_version", "yaml-seed")
    overrides.setdefault("evolution_step", 0)
    overrides.setdefault("superseded_by", None)
    overrides.setdefault("blue_reason_codes", [])
    overrides.setdefault("tags", [])
    overrides.setdefault("preconditions", [])
    overrides.setdefault("kappa", [])
    overrides.setdefault("chain", [])
    overrides.setdefault("source_run_id", None)
    overrides.setdefault("source_seed_id", None)
    overrides.setdefault("variant_hash", None)
    overrides.setdefault("technique_name", "")
    overrides.setdefault("mechanism", "")
    overrides.setdefault("exploited_assumption", "")
    overrides.setdefault("target_model", "")
    overrides.setdefault("bypass", category == "positive")
    overrides.setdefault("lethal", category == "positive")
    return MemoryUnit.from_dict(overrides)


def load_yaml_seeds(
    store: Store,
    *,
    seeds_root: Path | str = DEFAULT_SEEDS_ROOT,
    lock_path: Path | str = DEFAULT_LOCK_PATH,
    extra_yaml_paths: Iterable[Path] = (),
) -> tuple[int, int]:
    """Walk seeds_root and upsert any yaml not already in seeds.lock.

    Returns ``(committed, skipped)``.
    """
    root = Path(seeds_root)
    lock = Path(lock_path)
    state = _read_lock(lock)
    committed = 0
    skipped = 0
    paths: list[Path] = list(extra_yaml_paths)
    if root.exists():
        for sub, _category in _CATEGORY_DIRS.items():
            sub_dir = root / sub
            if not sub_dir.exists():
                continue
            for p in sorted(sub_dir.glob("*.yaml")):
                paths.append(p)

    for p in paths:
        category = _CATEGORY_DIRS.get(p.parent.name, "")
        rel_key = str(p)
        blob = p.read_bytes()
        digest = _sha256(blob)
        prior = state.get(rel_key)
        if prior and prior.get("sha256") == digest:
            skipped += 1
            continue
        raw = yaml.safe_load(blob)
        if not isinstance(raw, dict):
            raise ValueError(f"yaml seed {p} did not parse to a dict")
        unit = _yaml_to_unit(raw, category=category, source_path=p)
        store.upsert_unit(unit)
        _append_lock(
            lock,
            {"path": rel_key, "sha256": digest, "unit_id": unit.unit_id},
        )
        committed += 1
    return committed, skipped
