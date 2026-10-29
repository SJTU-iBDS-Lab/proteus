"""CLI: Drive the outer red-team evolution loop.

Wraps :func:`red_team.evolution.driver.run_evolution` so the
full Algorithm 1 loop can be launched from the shell against a seed +
repo pair.

Usage
-----
    uv run python scripts/red_team/run_evolution.py \
        --seed seed-04-privesc-suid-shell \
        --repo slopbuster \
        --target-model glm-5 \
        --rounds 3 \
        --blue-team ours

Outputs (under ``<--out>/<run_id>/``)
-------------------------------------
* ``log.jsonl``       — one row per audited round plus S2/S3 expansions.
* ``attack_log.jsonl``— BM25-searchable attack log.
* ``summary.json``    — terminal roll-up (rounds_ran, first_bypass_round,
  total_bypass, total_lethal, total_cost_usd, ...).

Design notes
------------
* Seed + repo loading: a ``SkillVariant`` is built from the
  malicious-seeds row and a ``RepositorySlice`` is synthesised in-process
  from ``target_repos.jsonl``.  The driver handles all further
  materialisation, sandbox dispatch, blue-team audit, and horizontal
  expansion.
* ``--ablation-json`` is dual-mode: if the argument starts with ``{`` it
  is parsed as inline JSON; otherwise treated as a path to a JSON file;
  finally falls back to parsing as JSON for forgiving behaviour.
* The driver does not emit per-round callbacks.  To surface progress we
  print a hint to ``tail -f <log.jsonl>`` at start, let the driver run to
  completion, and then re-read ``log.jsonl`` to print a compact per-round
  summary line (``round=<t> bypass=<b> lethal=<l> cost=$<x> stop=<r>``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice  # noqa: E402
from core.schemas import (  # noqa: E402
    AttackObjective,
    Channel,
    SkillVariant,
    Topology,
)
from red_team.evolution.driver import run_evolution  # noqa: E402
from red_team.util.run_logging import bootstrap_from_ini  # noqa: E402

# Bootstrap human-readable progress logging.  Override the ini path via
# $LOG_INI; override the rotating-file destination via $LOG_FILE.  The
# runner *also* attaches a per-run ``run.log`` under each output dir.
bootstrap_from_ini(os.environ.get("LOG_INI"))
from red_team.skill_poisoner import (  # noqa: E402
    POISON_PAYLOADS,
    build_poisoned_prompt,
)


# --------------------------------------------------------------------- paths

DEFAULT_SEEDS_PATH = ROOT / "data" / "seeds" / "malicious_seeds.jsonl"
DEFAULT_TARGETS_PATH = ROOT / "data" / "targets" / "target_repos.jsonl"
DEFAULT_SKILLS_ROOT = ROOT / "data" / "skills" / "clawhub" / "skills"
_MODELS_YAML = ROOT / "configs" / "models.yaml"


def _model_default(*path: str) -> str:
    """Pull a default model id from configs/models.yaml (best-effort).

    Used as argparse default so the CLI matches the central config
    without forcing yaml-aware boot.  Empty string on any error means
    argparse will accept whatever the user passed via CLI.
    """
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
        for key in path:
            data = (data or {}).get(key)
        return str(data) if data else ""
    except Exception:
        return ""


# --------------------------------------------------------------------- loaders


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _load_seed_records(path: Path) -> list[dict[str, Any]]:
    """Load either the canonical JSONL corpus or a frozen JSON manifest."""
    if path.suffix.lower() != ".json":
        return _load_jsonl(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("seeds")
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(
            f"seed manifest {path} must be a JSON list or an object with a "
            "'seeds' list"
        )
    return payload


def _find_seed(seeds: list[dict[str, Any]], seed_id: str) -> dict[str, Any]:
    """Return the seed record matching ``seed_id`` on either ``seed_id`` or ``id``."""
    for row in seeds:
        if row.get("seed_id") == seed_id or row.get("id") == seed_id:
            return row
    raise KeyError(f"seed {seed_id!r} not found in {DEFAULT_SEEDS_PATH}")


def _find_repo(repos: list[dict[str, Any]], repo_id: str) -> dict[str, Any]:
    """Return the repo row matching ``repo_id``."""
    for row in repos:
        if row.get("repo_id") == repo_id:
            return row
    raise KeyError(f"repo {repo_id!r} not found in {DEFAULT_TARGETS_PATH}")


# --------------------------------------------------------------------- builders


def _objective_from_seed(seed: dict[str, Any]) -> AttackObjective:
    """Resolve the paper-canonical objective from a seed record."""
    canonical = seed.get("o")
    if isinstance(canonical, str) and canonical:
        return AttackObjective(canonical)
    legacy = seed.get("attack_type") or seed.get("primary_attack_type")
    if isinstance(legacy, str) and legacy:
        return AttackObjective.from_payload_key(legacy)
    raise ValueError(
        "seed is missing an objective: expected 'o' or 'attack_type' / "
        "'primary_attack_type'"
    )


def _payload_key_for_poison(seed: dict[str, Any], o: AttackObjective) -> str:
    """Return the POISON_PAYLOADS key used to poison the seed baseline."""
    payload = seed.get("payload_template")
    if isinstance(payload, str) and payload in POISON_PAYLOADS:
        return payload
    return o.value


def _build_seed_variant(seed: dict[str, Any]) -> SkillVariant:
    """Instantiate a baseline :class:`SkillVariant` from a seed record.

    This entry point does *not* materialise a poisoned chain on disk —
    the driver handles that inside its sandbox dispatch.  We set ``d``
    to the poisoned prompt text as a lightweight baseline so the variant
    is self-describing for logging.
    """
    o = _objective_from_seed(seed)
    tau = Topology(seed["tau"])
    chain = list(seed.get("chain") or [])
    if not chain:
        raise ValueError(f"seed {seed.get('seed_id')!r} has empty chain")

    payload_key = _payload_key_for_poison(seed, o)
    baseline_prompt = build_poisoned_prompt(payload_key)

    channels = [Channel(k) for k in seed.get("kappa") or []]
    metadata = dict(seed.get("metadata") or {})
    metadata.setdefault("seed_id", seed.get("seed_id") or seed.get("id") or "")
    metadata.setdefault("payload_template", payload_key)

    poison_index = int(seed.get("poison_index", -1))

    return SkillVariant(
        o=o,
        tau=tau,
        c=dict(seed.get("c") or {}),
        kappa=channels,
        d=baseline_prompt,
        poison_index=poison_index,
        chain=chain,
        metadata=metadata,
    )


def _build_repo_slice(row: dict[str, Any]) -> RepositorySlice:
    """Construct a :class:`RepositorySlice` from a target_repos.jsonl row."""
    repo_id = row["repo_id"]
    readme_text = str(row.get("readme_md") or "")
    file_tree = list(row.get("file_tree") or [])
    theme = str(row.get("theme_summary") or "")
    themes = sorted({tok for tok in theme.lower().split() if tok}) or [
        t for t in (row.get("domain", "").lower().split("/")) if t
    ]
    return RepositorySlice(
        repo_id=repo_id,
        repo_root=f"data/targets/{repo_id}",
        readme_path="README.md" if readme_text else None,
        readme_content=readme_text,
        file_tree=file_tree,
        dependency_files=[
            f for f in file_tree if Path(f).name in {
                "package.json", "pyproject.toml", "requirements.txt",
                "poetry.lock", "uv.lock", "Pipfile", "Cargo.toml", "go.mod",
            }
        ],
        labels={},
        metadata={
            "file_count": len(file_tree),
            "domain": row.get("domain", ""),
            "one_liner": row.get("one_liner", ""),
            "theme_summary": theme,
        },
        extracted_themes=themes,
    )


def _resolve_repo_slice(
    rows: list[dict[str, Any]], repo_id: str
) -> RepositorySlice:
    """Resolve a legacy repo card, falling back to the downloaded skill."""
    try:
        return _build_repo_slice(_find_repo(rows, repo_id))
    except KeyError:
        skill_root = DEFAULT_SKILLS_ROOT / repo_id
        if not skill_root.is_dir():
            raise
        skill_md = skill_root / "SKILL.md"
        readme = (
            skill_md.read_text(encoding="utf-8", errors="replace")
            if skill_md.is_file()
            else ""
        )
        inventory = sorted(
            str(path.relative_to(skill_root))
            for path in skill_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )
        return RepositorySlice(
            repo_id=repo_id,
            repo_root=str(skill_root),
            readme_path="SKILL.md" if skill_md.is_file() else None,
            readme_content=readme,
            file_tree=inventory,
            dependency_files=[
                path for path in inventory if Path(path).name in {
                    "package.json", "pyproject.toml", "requirements.txt",
                    "poetry.lock", "uv.lock", "Pipfile", "Cargo.toml",
                    "go.mod",
                }
            ],
            labels={},
            metadata={"file_count": len(inventory), "source": "clawhub_skill"},
            extracted_themes=sorted(
                {token for token in repo_id.lower().replace("-", " ").split()}
            ),
        )


def _parse_ablation(spec: str | None) -> dict[str, Any]:
    """Parse ``--ablation-json`` as inline JSON string or a path to a JSON file.

    Dual-mode: if ``spec`` starts with ``{``, parse as literal JSON; else
    try a filesystem path; else last-resort parse as JSON.
    """
    if not spec:
        return {}
    stripped = spec.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        data = json.loads(stripped)
    else:
        candidate = Path(stripped)
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("ablation must decode to a JSON object")
    return data


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the outer red-team evolution loop (Algorithm 1) against a "
            "(seed, repo) pair. Wraps red_team.evolution.driver.run_evolution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ablation JSON keys (passed via --ablation-json):\n"
            "  disable_tool             - disable a specific mutator tool by name\n"
            "  tool_use_off             - skip the LLM loop, use RandomMutation baseline\n"
            "  disable_expansion        - skip horizontal (S2/S3) expansion on bypass+lethal\n"
            "  search_space             - override search space (e.g. 'evasion')\n"
            "  force_topology           - e.g. 'single' to truncate chain to 1 skill\n"
            "  forbid_channel_edit      - forbid Channel/kappa edits in mutator\n"
            "  feedback                 - 'scalar' hides findings from the mutator\n"
            "  memory_enabled           - toggle BM25 attack-log memory\n"
            "  disable_family_enumeration - skip family enumeration in S3 surface expansion\n"
        ),
    )
    parser.add_argument(
        "--seed",
        required=True,
        help="Seed ID from malicious_seeds.jsonl (required).",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repo ID from target_repos.jsonl (required).",
    )
    parser.add_argument(
        "--target-model",
        default=_model_default("sandbox", "target") or "gpt-5.4-mini",
        help="Target LLM model id (default: configs/models.yaml sandbox.target).",
    )
    parser.add_argument(
        "--mutator-model",
        default=_model_default("red_team", "mutator") or "deepseek-v4-pro",
        help="Mutator LLM model id (default: configs/models.yaml red_team.mutator).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of S1 rounds to run (default: 20).",
    )
    parser.add_argument(
        "--free-expansions",
        type=int,
        default=3,
        help="Free expansion budget for S2 path expansion (default: 3).",
    )
    parser.add_argument(
        "--surface-k",
        type=int,
        default=10,
        help="S3 surface expansion candidate count k (default: 10).",
    )
    parser.add_argument(
        "--surface-k-prime",
        type=int,
        default=3,
        help="S3 surface expansion executed count k' (default: 3).",
    )
    parser.add_argument(
        "--blue-team",
        choices=("ours", "aig", "skillvetter", "llm_zeroshot"),
        default="ours",
        help="Blue-team auditor to use (default: ours).",
    )
    parser.add_argument(
        "--sandbox-container",
        default="skillrt-openclaw-00",
        help="Sandbox container name (default: skillrt-openclaw-00).",
    )
    parser.add_argument(
        "--ablation-json",
        default=None,
        help=(
            "Inline JSON string OR path to JSON file describing ablation "
            "flags. See --help epilog for the supported keys."
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/evolution",
        help="Output directory (default: outputs/evolution).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: auto-generated from seed+repo+model).",
    )
    parser.add_argument(
        "--seeds-file",
        default=str(DEFAULT_SEEDS_PATH),
        help="Override path to malicious_seeds.jsonl.",
    )
    parser.add_argument(
        "--targets-file",
        default=str(DEFAULT_TARGETS_PATH),
        help="Override path to target_repos.jsonl.",
    )
    return parser


# --------------------------------------------------------------------- main


def _iter_log_rows(log_path: Path) -> list[dict[str, Any]]:
    """Read back ``log.jsonl`` as a list of dicts (best-effort)."""
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def main(argv: list[str] | None = None) -> int:
    # Enable strict chain-id existence check in SearchSpace.validate —
    # any S2/S3 chain rewrite referencing a missing ClawHub skill fails
    # fast at the mutator boundary. Set inside main() rather than at
    # module import so unit tests that just import this file do not
    # contaminate the global env.
    os.environ.setdefault("CLAW_VALIDATE_CHAIN_IDS", "1")

    args = build_parser().parse_args(argv)

    seeds_path = Path(args.seeds_file)
    targets_path = Path(args.targets_file)

    if not seeds_path.is_file():
        print(f"error: seeds file not found: {seeds_path}", file=sys.stderr)
        return 2
    if not targets_path.is_file():
        print(f"error: targets file not found: {targets_path}", file=sys.stderr)
        return 2

    seeds = _load_seed_records(seeds_path)
    repos = _load_jsonl(targets_path)

    try:
        seed_row = _find_seed(seeds, args.seed)
        repo = _resolve_repo_slice(repos, args.repo)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    variant = _build_seed_variant(seed_row)
    ablation = _parse_ablation(args.ablation_json)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Emit a tail hint before the (potentially long) driver call; users can
    # follow per-round progress in another shell via ``tail -f <log.jsonl>``.
    tentative_run_id = args.run_id or "<auto-generated>"
    print(
        f"[run_evolution] starting: seed={args.seed} repo={args.repo} "
        f"rounds={args.rounds} blue_team={args.blue_team} "
        f"target_model={args.target_model}"
    )
    print(
        f"[run_evolution] progress hint: "
        f"tail -f {out_root}/{tentative_run_id}/log.jsonl"
    )

    run_dir = run_evolution(
        variant,
        repo,
        target_model=args.target_model,
        mutator_model=args.mutator_model,
        rounds=args.rounds,
        free_expansions=args.free_expansions,
        surface_k=args.surface_k,
        surface_k_prime=args.surface_k_prime,
        ablation=ablation,
        blue_team=args.blue_team,
        sandbox_container=args.sandbox_container,
        run_id=args.run_id,
        output_root=out_root,
    )

    log_path = run_dir / "log.jsonl"
    summary_path = run_dir / "summary.json"

    # -- per-round summary lines ---------------------------------------------

    rows = _iter_log_rows(log_path)
    for row in rows:
        if row.get("is_expansion"):
            continue
        r = row.get("round")
        b = bool(row.get("bypassed"))
        le = bool(row.get("lethal"))
        cost = float(row.get("cost_usd") or 0.0)
        stop = row.get("stop_reason") or ""
        print(
            f"round={r} bypass={str(b).lower()} "
            f"lethal={str(le).lower()} cost=${cost:.4f} stop={stop}"
        )

    # -- summary -------------------------------------------------------------

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    print(f"log_path     : {log_path}")
    print(f"summary_path : {summary_path}")
    print(
        f"rounds_ran={summary.get('rounds_ran')} "
        f"first_bypass_round={summary.get('first_bypass_round')}"
    )
    print(
        f"total_bypass={summary.get('total_bypass')} "
        f"total_lethal={summary.get('total_lethal')} "
        f"total_cost_usd={summary.get('total_cost_usd')}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
