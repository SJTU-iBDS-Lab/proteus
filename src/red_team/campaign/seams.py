"""Injection seams for the campaign runner.

Every call-out to a heavy dependency (LLM agent, sandbox, blue team,
oracle, horizontal expansion) is reached through a callable seam so
unit tests can substitute stubs without touching docker / the network.

The five seams consumed by :mod:`red_team.campaign.runner` are:

* ``default_agent_run``          — :func:`red_team.agent.runtime.agent_run`
* ``default_horizontal_expand``  — :func:`red_team.evolution.horizontal_expand.horizontal_expand`
* ``default_oracle_lethal``      — :func:`red_team.oracle.lethality.lethal`
* ``call_blue_team``             — dispatch for the blue-team audit method
* ``sandbox_run``                — dispatch for the sandbox executor

Heavy imports (``blue_team.orchestrator`` / ``sandbox.openclaw_runner``)
are lazy so importing ``red_team.campaign`` does not pull docker into
the Python process.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.agent.runtime import agent_run as default_agent_run
from red_team.evolution.horizontal_expand import (
    horizontal_expand as default_horizontal_expand,
)
from red_team.oracle.lethality import lethal as default_oracle_lethal
from sandbox.honeypot import HoneypotResult

__all__ = [
    "default_agent_run",
    "default_horizontal_expand",
    "default_oracle_lethal",
    "call_blue_team",
    "sandbox_run",
    "load_sandbox_cfg",
    "DEFAULT_CHAIN_ROOT",
    "DEFAULT_SKILL_ROOT",
    "DEFAULT_SANDBOX_CFG",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_CHAIN_ROOT = Path("/tmp/poisoned_chains")
DEFAULT_SKILL_ROOT = Path("data/skills/clawhub/skills")
DEFAULT_SANDBOX_CFG = Path("configs/sandbox.yaml")


# ---------------------------------------------------------------------------
# Sandbox config loader
# ---------------------------------------------------------------------------


def load_sandbox_cfg(
    container: str,
    cfg_path: Path = DEFAULT_SANDBOX_CFG,
) -> Any:
    """Build a :class:`sandbox.openclaw_runner.SandboxConfig` from YAML."""
    from sandbox.openclaw_runner import SandboxConfig  # type: ignore

    data: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                data = parsed
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("failed to parse %s (%s); using defaults", cfg_path, exc)

    return SandboxConfig(
        container_name=container or str(data.get("container_name", "skillrt-openclaw")),
        image=str(data.get("image", "openclaw-eval:latest")),
        timeout=int(data.get("timeout", 300)),
    )


# ---------------------------------------------------------------------------
# Blue-team dispatch
# ---------------------------------------------------------------------------


def call_blue_team(
    variant: SkillVariant,
    repo: RepositorySlice,
    method: str,
    blue_team_fn: Callable[..., list[PaperFinding]] | None,
    *,
    sandbox_container: str = "skillrt-openclaw-00",
) -> list[PaperFinding]:
    """Return ``F_t`` for the variant under the selected blue-team method.

    Precedence:

    1. Explicit ``blue_team_fn`` seam wins (unit tests).
    2. ``method == "ours"`` calls the CLAW blue-team orchestrator.
    3. ``method == "skillvetter"`` runs the vetter baseline in OpenClaw.
    4. ``aig`` / ``llm_zeroshot`` raise :class:`NotImplementedError`.
    """
    if blue_team_fn is not None:
        return list(blue_team_fn(variant, repo))

    if method == "ours":
        try:
            from blue_team.orchestrator.blue_orchestrator import audit  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive import
            _LOGGER.warning(
                "blue_team 'ours' import failed (%s); returning empty findings",
                exc,
            )
            return []
        try:
            result = audit(variant, repo)
            return list(getattr(result, "findings", []) or [])
        except Exception as exc:  # pragma: no cover - defensive runtime
            _LOGGER.warning(
                "blue_team 'ours' audit raised %s; returning empty findings",
                exc,
            )
            return []

    if method == "skillvetter":
        return _run_skill_vetter_blue_team(variant, sandbox_container)

    if method == "aig":
        return _run_aig_blue_team(variant)

    if method == "llm_zeroshot":
        raise NotImplementedError(f"baseline {method!r} not wired yet")

    raise ValueError(f"unknown blue_team method: {method!r}")


def _run_skill_vetter_blue_team(
    variant: SkillVariant,
    sandbox_container: str,
) -> list[PaperFinding]:
    """SkillVetter baseline audit — materialize chain + convert verdicts."""
    from core.schemas.paper_finding import Evidence
    from red_team.skill_poisoner import materialize_poisoned_chain
    from sandbox.skill_vetter_runner import run_skill_vetter_audit

    chain = list(variant.chain or [])
    if not chain:
        return []

    chain_src_paths = [DEFAULT_SKILL_ROOT / sid for sid in chain]
    for p in chain_src_paths:
        if not p.is_dir():
            _LOGGER.warning(
                "skillvetter: chain skill %r not found on host; skipping audit",
                p,
            )
            return []

    tmp_root = Path(tempfile.gettempdir()) / "skill_vetter_materialize"
    tmp_root.mkdir(parents=True, exist_ok=True)
    # Per-process subdir prevents concurrent runs (same chain hash from
    # different seeds) from racing on shutil.rmtree.  Without this,
    # 10-concurrent batches see ``OSError: [Errno 39] Directory not
    # empty`` when sibling procs write into the dir mid-rmtree.
    host_chain_dir = tmp_root / f"{os.getpid()}-{variant.content_hash()}"

    materialized, poisoned_md = materialize_poisoned_chain(
        chain_src_paths,
        variant.o.value,
        host_chain_dir,
        poison_index=variant.poison_index,
    )
    try:
        pidx = (
            variant.poison_index
            if variant.poison_index >= 0
            else len(chain) + variant.poison_index
        )
        pidx = max(0, min(len(chain) - 1, pidx))
        orig_md = DEFAULT_SKILL_ROOT / chain[pidx] / "SKILL.md"
        benign_body = (
            orig_md.read_text(encoding="utf-8", errors="replace")
            if orig_md.is_file()
            else ""
        )
        separator = (
            ""
            if benign_body.endswith("\n\n")
            else ("\n\n" if benign_body.endswith("\n") else "\n\n")
        )
        poisoned_md.write_text(
            benign_body + separator + (variant.d or "").lstrip("\n"),
            encoding="utf-8",
        )
        scripts = (variant.c or {}).get("scripts")
        if isinstance(scripts, dict):
            for rel, body in scripts.items():
                if not isinstance(rel, str) or not isinstance(body, str):
                    continue
                if rel.startswith("/") or ".." in rel.split("/") or rel == "SKILL.md":
                    continue
                target = materialized[pidx] / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("skillvetter: materialization overlay failed (%s)", exc)

    cfg = load_sandbox_cfg(sandbox_container)

    try:
        result = run_skill_vetter_audit(materialized, cfg)
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("skillvetter audit raised %s; returning empty findings", exc)
        return []

    findings: list[PaperFinding] = []
    for idx, v in enumerate(result.verdicts):
        verdict = v.get("verdict")
        if verdict not in {"REJECT", "CAUTION"}:
            continue
        severity = "high" if verdict == "REJECT" else "medium"
        flags = v.get("red_flags") or []
        evidence_snippet = "; ".join(flags)[:300] if flags else v.get("reasoning", "")[:300]
        findings.append(
            PaperFinding(
                finding_id=f"skillvetter:{idx:02d}",
                layer="skillvetter_baseline",
                category=f"skill_vet_{verdict.lower()}",
                reason_code=f"skillvetter_{verdict.lower()}",
                confidence=0.9,
                severity=severity,
                evidence=Evidence(
                    file=f"{v.get('skill','?')}/SKILL.md",
                    line_range=None,
                    snippet=evidence_snippet,
                ),
                explanation=v.get("reasoning", "")[:400],
                metadata={
                    "origin_skill": v.get("skill"),
                    "risk_level": v.get("risk_level"),
                    "red_flags": list(flags),
                },
            )
        )
    return findings


def _run_aig_blue_team(variant: SkillVariant) -> list[PaperFinding]:
    """AIG (AI-Infra-Guard) gray-box baseline audit.

    Materialises the chain on disk (so the poisoned content is what
    AIG actually sees), then runs ``gray_box.chain_scanner.scan_chain_async``
    over every skill in the chain.  AIG's ``ChainScanResult`` is
    converted to :class:`PaperFinding` entries with severity mapped
    from the AIG vulnerability levels (Critical/High → high,
    Medium → medium, others → low).  When AIG flags the chain as
    harmful overall (``is_harmful=True``) we emit a chain-level
    finding even if no individual skill scan produced vulnerabilities,
    so the bypass predicate fires correctly.

    AIG is host-side (no docker), so this seam ignores
    ``sandbox_container`` — chain dirs land in a tmp staging path
    distinct from the openclaw runner's deploy path.
    """
    import asyncio
    from core.schemas.paper_finding import Evidence
    from gray_box.chain_scanner import scan_chain_async
    from red_team.skill_poisoner import materialize_poisoned_chain

    chain = list(variant.chain or [])
    if not chain:
        return []

    chain_src_paths = [DEFAULT_SKILL_ROOT / sid for sid in chain]
    for p in chain_src_paths:
        if not p.is_dir():
            _LOGGER.warning(
                "aig: chain skill %r not found on host; skipping audit", p,
            )
            return []

    tmp_root = Path(tempfile.gettempdir()) / "aig_materialize"
    tmp_root.mkdir(parents=True, exist_ok=True)
    # Per-process subdir — prevents shutil.rmtree races between
    # concurrent runs sharing chain hash.  See skillvetter version above.
    host_chain_dir = tmp_root / f"{os.getpid()}-{variant.content_hash()}"

    materialized, poisoned_md = materialize_poisoned_chain(
        chain_src_paths,
        variant.o.value,
        host_chain_dir,
        poison_index=variant.poison_index,
    )
    # Overlay mutator's new_d/new_c.scripts on top of the poisoned
    # SKILL.md so AIG scans the exact bytes the target would receive
    # — same overlay routine as the SkillVetter path.
    try:
        pidx = (
            variant.poison_index
            if variant.poison_index >= 0
            else len(chain) + variant.poison_index
        )
        pidx = max(0, min(len(chain) - 1, pidx))
        if (variant.metadata or {}).get("doc_mode") == "full":
            # Full-document baselines: variant.d IS the entire SKILL.md —
            # write verbatim so AIG scans victim-identical bytes.
            poisoned_md.write_text(variant.d or "", encoding="utf-8")
        else:
            orig_md = DEFAULT_SKILL_ROOT / chain[pidx] / "SKILL.md"
            benign_body = (
                orig_md.read_text(encoding="utf-8", errors="replace")
                if orig_md.is_file()
                else ""
            )
            separator = (
                ""
                if benign_body.endswith("\n\n")
                else ("\n\n" if benign_body.endswith("\n") else "\n\n")
            )
            poisoned_md.write_text(
                benign_body + separator + (variant.d or "").lstrip("\n"),
                encoding="utf-8",
            )
        scripts = (variant.c or {}).get("scripts")
        if isinstance(scripts, dict):
            for rel, body in scripts.items():
                if not isinstance(rel, str) or not isinstance(body, str):
                    continue
                if rel.startswith("/") or ".." in rel.split("/") or rel == "SKILL.md":
                    continue
                target = materialized[pidx] / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("aig: materialization overlay failed (%s)", exc)

    try:
        result = asyncio.run(
            scan_chain_async(
                subgraph_id=variant.content_hash()[:12],
                skill_ids=[p.name for p in materialized],
                skills_root=host_chain_dir,
                primary_attack_type=variant.o.value,
                hop_length=len(chain),
                language="en",
                max_concurrent=len(chain),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("aig: chain scan raised %s; returning empty findings", exc)
        return []

    findings: list[PaperFinding] = []
    severity_map = {
        "critical": "high", "high": "high",
        "medium": "medium", "low": "low",
    }
    n_errored = 0
    for sk_idx, sk in enumerate(result.skill_results):
        if sk.error:
            n_errored += 1
            _LOGGER.warning(
                "aig: skill %s scan errored: %s",
                sk.skill_id, sk.error[:200],
            )
            continue
        for v_idx, v in enumerate(sk.vulnerabilities):
            sev = severity_map.get(v.level.lower(), "medium")
            findings.append(
                PaperFinding(
                    finding_id=f"aig:{sk_idx:02d}.{v_idx:02d}",
                    layer="aig_baseline",
                    category=f"aig_{v.risk_type or 'unknown'}".lower().replace(" ", "_"),
                    reason_code=f"aig_{v.level.lower()}",
                    confidence=0.9,
                    severity=sev,
                    evidence=Evidence(
                        file=f"{sk.skill_id}/SKILL.md",
                        line_range=None,
                        snippet=v.description[:300],
                    ),
                    explanation=v.suggestion[:300] if v.suggestion else v.description[:300],
                    metadata={
                        "origin_skill": sk.skill_id,
                        "aig_level": v.level,
                        "aig_risk_type": v.risk_type,
                        "aig_score": sk.score,
                    },
                )
            )
    # Scanner-error guard: if every per-skill scan errored, emit a
    # SCANNER_ERROR finding so the audit does NOT auto-approve a
    # broken scan.  Without this, a Python crash / rate-limit / OOM
    # in AIG would silently let the variant through (bypass=True),
    # which is the opposite of what we want for paper data.
    if n_errored == len(result.skill_results) and result.skill_results:
        first_err = next(
            (sk.error for sk in result.skill_results if sk.error),
            "AIG subprocess error",
        )
        findings.append(
            PaperFinding(
                finding_id="aig:scanner.error",
                layer="aig_baseline",
                category="aig_scanner_failed",
                reason_code="aig_subprocess_error",
                confidence=1.0,
                severity="high",
                evidence=Evidence(
                    file="<aig-scanner>",
                    line_range=None,
                    snippet=first_err[:300],
                ),
                explanation=(
                    "All AIG per-skill scans failed; treating as audit "
                    "REJECT to avoid silently approving the variant on a "
                    "scanner outage."
                ),
                metadata={"errored_scans": n_errored},
            )
        )
        return findings

    # Chain-level finding when AIG marks harmful but per-skill produced
    # no vuln (rare; usually means the chain analysis stage flagged
    # composition risk that individual skill scans missed).
    if result.is_harmful and not findings:
        findings.append(
            PaperFinding(
                finding_id="aig:chain.composite",
                layer="aig_baseline",
                category="aig_chain_harmful",
                reason_code="aig_chain_composite",
                confidence=0.85,
                severity="high",
                evidence=Evidence(
                    file="<chain>",
                    line_range=None,
                    snippet=result.chain_risk_summary[:300],
                ),
                explanation=" | ".join(result.harm_reasons)[:400],
                metadata={
                    "min_score": result.min_score,
                    "total_vulns": result.total_vulns,
                    "confidence_label": result.confidence,
                },
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Sandbox dispatch
# ---------------------------------------------------------------------------


def sandbox_run(
    variant: SkillVariant,
    chain: list[str],
    target_model: str,
    container: str,
    sandbox_run_fn: Callable[..., Any] | None,
    skill_root: Path = DEFAULT_SKILL_ROOT,
    poison_host_root: Path = DEFAULT_CHAIN_ROOT,
) -> tuple[HoneypotResult, list[dict[str, Any]], Path | None]:
    """Execute ``variant`` against the sandbox.

    Returns ``(honeypot, tool_calls, host_chain_dir)``.  The third field
    is the host path of the materialized poisoned chain directory, or
    ``None`` when a test seam overrode ``sandbox_run_fn``.
    """
    if sandbox_run_fn is not None:
        ret = sandbox_run_fn(variant, chain, target_model, container)
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret
        # Back-compat: old seam signature returned (hp, tool_calls).
        hp, tool_calls = ret
        return hp, tool_calls, None

    from sandbox.openclaw_chain_runner import run_poisoned_chain  # type: ignore

    cfg = load_sandbox_cfg(container)
    chain_paths = [skill_root / sid for sid in chain]
    raw_scripts = (variant.c or {}).get("scripts")
    poisoned_scripts: dict[str, str] | None = None
    if isinstance(raw_scripts, dict) and raw_scripts:
        poisoned_scripts = {
            str(k): str(v)
            for k, v in raw_scripts.items()
            if isinstance(k, str) and isinstance(v, str) and v
        }
        if not poisoned_scripts:
            poisoned_scripts = None
    poisoned = run_poisoned_chain(
        chain_paths,
        cfg,
        subgraph_id=variant.content_hash(),
        primary_attack_type=variant.o.value,
        poison_attack_type=variant.o.value,
        poisoned_host_root=poison_host_root,
        poisoned_skill_md=variant.d,
        poisoned_skill_md_mode=str(
            (variant.metadata or {}).get("doc_mode", "append")
        ),
        poisoned_scripts=poisoned_scripts,
        target_model=target_model,
    )
    host_chain_dir_str = (poisoned.metadata or {}).get("host_chain_dir")
    host_chain_dir = Path(str(host_chain_dir_str)) if host_chain_dir_str else None
    return poisoned.honeypot, list(poisoned.tool_calls or []), host_chain_dir
