"""Orchestrate extractors, scanners, probes, and report assembly."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import (
    AuditReport,
    RepositorySlice,
    RuntimeTraceSummary,
    SkillArtifact,
    StaticFinding,
    StructuredFindings,
)
from core.schemas import PaperFinding, SkillVariant
from core.schemas.paper_finding import Evidence

from ..decorators import FindingDecorator
from ..extractors import CodeExtractor, DocExtractor, RepoSliceExtractor
from ..extractors.llm_code_extractor import refine_skill_tags
from ..probes import DynamicProbeRunner
from ..scanners import (
    ChainRiskPrecheckScanner,
    GuidanceInjectionScanner,
    RepoContextScanner,
    SemanticGapScanner,
)
from ..verdict import AuditReportBuilder, RiskScorecard, VetoAggregator
from .ablation_config import BlueTeamAblationConfig

logger = logging.getLogger(__name__)


class BlueOrchestrator:
    def __init__(self, config: BlueTeamAblationConfig | None = None) -> None:
        self.config: BlueTeamAblationConfig = (
            config or BlueTeamAblationConfig.from_mode("full")
        )
        self.doc_extractor: DocExtractor = DocExtractor()
        self.code_extractor: CodeExtractor = CodeExtractor()
        self.repo_extractor: RepoSliceExtractor = RepoSliceExtractor()
        self.scanners: list[
            RepoContextScanner | GuidanceInjectionScanner | SemanticGapScanner
        ] = []
        if self.config.enable_repo_context:
            self.scanners.append(RepoContextScanner())
        if self.config.enable_guidance_scanner:
            self.scanners.append(GuidanceInjectionScanner())
        if self.config.enable_semantic_gap:
            self.scanners.append(SemanticGapScanner())
        self.chain_risk_scanner: ChainRiskPrecheckScanner = ChainRiskPrecheckScanner()
        self.probe_runner: DynamicProbeRunner = DynamicProbeRunner()
        self.finding_decorator: FindingDecorator = FindingDecorator()
        self.scorecard: RiskScorecard = RiskScorecard()
        self.verdict: VetoAggregator = VetoAggregator()
        self.report_builder: AuditReportBuilder = AuditReportBuilder()

    def audit(self, skill: SkillArtifact, repo: RepositorySlice) -> AuditReport:
        doc_analysis = self.doc_extractor.extract(skill)
        code_analysis = self.code_extractor.extract(skill)
        repo_analysis = self.repo_extractor.extract(repo)
        findings = []
        structured_findings = StructuredFindings()
        for scanner in self.scanners:
            scanner_findings, scanner_structured_findings = scanner.scan_structured(
                skill,
                repo,
                doc_analysis=doc_analysis,
                code_analysis=code_analysis,
                repo_analysis=repo_analysis,
            )
            findings.extend(scanner_findings)
            structured_findings.extend(scanner_structured_findings)
        runtime_trace_summary: RuntimeTraceSummary | None = None
        if self.config.enable_dynamic_probe:
            probe_output = self.probe_runner.run(
                skill, repo, code_analysis=code_analysis
            )
            raw_runtime_trace_summary = probe_output.get("runtime_trace_summary")
            runtime_trace_summary = (
                raw_runtime_trace_summary
                if isinstance(raw_runtime_trace_summary, RuntimeTraceSummary)
                else None
            )

        # LLM-powered I/O Binding Analysis: refine source/sink tags
        # before chain risk evaluation to reduce false positives from
        # keyword-based tag detection.
        original_source = list(skill.source_tags)
        original_sink = list(skill.sink_tags)
        refined_src, refined_sink, llm_result = refine_skill_tags(
            skill, layer="alignment"
        )
        if llm_result.used_llm:
            skill.source_tags = refined_src
            skill.sink_tags = refined_sink
            logger.info(
                "LLM tag refinement for %s: source %s→%s, sink %s→%s",
                skill.skill_id,
                original_source,
                refined_src,
                original_sink,
                refined_sink,
            )

        if self.config.enable_chain_risk:
            findings.extend(
                self.chain_risk_scanner.scan(
                    skill,
                    repo,
                    doc_analysis=doc_analysis,
                    code_analysis=code_analysis,
                    runtime_trace_summary=runtime_trace_summary,
                )
            )
            composition_risk_edges = list(self.chain_risk_scanner.last_edges)
        else:
            composition_risk_edges = []
        decorated_feedback = self.finding_decorator.decorate(skill, findings)
        risk_scorecard = self.scorecard.build(
            skill, findings, runtime_trace_summary, composition_risk_edges
        )
        verdict = self.verdict.choose(findings, risk_scorecard)
        return self.report_builder.build(
            skill,
            findings,
            decorated_feedback,
            runtime_trace_summary,
            composition_risk_edges,
            risk_scorecard,
            verdict,
            structured_findings,
        )


# ---------------------------------------------------------------------------
# Red-team bridge — module-level audit() adapter.
# ---------------------------------------------------------------------------


@dataclass
class _RedTeamAuditResult:
    """Lightweight shim returned to the red-team driver.

    Exposes ``.findings`` as ``list[PaperFinding]`` so
    :func:`red_team.agent.prompts.render_findings_block` (which calls
    ``f.decorated_feedback_tag()``) works without a second conversion
    step.  The full :class:`AuditReport` is kept on ``.report`` for
    callers that want composition-risk / scorecard details.
    """

    findings: list[PaperFinding]
    report: AuditReport


def _static_to_paper(sf: StaticFinding) -> PaperFinding:
    line_range: tuple[int, int] | None = None
    if sf.line_start or sf.line_end:
        line_range = (int(sf.line_start or 0), int(sf.line_end or 0))
    return PaperFinding(
        finding_id=sf.finding_id,
        layer=sf.stage.value,
        category=sf.reason_code or sf.scanner,
        reason_code=sf.reason_code,
        confidence=0.8,
        severity=sf.severity.value,
        evidence=Evidence(
            file=sf.file or "",
            line_range=line_range,
            snippet=sf.evidence or "",
        ),
        explanation=sf.explanation or "",
        metadata=dict(sf.metadata or {}),
    )


def _skill_root() -> Path:
    """Locate the ClawHub skill root (respects ``CLAW_DATA_ROOT`` override).

    Mirrors the same env-var contract as
    :mod:`red_team.evolution.search_space` so the blue team and the
    search-space validator always agree on which directory holds the
    benign skill mirror.
    """
    env = os.environ.get("CLAW_DATA_ROOT")
    if env:
        return Path(env) / "skills" / "clawhub" / "skills"
    # orchestrator.py → blue_team → src → <repo>
    return Path(__file__).resolve().parents[3] / "data" / "skills" / "clawhub" / "skills"


def _compose_full_skill_md(variant: SkillVariant) -> str:
    """Return the SKILL.md the sandbox actually serves: benign + payload.

    Blue-team audit must see the same bytes the marketplace publishes —
    which is the **original benign skill body** with the mutator's
    poison payload (``variant.d``) appended.  Keeping this symmetric
    with :func:`sandbox.openclaw_chain_runner.run_poisoned_chain` is
    critical: a mismatched view causes scanners like RepoContextScanner
    to underestimate repo_overlap (they'd only see the payload) and
    SemanticGapScanner to over-declare UNDER_DECLARATION (they'd miss
    the benign declarations in the skill's frontmatter/body).

    Falls back to ``variant.d`` when the sink skill's benign SKILL.md
    cannot be located (e.g. fixture test runs without the full
    marketplace mirror on disk).
    """
    chain = list(variant.chain or [])
    if not chain:
        return variant.d or ""
    pidx = variant.poison_index
    if pidx < 0:
        pidx += len(chain)
    pidx = max(0, min(len(chain) - 1, pidx))

    orig_md = _skill_root() / chain[pidx] / "SKILL.md"
    if not orig_md.is_file():
        return variant.d or ""

    try:
        benign_body = orig_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return variant.d or ""

    payload = variant.d or ""
    separator = (
        "" if benign_body.endswith("\n\n")
        else ("\n\n" if benign_body.endswith("\n") else "\n\n")
    )
    return benign_body + separator + payload.lstrip("\n")


def _chain_to_artifacts(variant: SkillVariant) -> list[SkillArtifact]:
    """Build one :class:`SkillArtifact` per skill in ``variant.chain``.

    Every non-poisoned skill is loaded straight from the ClawHub mirror
    via :func:`shared.loaders.skill_loader.load_skill_artifact`, which
    populates ``skill_doc / code_files / source_tags / sink_tags`` from
    the real on-disk files — the same view an audit system would get
    for a marketplace publication.

    The poisoned skill (``variant.poison_index``) reuses the loader for
    code / tags, then **overrides its SKILL.md** with the mutator's
    payload composed on top of the benign body (see
    :func:`_compose_full_skill_md`) plus any ``variant.c['scripts']``
    rewrites from the mutator.  This way the blue team sees what the
    sandbox serves, while still benefiting from the loader's capability
    detection on the benign portion.

    When the ClawHub mirror is not reachable (e.g. fixture-based unit
    tests), we fall back to a single synthetic artifact composed
    entirely from the variant itself.
    """
    # Lazy import: keeps tests that mock `audit` from pulling shared/
    # dependencies.
    from shared.loaders.skill_loader import load_skill_artifact

    chain = list(variant.chain or [])
    if not chain:
        return [_synthetic_artifact(variant)]

    root = _skill_root()
    pidx = variant.poison_index
    if pidx < 0:
        pidx += len(chain)
    pidx = max(0, min(len(chain) - 1, pidx))

    artifacts: list[SkillArtifact] = []
    for i, skill_id in enumerate(chain):
        skill_dir = root / skill_id
        try:
            art = load_skill_artifact(skill_dir, source="clawhub_mirror")
        except (FileNotFoundError, OSError):
            # Mirror missing — synthesize minimal artifact for this slot.
            art = SkillArtifact(
                skill_id=skill_id,
                skill_root=str(skill_dir),
                skill_doc_path="SKILL.md",
                entry_scripts=[],
                skill_doc="",
                code_files={},
                metadata={"source": "missing_mirror"},
                permissions_declared=[],
            )

        if i == pidx:
            # Overlay the poisoned SKILL.md (benign body + mutator payload).
            art.skill_doc = _compose_full_skill_md(variant)
            # Overlay any mutator-authored script files on top of the
            # benign code_files so the blue team sees the executed code.
            mut_scripts = (variant.c or {}).get("scripts")
            if isinstance(mut_scripts, dict):
                for rel, body in mut_scripts.items():
                    if isinstance(rel, str) and isinstance(body, str) and rel:
                        art.code_files[rel] = body
            # Stamp provenance so aggregated findings can be traced back.
            art.metadata = {
                **(art.metadata or {}),
                "role": "poisoned_sink" if i == len(chain) - 1 else "poisoned_step",
                "chain_position": i,
                "chain_length": len(chain),
                "objective": variant.o.value,
                "variant_hash": variant.content_hash(),
            }
        else:
            art.metadata = {
                **(art.metadata or {}),
                "role": "benign_chain_peer",
                "chain_position": i,
                "chain_length": len(chain),
            }
        artifacts.append(art)

    return artifacts


def _synthetic_artifact(variant: SkillVariant) -> SkillArtifact:
    """Fallback when no chain is available (unit tests etc.)."""
    c: dict[str, Any] = dict(variant.c or {})
    scripts: dict[str, str] = {}
    raw_scripts = c.get("scripts")
    if isinstance(raw_scripts, dict):
        for path, body in raw_scripts.items():
            if isinstance(path, str) and isinstance(body, str):
                scripts[path] = body
    entry_script = c.get("entry_script")
    entry_scripts: list[str] = []
    if isinstance(entry_script, str) and entry_script:
        entry_scripts.append(entry_script)
        scripts.setdefault(entry_script, "")
    seed_id = str(variant.metadata.get("seed_id") or variant.content_hash())
    return SkillArtifact(
        skill_id=seed_id,
        skill_root=f"/tmp/red_team_audit/{seed_id}",
        skill_doc_path="SKILL.md",
        entry_scripts=entry_scripts,
        skill_doc=_compose_full_skill_md(variant),
        code_files=scripts,
        metadata={"source": "red_team_variant", "objective": variant.o.value},
        permissions_declared=list(c.get("declared_apis") or []),
    )


def _variant_to_artifact(variant: SkillVariant) -> SkillArtifact:
    """Back-compat single-skill wrapper — only the poisoned sink.

    Retained so internal callers that want the sink artifact alone
    (e.g. for debug dumps) still have a one-shot.  The production
    :func:`audit` path uses :func:`_chain_to_artifacts` for full chain
    coverage.
    """
    artifacts = _chain_to_artifacts(variant)
    # Pick the poisoned one (chain_position matches poison_index).
    for art in artifacts:
        role = (art.metadata or {}).get("role", "")
        if role in ("poisoned_sink", "poisoned_step"):
            return art
    return artifacts[0]


def audit(
    variant: SkillVariant,
    repo: RepositorySlice,
    *,
    config: BlueTeamAblationConfig | None = None,
) -> _RedTeamAuditResult:
    """Audit a red-team :class:`SkillVariant` under the full blue pipeline.

    Runs :class:`BlueOrchestrator` once per skill in ``variant.chain``
    (not just the poisoned sink) so:

    * The benign source skill's ``source_tags`` are visible to the
      Layer-E ``ChainRiskPrecheckScanner`` when it runs on the poisoned
      sink (via shared ``skill.chain_context`` metadata we stamp below).
    * LLM alignment / gatekeeper scanners can compare the doc & code of
      every skill the user installs, not only the one carrying the
      mutator's payload.

    Per-skill :class:`StaticFinding` lists are concatenated and each
    finding's ``metadata['origin_skill']`` is stamped with the
    originating skill id so the red-team driver can trace which skill
    in the chain produced which finding.  The ``.report`` field
    returns the audit report of the **poisoned** skill — keeping the
    veto semantics unchanged for callers that look at
    ``report.verdict`` directly.
    """
    artifacts = _chain_to_artifacts(variant)
    orch = BlueOrchestrator(config=config)

    # First pass — every skill's standalone audit.  We accumulate
    # StaticFindings (not PaperFindings yet so metadata survives).
    per_skill_reports: dict[str, AuditReport] = {}
    aggregated: list[StaticFinding] = []
    poisoned_report: AuditReport | None = None
    for art in artifacts:
        report = orch.audit(art, repo)
        per_skill_reports[art.skill_id] = report
        role = (art.metadata or {}).get("role", "")
        if role.startswith("poisoned"):
            poisoned_report = report
        for sf in report.findings:
            # Stamp provenance so the driver can split findings by skill.
            md = dict(sf.metadata or {})
            md.setdefault("origin_skill", art.skill_id)
            md.setdefault("chain_position", (art.metadata or {}).get("chain_position"))
            sf.metadata = md
            aggregated.append(sf)

    # If no poisoned artifact (shouldn't happen when chain is populated),
    # fall back to the first report.
    if poisoned_report is None:
        poisoned_report = next(iter(per_skill_reports.values()))
        # Rebuild poisoned_report.findings to include aggregated set so
        # verdict-level summaries reflect the full chain view.
    poisoned_report.findings = list(aggregated)

    paper_findings = [_static_to_paper(sf) for sf in aggregated]
    return _RedTeamAuditResult(findings=paper_findings, report=poisoned_report)
