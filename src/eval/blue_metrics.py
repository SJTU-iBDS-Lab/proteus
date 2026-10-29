"""Blue-team evaluation on subgraph eval splits.

Approach: each subgraph has a ``primary_attack_type`` ground-truth label. We run
``AuditPipeline`` on every unique skill referenced by the split, treat
``verdict != APPROVED`` as a positive detection, then aggregate to subgraph
level (subgraph "detected" iff every node skill is flagged). Optional benign
skills, sampled from the skill corpus and not present in any split, provide
negatives for precision / FPR computation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from blue_team import AuditPipeline, BlueTeamAblationConfig
from blue_team.flow_simulator import CrossSkillFlowSimulator
from blue_team.flow_simulator.cross_skill_analyzer import SubgraphContext
from core.enums import VerdictStatus
from core.models import AuditReport


@dataclass(slots=True)
class SkillPrediction:
    skill_id: str
    verdict: str
    predicted_positive: bool
    finding_count: int
    stages: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphOutcome:
    subgraph_id: str
    primary_attack_type: str
    node_skill_ids: list[str]
    detected: bool
    missing_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AttackFamilyMetrics:
    family: str
    total: int
    detected: int
    recall: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class BlueEvalReport:
    split_path: str
    config_mode: str
    total_subgraphs: int
    total_unique_skills: int
    subgraph_recall: float
    per_family: list[AttackFamilyMetrics]
    benign_count: int = 0
    benign_false_positives: int = 0
    fpr: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    skipped_subgraphs: int = 0
    predictions: list[SkillPrediction] = field(default_factory=list)
    subgraphs: list[SubgraphOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "split_path": self.split_path,
            "config_mode": self.config_mode,
            "total_subgraphs": self.total_subgraphs,
            "total_unique_skills": self.total_unique_skills,
            "subgraph_recall": self.subgraph_recall,
            "per_family": [m.to_dict() for m in self.per_family],
            "benign_count": self.benign_count,
            "benign_false_positives": self.benign_false_positives,
            "fpr": self.fpr,
            "precision": self.precision,
            "f1": self.f1,
            "skipped_subgraphs": self.skipped_subgraphs,
            "predictions": [p.to_dict() for p in self.predictions],
            "subgraphs": [s.to_dict() for s in self.subgraphs],
        }


AuditFn = Callable[[Path, Path], AuditReport]


def load_eval_subgraphs(split_path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with open(split_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_skill_ids(subgraph: dict[str, object]) -> list[str]:
    nodes = subgraph.get("nodes", [])
    ids: list[str] = []
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            raw = node.get("node_id") or node.get("skill_id")
            if not isinstance(raw, str):
                continue
            skill_id = raw.removeprefix("node:")
            if skill_id:
                ids.append(skill_id)
    return ids


def _parse_subgraph_context(sg: dict[str, object]) -> SubgraphContext:
    """Convert raw subgraph dict to SubgraphContext for flow simulation."""
    sg_id = str(sg.get("subgraph_id", ""))
    attack_type = str(sg.get("primary_attack_type", "unknown"))
    node_ids = _extract_skill_ids(sg)
    raw_edges = sg.get("edges", [])
    edges = [
        {"edge_id": str(e.get("edge_id", "")), "order_index": str(e.get("order_index", 0))}
        for e in (raw_edges if isinstance(raw_edges, list) else [])
        if isinstance(e, dict)
    ]
    raw_lineage = sg.get("lineage", {})
    lineage = raw_lineage if isinstance(raw_lineage, dict) else {}
    return SubgraphContext(
        subgraph_id=sg_id,
        attack_type=attack_type,
        node_skill_ids=node_ids,
        edges=edges,
        lineage=lineage,
    )


def _default_audit_fn(config: BlueTeamAblationConfig) -> AuditFn:
    pipeline = AuditPipeline(config=config)

    def _audit(skill_root: Path, repo_root: Path) -> AuditReport:
        return pipeline.audit_paths(skill_root, repo_root)

    return _audit


def _predict_skill(
    skill_id: str,
    skills_root: Path,
    audit_fn: AuditFn,
) -> SkillPrediction:
    skill_dir = skills_root / skill_id
    if not skill_dir.exists():
        return SkillPrediction(
            skill_id=skill_id,
            verdict="",
            predicted_positive=False,
            finding_count=0,
            stages=[],
            error=f"skill directory missing: {skill_dir}",
        )
    try:
        report = audit_fn(skill_dir, skill_dir)
    except Exception as exc:  # boundary: any loader/scanner error
        return SkillPrediction(
            skill_id=skill_id,
            verdict="",
            predicted_positive=False,
            finding_count=0,
            stages=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    stages = sorted({finding.stage.value for finding in report.findings})
    return SkillPrediction(
        skill_id=skill_id,
        verdict=report.verdict.value,
        predicted_positive=report.verdict != VerdictStatus.APPROVED,
        finding_count=len(report.findings),
        stages=stages,
    )


def _sample_benign_skills(
    skills_root: Path, exclude: set[str], count: int, seed: int
) -> list[str]:
    if count <= 0 or not skills_root.exists():
        return []
    rng = random.Random(seed)
    candidates = [
        child.name
        for child in skills_root.iterdir()
        if child.is_dir() and child.name not in exclude and (child / "SKILL.md").exists()
    ]
    rng.shuffle(candidates)
    return candidates[:count]


def evaluate_blue_on_split(
    split_path: str | Path,
    skills_root: str | Path,
    *,
    config: BlueTeamAblationConfig | None = None,
    benign_sample_size: int = 0,
    benign_seed: int = 0,
    audit_fn: AuditFn | None = None,
    skill_limit: int = 0,
) -> BlueEvalReport:
    """Run the blue pipeline on every skill in ``split_path`` and compute metrics."""

    split_path = Path(split_path)
    skills_root = Path(skills_root)
    cfg = config or BlueTeamAblationConfig.from_mode("full")
    predict_fn = audit_fn or _default_audit_fn(cfg)

    subgraphs = load_eval_subgraphs(split_path)
    unique_skill_ids: list[str] = []
    seen: set[str] = set()
    for sg in subgraphs:
        for sid in _extract_skill_ids(sg):
            if sid not in seen:
                seen.add(sid)
                unique_skill_ids.append(sid)

    if skill_limit > 0:
        unique_skill_ids = unique_skill_ids[:skill_limit]
        seen = set(unique_skill_ids)

    predictions: dict[str, SkillPrediction] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_predict_skill, sid, skills_root, predict_fn): sid
            for sid in unique_skill_ids
        }
        for future in as_completed(futures):
            sid = futures[future]
            predictions[sid] = future.result()

    # Phase 2: Cross-skill flow simulation for subgraphs with undetected skills
    flow_sim = CrossSkillFlowSimulator(skills_root)

    subgraph_outcomes: list[SubgraphOutcome] = []
    family_totals: dict[str, int] = {}
    family_detected: dict[str, int] = {}
    skipped = 0
    # Collect subgraphs needing flow simulation
    pending_flow_sim: list[tuple[dict[str, object], str, str, list[str]]] = []

    for sg in subgraphs:
        sg_id = str(sg.get("subgraph_id", ""))
        family = str(sg.get("primary_attack_type", "unknown"))
        node_ids = _extract_skill_ids(sg)
        if skill_limit > 0 and any(sid not in seen for sid in node_ids):
            skipped += 1
            continue
        missing = [
            sid
            for sid in node_ids
            if sid not in predictions or predictions[sid].error
        ]
        if missing:
            skipped += 1
            subgraph_outcomes.append(
                SubgraphOutcome(
                    subgraph_id=sg_id,
                    primary_attack_type=family,
                    node_skill_ids=node_ids,
                    detected=False,
                    missing_skills=missing,
                )
            )
            continue
        detected = bool(node_ids) and all(
            predictions[sid].predicted_positive for sid in node_ids
        )
        if detected:
            family_totals[family] = family_totals.get(family, 0) + 1
            family_detected[family] = family_detected.get(family, 0) + 1
            subgraph_outcomes.append(
                SubgraphOutcome(
                    subgraph_id=sg_id,
                    primary_attack_type=family,
                    node_skill_ids=node_ids,
                    detected=True,
                )
            )
        else:
            # Not all skills flagged — queue for cross-skill flow simulation
            pending_flow_sim.append((sg, sg_id, family, node_ids))

    # Run cross-skill flow simulation on undetected subgraphs
    if pending_flow_sim:
        sim_contexts = [
            _parse_subgraph_context(sg) for sg, _, _, _ in pending_flow_sim
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            sim_results = list(pool.map(
                lambda ctx: flow_sim.simulate(ctx, layer="flow_sim"),
                sim_contexts,
            ))

        for (sg, sg_id, family, node_ids), sim_result in zip(
            pending_flow_sim, sim_results
        ):
            family_totals[family] = family_totals.get(family, 0) + 1
            # If flow simulation confirms attack chain, mark as detected
            # and elevate individual skill predictions
            if (
                sim_result.used_llm
                and sim_result.is_attack_chain
                and sim_result.confidence in ("high", "medium")
            ):
                family_detected[family] = family_detected.get(family, 0) + 1
                # Elevate predictions for skills identified as risky
                for sid in node_ids:
                    if sid in predictions and not predictions[sid].predicted_positive:
                        predictions[sid] = SkillPrediction(
                            skill_id=sid,
                            verdict="conditional",
                            predicted_positive=True,
                            finding_count=predictions[sid].finding_count,
                            stages=predictions[sid].stages + ["cross_skill_flow_sim"],
                        )
                subgraph_outcomes.append(
                    SubgraphOutcome(
                        subgraph_id=sg_id,
                        primary_attack_type=family,
                        node_skill_ids=node_ids,
                        detected=True,
                    )
                )
            else:
                subgraph_outcomes.append(
                    SubgraphOutcome(
                        subgraph_id=sg_id,
                        primary_attack_type=family,
                        node_skill_ids=node_ids,
                        detected=False,
                    )
                )

    scored_total = sum(family_totals.values())
    detected_total = sum(family_detected.values())
    recall = detected_total / scored_total if scored_total else 0.0

    per_family = sorted(
        (
            AttackFamilyMetrics(
                family=family,
                total=total,
                detected=family_detected.get(family, 0),
                recall=round(
                    family_detected.get(family, 0) / total if total else 0.0, 4
                ),
            )
            for family, total in family_totals.items()
        ),
        key=lambda m: m.family,
    )

    benign_ids = _sample_benign_skills(
        skills_root, exclude=seen, count=benign_sample_size, seed=benign_seed
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        benign_predictions = list(pool.map(
            lambda sid: _predict_skill(sid, skills_root, predict_fn),
            benign_ids,
        ))
    benign_scored = [p for p in benign_predictions if not p.error]
    benign_fp = sum(1 for p in benign_scored if p.predicted_positive)
    fpr = benign_fp / len(benign_scored) if benign_scored else 0.0

    tp = detected_total
    fp = benign_fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return BlueEvalReport(
        split_path=str(split_path),
        config_mode=cfg.mode,
        total_subgraphs=len(subgraphs),
        total_unique_skills=len(unique_skill_ids),
        subgraph_recall=round(recall, 4),
        per_family=per_family,
        benign_count=len(benign_scored),
        benign_false_positives=benign_fp,
        fpr=round(fpr, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        skipped_subgraphs=skipped,
        predictions=list(predictions.values()) + benign_predictions,
        subgraphs=subgraph_outcomes,
    )


def render_markdown(report: BlueEvalReport) -> str:
    lines = [
        "# Blue Eval Report",
        "",
        f"- Split: `{report.split_path}`",
        f"- Config: `{report.config_mode}`",
        f"- Subgraphs scored: {report.total_subgraphs - report.skipped_subgraphs}"
        f" / {report.total_subgraphs} (skipped {report.skipped_subgraphs})",
        f"- Unique skills audited: {report.total_unique_skills}",
        f"- Overall subgraph recall: **{report.subgraph_recall:.4f}**",
        f"- Benign sample: {report.benign_count}"
        f" (false positives: {report.benign_false_positives}, FPR: {report.fpr:.4f})",
        f"- Precision: {report.precision:.4f} — F1: {report.f1:.4f}",
        "",
        "## Per-family recall",
        "",
        "| Attack family | Total | Detected | Recall |",
        "|---------------|-------|----------|--------|",
    ]
    for metric in report.per_family:
        lines.append(
            f"| {metric.family} | {metric.total} | {metric.detected} |"
            f" {metric.recall:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: BlueEvalReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "blue_eval_metrics.json"
    md_path = output_dir / "blue_eval_metrics.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "AttackFamilyMetrics",
    "AuditFn",
    "BlueEvalReport",
    "SkillPrediction",
    "SubgraphOutcome",
    "evaluate_blue_on_split",
    "load_eval_subgraphs",
    "render_markdown",
    "write_report",
]


def _unused(_: Iterable[Sequence[str]]) -> None:  # pragma: no cover - type import guard
    return None
