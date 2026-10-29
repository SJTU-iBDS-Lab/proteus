"""Repo-aware heuristics used by the context scanner."""

from core.enums import ContextLabel


def assess_repo_context(
    skill_tokens: set[str], repo_tokens: set[str]
) -> tuple[ContextLabel, float, list[str]]:
    if not skill_tokens or not repo_tokens:
        return ContextLabel.UNKNOWN, 0.0, []
    overlap = sorted(skill_tokens & repo_tokens)
    ratio = len(overlap) / max(1, len(skill_tokens))
    if ratio >= 0.35:
        return ContextLabel.MATCH, ratio, overlap
    if ratio >= 0.12:
        return ContextLabel.PARTIAL_MATCH, ratio, overlap
    return ContextLabel.CONFLICT, ratio, overlap
