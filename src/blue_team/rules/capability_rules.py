"""Capability comparison helpers."""

from core.enums import SemanticAlignment


def compare_capabilities(
    declared: set[str], implemented: set[str]
) -> tuple[SemanticAlignment, dict[str, list[str]]]:
    undeclared = sorted(implemented - declared)
    unsupported = sorted(declared - implemented)
    if undeclared and unsupported:
        status = SemanticAlignment.MIXED
    elif undeclared:
        status = SemanticAlignment.UNDER_DECLARATION
    elif unsupported:
        status = SemanticAlignment.OVER_DECLARATION
    else:
        status = SemanticAlignment.MATCH
    return status, {"undeclared": undeclared, "unsupported": unsupported}
