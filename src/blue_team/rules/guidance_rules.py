"""Guidance-pattern matching rules."""

from core.constants import GUIDANCE_PATTERNS
from core.enums import Severity


class GuidanceRuleSet:
    def match(self, lines: list[str]) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for index, line in enumerate(lines, start=1):
            lowered = line.lower()
            for reason_code, keywords in GUIDANCE_PATTERNS.items():
                hit = next(
                    (keyword for keyword in keywords if keyword in lowered), None
                )
                if hit is None:
                    continue
                severity = (
                    Severity.MEDIUM
                    if reason_code in ("autonomy_encouragement", "disguised_required_step")
                    else Severity.LOW
                )
                matches.append(
                    {
                        "reason_code": reason_code,
                        "severity": severity,
                        "line_start": index,
                        "line_end": index,
                        "evidence": line.strip(),
                        "explanation": f"Matched guidance cue '{hit}'",
                    }
                )
        return matches
