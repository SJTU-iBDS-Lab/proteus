"""Wrapper to call AI-Infra-Guard mcp-scan on a single skill directory.

Invokes AIG's Python agent framework as a subprocess with hard timeout,
parses the structured resultUpdate JSON / <vuln> XML output, and returns
typed SkillScanResult.

Uses subprocess.run (not asyncio) to guarantee child-process cleanup on
timeout — asyncio.wait_for only cancels the coroutine, leaving the AIG
process running indefinitely.

Configuration (matches the rest of the pipeline):
  - Model name: ``configs/models.yaml`` → ``aig.model``
  - Endpoint:   ``LITELLM_URL`` env (default ``http://127.0.0.1:4000``)
  - Auth:       ``LITELLM_KEY`` env (default ``sk-claw-local``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .aig_types import AIGVulnerability, SkillScanResult

logger = logging.getLogger(__name__)

# Path to AIG mcp-scan directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AIG_ROOT = _PROJECT_ROOT / "external" / "AI-Infra-Guard" / "mcp-scan"
# AIG mcp-scan launches an LLM-driven multi-step agent (info-gather →
# scan → synthesise) and one full scan typically takes 4-10 minutes
# against ``glm-5``-class models on a non-trivial skill.  Default 900s
# avoids spurious TimeoutExpired on poisoned chain skills.  Override
# via ``AIG_TIMEOUT_S`` env when batch-running.
_TIMEOUT = int(os.environ.get("AIG_TIMEOUT_S", "900"))

# Load .env for LLM credentials
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


def _aig_model() -> str:
    """Return the AIG model id from configs/models.yaml (or env override)."""
    if (override := os.environ.get("AIG_MODEL", "").strip()):
        return override
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load((_PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
        return (cfg or {}).get("aig", {}).get("model") or "glm-5"
    except Exception:
        return "glm-5"


def _get_aig_env() -> dict[str, str]:
    """Build environment variables for AIG subprocess.

    AIG expects separate env vars for each of its internal LLM roles
    (thinking, coding, fast).  We map them all to the same proxy
    endpoint + key + model.
    """
    env = os.environ.copy()
    api_key = os.getenv("LITELLM_KEY", "sk-claw-local")
    base_url = os.getenv("LITELLM_URL", "http://127.0.0.1:4000")
    model = _aig_model()

    for k in ("OPENROUTER_API_KEY", "THINKING_API_KEY", "CODING_API_KEY", "FAST_API_KEY"):
        env[k] = api_key
    for k in ("DEFAULT_BASE_URL", "THINKING_BASE_URL", "CODING_BASE_URL", "FAST_BASE_URL"):
        env[k] = base_url
    for k in ("DEFAULT_MODEL", "THINKING_MODEL", "CODING_MODEL", "FAST_MODEL"):
        env[k] = model
    # Forward the AIG LLM-call timeout (read by the patched
    # ``utils/llm.py`` inside AIG itself).  Default 600s — well above
    # glm-5's ~120s reasoning ceiling on a 30K-context call.
    env["AIG_LLM_TIMEOUT_S"] = os.environ.get("AIG_LLM_TIMEOUT_S", "600")
    # AIG defaults to streaming; LiteLLM/ChatAnywhere often drop
    # mid-stream ("peer closed connection ... incomplete chunked
    # read").  Default to non-stream mode for stability.  Override
    # with ``AIG_STREAM=1`` if you need streaming.
    env["AIG_STREAM"] = os.environ.get("AIG_STREAM", "0")

    return env


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_result_update(text: str) -> tuple[list[AIGVulnerability], int]:
    """Parse AIG's resultUpdate JSON from log output.

    AIG emits a JSON line like:
      {"type":"resultUpdate","content":{"score":35,"results":[{...}],...}}
    """
    vulns: list[AIGVulnerability] = []
    score = 100

    for line in text.split("\n"):
        if '"resultUpdate"' not in line:
            continue
        brace_start = line.find("{")
        if brace_start < 0:
            continue
        try:
            data = json.loads(line[brace_start:])
        except json.JSONDecodeError:
            continue
        if data.get("type") != "resultUpdate":
            continue
        content = data.get("content", {})
        raw_score = content.get("score")
        if isinstance(raw_score, (int, float)):
            score = int(raw_score)
        for item in content.get("results", []):
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            desc = item.get("description", "")
            if title and desc:
                vulns.append(
                    AIGVulnerability(
                        title=str(title).strip(),
                        description=str(desc).strip(),
                        risk_type=str(item.get("risk_type", "")).strip(),
                        level=str(item.get("level", "Medium")).strip(),
                        suggestion=str(item.get("suggestion", "")).strip(),
                    )
                )
        break  # Only need the first resultUpdate

    return vulns, score


def _parse_vuln_xml(text: str) -> list[AIGVulnerability]:
    """Parse AIG's <vuln> XML blocks from output text."""
    pattern = re.compile(r"<vuln>\s*(.*?)\s*</vuln>", re.DOTALL)
    blocks = pattern.findall(text)

    vulns: list[AIGVulnerability] = []
    # AIG's reviewer prompt sometimes emits ``<vuln>`` blocks with
    # explicit "no-vulnerability" markers — title="None", risk_type="None
    # confirmed", level="" / "None" / "Info" — when it determined the
    # repository is benign instructional content.  Treat those as
    # "scan succeeded, 0 vulns" (skip), NOT as Medium-severity findings.
    _NULL_TOKENS = {"", "none", "n/a", "info", "informational", "unknown"}
    for block in blocks:
        title = _extract_tag(block, "title")
        desc = _extract_tag(block, "desc")
        risk_type = _extract_tag(block, "risk_type")
        level = _extract_tag(block, "level")
        suggestion = _extract_tag(block, "suggestion")

        if not (title and desc and risk_type):
            continue
        if (risk_type or "").strip().lower() in _NULL_TOKENS:
            continue
        if (level or "").strip().lower() in _NULL_TOKENS:
            continue
        if (title or "").strip().lower() in _NULL_TOKENS:
            continue

        vulns.append(
            AIGVulnerability(
                title=title.strip(),
                description=desc.strip(),
                risk_type=risk_type.strip(),
                level=level.strip() if level else "Medium",
                suggestion=suggestion.strip() if suggestion else "",
            )
        )

    return vulns


def _extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Core scan function (synchronous subprocess with hard timeout)
# ---------------------------------------------------------------------------

def _run_aig_scan(skill_dir: str, language: str = "en") -> tuple[str, int]:
    """Run AIG mcp-scan as a subprocess with hard timeout.

    Returns:
        (combined stdout+stderr output, return code)
    """
    if not _AIG_ROOT.exists():
        raise FileNotFoundError(
            f"AI-Infra-Guard mcp-scan not found at {_AIG_ROOT}. "
            "Run: git clone https://github.com/Tencent/AI-Infra-Guard.git external/AI-Infra-Guard"
        )

    cmd = [
        # AIG's agent framework has its own heavy deps (loguru, etc.) that
        # the campaign venv need not carry; AIG_PYTHON points at an
        # interpreter that has them (defaults to the current one).
        os.environ.get("AIG_PYTHON", sys.executable),
        str(_AIG_ROOT / "main.py"),
        "--repo", str(Path(skill_dir).resolve()),
        "--language", language,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        cwd=str(_AIG_ROOT),
        env=_get_aig_env(),
    )
    full_output = result.stdout + "\n" + result.stderr
    return full_output, result.returncode


def scan_skill(
    skill_id: str,
    skill_path: str,
    *,
    language: str = "en",
) -> SkillScanResult:
    """Scan a single skill directory with AIG mcp-scan.

    Args:
        skill_id: Skill identifier.
        skill_path: Absolute path to the skill directory.
        language: Output language ("en" or "zh").

    Returns:
        SkillScanResult with vulnerabilities and safety score.
    """
    t0 = time.time()

    if not Path(skill_path).exists():
        return SkillScanResult(
            skill_id=skill_id,
            skill_path=skill_path,
            error=f"Skill directory not found: {skill_path}",
        )

    try:
        full_output, rc = _run_aig_scan(skill_path, language)
    except subprocess.TimeoutExpired:
        return SkillScanResult(
            skill_id=skill_id,
            skill_path=skill_path,
            error=f"AIG scan timed out after {_TIMEOUT}s",
            elapsed_seconds=time.time() - t0,
        )
    except Exception as exc:
        return SkillScanResult(
            skill_id=skill_id,
            skill_path=skill_path,
            error=f"AIG scan error: {type(exc).__name__}: {exc}",
            elapsed_seconds=time.time() - t0,
        )

    # Try both parsers — resultUpdate JSON is more reliable
    json_vulns, json_score = _parse_result_update(full_output)
    xml_vulns = _parse_vuln_xml(full_output)

    # Detect silent failure: subprocess returned non-zero AND parsers
    # found nothing.  Previously we returned ``score=100`` here, which
    # made an AIG crash indistinguishable from a clean scan — a
    # false-negative bug that masked missing dependencies / config
    # errors.  Surface the upstream error verbatim so callers can act.
    if rc != 0 and not json_vulns and not xml_vulns:
        # Pull the trailing chunk of the subprocess output as the error
        # context — typically contains the Python traceback or the
        # last few log lines that explain why AIG aborted.
        tail = (full_output or "").rstrip().splitlines()
        ctx = "\n".join(tail[-12:]) if tail else ""
        elapsed = time.time() - t0
        logger.warning(
            "AIG scan %s exited rc=%d in %.1fs with no parseable result; "
            "treating as scanner error",
            skill_id, rc, elapsed,
        )
        return SkillScanResult(
            skill_id=skill_id,
            skill_path=skill_path,
            error=f"AIG subprocess rc={rc} (no parseable output): {ctx[-1500:]}",
            elapsed_seconds=elapsed,
        )

    # Prefer JSON results; fall back to XML if JSON found nothing
    if json_vulns:
        vulns = json_vulns
        score = json_score
    elif xml_vulns:
        vulns = xml_vulns
        score = 100  # No score from XML parser
    else:
        vulns = []
        score = 100

    elapsed = time.time() - t0
    logger.info(
        "AIG scan %s: %d vulns (json=%d, xml=%d), score=%d, %.1fs",
        skill_id, len(vulns), len(json_vulns), len(xml_vulns), score, elapsed,
    )

    return SkillScanResult(
        skill_id=skill_id,
        skill_path=skill_path,
        score=score,
        vulnerabilities=vulns,
        elapsed_seconds=elapsed,
    )


async def scan_skill_async(
    skill_id: str,
    skill_path: str,
    *,
    language: str = "en",
) -> SkillScanResult:
    """Async wrapper — runs the synchronous scan in a thread executor.

    subprocess.run with hard timeout is the only reliable way to kill the
    AIG child process.  We offload to a thread so callers can still use
    asyncio.gather for concurrency.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: scan_skill(skill_id, skill_path, language=language)
    )
