"""Thin wrapper around Docker to execute red-team seeds against skills.

Adapted from ref_repo/SkillAttack/stages/simulator/openclaw.py.
Only the container-exec logic is ported; all SkillAttack base-class and
config-loader dependencies are removed.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SandboxResult:
    subgraph_id: str
    skill_id: str
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    trajectory: list[dict[str, object]] = field(default_factory=list)
    observations: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SandboxConfig:
    container_name: str = "skillrt-openclaw"
    image: str = "openclaw-eval:latest"
    timeout: int = 300
    model: str = ""
    api_key: str = ""
    base_url: str = ""


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or f"timeout after {timeout}s"),
        )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result


def preflight(config: SandboxConfig) -> dict[str, object]:
    docker_ver = _run_cmd(
        ["docker", "version", "--format", "{{.Server.Version}}"], timeout=10
    )
    if docker_ver.returncode != 0:
        return {
            "ok": False,
            "code": "docker_unavailable",
            "message": (docker_ver.stderr or "Docker not available").strip(),
        }

    inspect = _run_cmd(
        ["docker", "inspect", "-f", "{{.State.Running}}", config.container_name],
        timeout=10,
    )
    if inspect.returncode != 0:
        return {
            "ok": False,
            "code": "container_not_found",
            "message": f"Container '{config.container_name}' not found. "
            f"Run: docker run -d --name {config.container_name} {config.image}",
        }

    running = (inspect.stdout or "").strip().lower() == "true"
    if not running:
        start = _run_cmd(["docker", "start", config.container_name], timeout=30)
        if start.returncode != 0:
            return {
                "ok": False,
                "code": "container_start_failed",
                "message": (start.stderr or "").strip(),
            }

    return {"ok": True, "container": config.container_name}


def _copy_to_container(
    container: str, local_path: Path, container_path: str
) -> None:
    _run_cmd(
        ["docker", "cp", str(local_path), f"{container}:{container_path}"],
        timeout=30,
        check=True,
    )


def _exec_in_container(
    container: str,
    cmd: str,
    *,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    docker_cmd = ["docker", "exec"]
    for key, val in (env or {}).items():
        docker_cmd.extend(["-e", f"{key}={val}"])
    docker_cmd.extend([container, "sh", "-lc", cmd])
    return _run_cmd(docker_cmd, timeout=timeout)


def _parse_tool_trace(stderr_text: str) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    raw = str(stderr_text or "")
    if not raw.strip():
        return [], [], {}

    tool_event_re = re.compile(
        r"^\[agent/embedded\] embedded run tool (?P<phase>start|end): "
        r"runId=(?P<run_id>\S+) tool=(?P<tool>\S+)"
    )

    trajectory: list[dict[str, object]] = []
    tool_calls: dict[str, dict[str, object]] = {}
    order: list[str] = []
    step = 0

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        match = tool_event_re.match(line)
        if not match:
            continue
        step += 1
        phase = match.group("phase")
        tool = match.group("tool")
        run_id = match.group("run_id")
        key = f"{tool}-{run_id}"

        trajectory.append({
            "step": step,
            "phase": phase,
            "tool": tool,
            "run_id": run_id,
        })
        if key not in tool_calls:
            tool_calls[key] = {"tool": tool, "status": "started"}
            order.append(key)
        if phase == "end":
            tool_calls[key]["status"] = "completed"

    calls = [tool_calls[k] for k in order]
    obs: dict[str, object] = {}
    if calls:
        obs["tool_call_count"] = len(calls)
    return trajectory, calls, obs


def run_seed(
    skill_path: Path,
    seed_turns: list[dict[str, str]],
    config: SandboxConfig,
    *,
    subgraph_id: str = "",
) -> SandboxResult:
    skill_id = skill_path.name
    container = config.container_name
    workspace = f"/tmp/sandbox/{skill_id}"

    _exec_in_container(container, f"rm -rf {shlex.quote(workspace)} && mkdir -p {shlex.quote(workspace)}")
    _copy_to_container(container, skill_path, f"{workspace}/skill")

    turns_json = json.dumps(seed_turns, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(turns_json)
        turns_file = f.name
    try:
        _copy_to_container(container, Path(turns_file), f"{workspace}/turns.json")
    finally:
        Path(turns_file).unlink(missing_ok=True)

    env = {}
    if config.api_key:
        env["ANTHROPIC_API_KEY"] = config.api_key
    if config.model:
        env["CLAUDE_MODEL"] = config.model
    if config.base_url:
        env["ANTHROPIC_BASE_URL"] = config.base_url

    exec_cmd = (
        f"cd {shlex.quote(workspace)} && "
        f"cat turns.json | claude --skill-dir skill --print --output-format json 2>/tmp/sandbox_stderr.log; "
        f"echo '---EXITCODE:'$?"
    )

    result = _exec_in_container(
        container, exec_cmd, timeout=config.timeout, env=env
    )

    stderr_result = _exec_in_container(
        container, "cat /tmp/sandbox_stderr.log 2>/dev/null || true", timeout=10
    )

    stdout = result.stdout or ""
    stderr = stderr_result.stdout or result.stderr or ""
    exit_code = result.returncode
    timed_out = exit_code == 124

    exit_match = re.search(r"---EXITCODE:(\d+)", stdout)
    if exit_match:
        exit_code = int(exit_match.group(1))
        stdout = stdout[: exit_match.start()].rstrip()

    trajectory, tool_calls, observations = _parse_tool_trace(stderr)

    return SandboxResult(
        subgraph_id=subgraph_id,
        skill_id=skill_id,
        exit_code=exit_code,
        stdout=stdout[:50000],
        stderr=stderr[:10000],
        timed_out=timed_out,
        tool_calls=tool_calls,
        trajectory=trajectory,
        observations=observations,
    )


__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "preflight",
    "run_seed",
]
