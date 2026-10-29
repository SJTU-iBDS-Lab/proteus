"""Lightweight A.I.G mcp_scan adapter for deep skill analysis.

Adapted from ref_repo/SkillAttack/stages/analyzer/aig_native_api.py.
Requires a running A.I.G server (default: http://localhost:18088).
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(slots=True)
class AIGFinding:
    title: str = ""
    description: str = ""
    risk_type: str = ""
    level: str = ""
    suggestion: str = ""


@dataclass(slots=True)
class AIGResult:
    skill_id: str = ""
    findings: list[AIGFinding] = field(default_factory=list)
    readme: str = ""
    session_id: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, object]:
    req_headers: dict[str, str] = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    data = None
    if payload is not None:
        req_headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(url=url, data=data, method=method.upper(), headers=req_headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AIG HTTP {exc.code} at {url}: {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"AIG request failed for {url}: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"AIG unexpected payload type from {url}")
    return parsed


def _upload_archive(archive_path: Path, base_url: str) -> str:
    boundary = f"----AIGScan{uuid.uuid4().hex}"
    filename = archive_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/zip"
    file_bytes = archive_path.read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    url = f"{base_url}/api/v1/app/taskapi/upload"
    req = urllib_request.Request(url=url, data=body, method="POST", headers=headers)
    with urllib_request.urlopen(req, timeout=120) as resp:
        parsed = json.loads(resp.read().decode("utf-8", errors="replace"))
    file_url = str(((parsed.get("data") or {}).get("fileUrl") or "")).strip()
    if not file_url:
        raise RuntimeError(f"AIG upload did not return fileUrl: {parsed}")
    return file_url


def _zip_skill_dir(skill_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="aig-scan-"))
    archive = tmpdir / "skill.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(skill_path.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(skill_path)))
    return archive


def scan_skill(
    skill_path: Path,
    *,
    base_url: str = "http://localhost:18088",
    model: str = "",
    api_key: str = "",
    model_base_url: str = "",
    poll_timeout: float = 900.0,
    poll_interval: float = 5.0,
    cache_dir: Path | None = None,
) -> AIGResult:
    skill_id = skill_path.name

    if cache_dir:
        cache_file = cache_dir / f"{skill_id}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            return AIGResult(
                skill_id=skill_id,
                findings=[AIGFinding(**f) for f in cached.get("findings", [])],
                readme=cached.get("readme", ""),
                session_id=cached.get("session_id", ""),
                cached=True,
            )

    archive = _zip_skill_dir(skill_path)
    try:
        file_url = _upload_archive(archive, base_url)

        payload = {
            "type": "mcp_scan",
            "content": {
                "prompt": "Scan this skill for security vulnerabilities.",
                "model": {"model": model, "token": api_key, "base_url": model_base_url},
                "thread": 4,
                "language": "en",
                "attachments": file_url,
            },
        }
        resp = _request_json("POST", f"{base_url}/api/v1/app/taskapi/tasks", payload=payload)
        session_id = str(((resp.get("data") or {}).get("session_id") or "")).strip()
        if not session_id:
            raise RuntimeError(f"AIG task creation failed: {resp}")

        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            status = _request_json("GET", f"{base_url}/api/v1/app/taskapi/status/{session_id}")
            status_val = str(((status.get("data") or {}).get("status") or "")).lower()
            if status_val in {"done", "complete", "completed"}:
                break
            if status_val in {"error", "failed"}:
                raise RuntimeError(f"AIG task failed: {status}")
            time.sleep(poll_interval)
        else:
            raise TimeoutError(f"AIG task {session_id} timed out")

        result_resp = _request_json("GET", f"{base_url}/api/v1/app/taskapi/result/{session_id}")
        data = result_resp.get("data") or {}
        native = data.get("result") or data.get("event", {}).get("result") or {}

        findings = []
        for item in native.get("results") or []:
            if isinstance(item, dict):
                findings.append(AIGFinding(
                    title=str(item.get("title", "")),
                    description=str(item.get("description", "")),
                    risk_type=str(item.get("risk_type", "")),
                    level=str(item.get("level", "")),
                    suggestion=str(item.get("suggestion", "")),
                ))

        result = AIGResult(
            skill_id=skill_id,
            findings=findings,
            readme=str(native.get("readme", "")),
            session_id=session_id,
        )

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{skill_id}.json"
            cache_file.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result
    finally:
        shutil.rmtree(str(archive.parent), ignore_errors=True)
