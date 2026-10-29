"""Unit + integration tests for :mod:`red_team.payloads.binary_catalog`.

Coverage:

* Catalog shape (6 entries, objective coverage, non-empty metadata).
* ``build_binary`` compiles each source, caches it, and is idempotent.
* ``get_binary`` returns a Bash wrapper whose base64 payload round-trips
  to the exact ELF bytes on disk.
* End-to-end: the ``honeypot_beacon`` wrapper, when executed by bash
  against a local httptest-style HTTP server, actually POSTs a payload
  file to the expected URL with the configured header.  This is the
  canonical evidence that the whole pipeline works — Go source → ELF →
  base64 wrapper → bash exec → real side effect.

Go is required.  If it isn't on PATH, the build-dependent tests skip
gracefully (they'll still fail CI where Go *should* be installed).
"""
from __future__ import annotations

import base64
import http.server
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from red_team.payloads import (
    BINARY_CATALOG,
    BinaryBuildError,
    BinaryNotFoundError,
    BinarySpec,
    build_binary,
    build_root,
    get_binary,
    list_binary_specs,
    summarize_catalog,
)


EXPECTED_BINARY_IDS = {
    "honeypot_beacon",
    "rc_installer",
    "privesc_grant",
    "content_tamperer",
    "file_shredder",
    "remote_stager",
}

EXPECTED_OBJECTIVES = {
    "data_exfiltration",
    "persistence",
    "privilege_escalation",
    "cleanup",
    "supply_chain_abuse",
    "cross_skill_chaining",
}


_GO_AVAILABLE = shutil.which("go") is not None
requires_go = pytest.mark.skipif(
    not _GO_AVAILABLE, reason="go toolchain unavailable"
)


@pytest.fixture()
def isolated_build_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the binary cache to a temp dir so parallel tests
    don't stomp on each other and we never pollute the real tree."""
    monkeypatch.setenv("CLAW_BINARY_BUILD_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


def test_catalog_has_exactly_six_entries() -> None:
    assert set(BINARY_CATALOG) == EXPECTED_BINARY_IDS
    assert len(BINARY_CATALOG) == 6


def test_catalog_covers_all_non_guidance_objectives() -> None:
    covered: set[str] = set()
    for spec in list_binary_specs():
        covered.update(spec.covers_objectives)
    assert EXPECTED_OBJECTIVES.issubset(covered), (
        f"missing objectives: {EXPECTED_OBJECTIVES - covered}"
    )
    # Guidance-injection is document-level only — no binary.
    assert "guidance_injection" not in covered


def test_every_spec_has_source_and_metadata() -> None:
    for spec in list_binary_specs():
        assert isinstance(spec, BinarySpec)
        assert spec.filename, spec.binary_id
        assert spec.description, spec.binary_id
        # Go source must actually exist on disk.
        main_go = spec.source_dir() / "main.go"
        assert main_go.is_file(), f"missing {main_go}"


def test_summarize_catalog_excludes_binary_content() -> None:
    summary = summarize_catalog()
    assert len(summary) == 6
    for row in summary:
        assert "content" not in row
        assert "template" not in row
        for key in (
            "binary_id",
            "filename",
            "language",
            "description",
            "covers_objectives",
            "params",
            "default_params",
            "usage_example",
        ):
            assert key in row, f"summary row missing {key}"
        assert row["language"] == "go"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@requires_go
def test_build_binary_produces_executable_elf(isolated_build_root: Path) -> None:
    path = build_binary("honeypot_beacon")
    assert path.is_file()
    # ELF magic bytes: 0x7f 'E' 'L' 'F'.
    assert path.read_bytes()[:4] == b"\x7fELF"
    # Cached under the isolated build root.
    assert path.is_relative_to(isolated_build_root)


@requires_go
def test_build_binary_is_idempotent(isolated_build_root: Path) -> None:
    first = build_binary("file_shredder")
    mtime_1 = first.stat().st_mtime
    time.sleep(0.01)
    second = build_binary("file_shredder")
    mtime_2 = second.stat().st_mtime
    # Same path, cache honored (mtime unchanged).
    assert first == second
    assert mtime_1 == mtime_2


@requires_go
def test_build_binary_unknown_id_raises(isolated_build_root: Path) -> None:
    with pytest.raises(BinaryNotFoundError):
        build_binary("nonsense")


# ---------------------------------------------------------------------------
# get_binary wrapper rendering
# ---------------------------------------------------------------------------


def _extract_heredoc_blob(wrapper: str, marker: str = "____CLAW_BIN____") -> bytes:
    """Pull the base64 payload from between two marker lines."""
    lines = wrapper.splitlines()
    # The opening marker appears as a trailing token on the `base64 -d
    # <<'MARKER'...` line; the closing marker is its own line. We walk
    # the lines, collect everything strictly between.
    in_heredoc = False
    blob_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_heredoc:
            if marker in line and line.lstrip().startswith("base64 -d"):
                in_heredoc = True
            continue
        if stripped == marker:
            break
        blob_lines.append(stripped)
    return base64.b64decode("".join(blob_lines))


@requires_go
def test_get_binary_wrapper_embeds_exact_elf(isolated_build_root: Path) -> None:
    elf_path = build_binary("honeypot_beacon")
    expected = elf_path.read_bytes()

    rendered = get_binary("honeypot_beacon")
    assert _extract_heredoc_blob(rendered["content"]) == expected


@requires_go
def test_get_binary_exports_params_as_env(isolated_build_root: Path) -> None:
    rendered = get_binary(
        "honeypot_beacon",
        {
            "HONEYPOT_URL": "http://10.0.0.42:4444/x",
            "LABEL": "tag with space",
        },
    )
    body = rendered["content"]
    # shlex.quote only quotes when needed; values with special chars
    # (spaces) get single-quoted, simple values stay bare. We assert
    # both forms show up correctly.
    assert "export HONEYPOT_URL=http://10.0.0.42:4444/x" in body
    assert "export LABEL='tag with space'" in body
    assert rendered["params_used"]["HONEYPOT_URL"] == "http://10.0.0.42:4444/x"


@requires_go
def test_get_binary_unknown_id_raises(isolated_build_root: Path) -> None:
    with pytest.raises(BinaryNotFoundError):
        get_binary("does-not-exist")


def test_get_binary_without_build_requires_prebuilt_artifact(
    isolated_build_root: Path,
) -> None:
    """When build=False and no cache exists, we raise BinaryBuildError
    instead of silently returning an empty wrapper."""
    with pytest.raises(BinaryBuildError):
        get_binary("honeypot_beacon", build=False)


# ---------------------------------------------------------------------------
# End-to-end: compile → wrap → bash exec → real HTTP side effect
# ---------------------------------------------------------------------------


class _CollectHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n > 0 else b""
        self.server.received_body = body  # type: ignore[attr-defined]
        self.server.received_label = self.headers.get("X-Agent-Label", "")  # type: ignore[attr-defined]
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args, **kwargs) -> None:  # silence noise
        pass


@pytest.fixture()
def local_http_server():
    """httptest-style local server — no real egress."""
    srv = socketserver.TCPServer(("127.0.0.1", 0), _CollectHandler)
    srv.received_body = b""              # type: ignore[attr-defined]
    srv.received_label = ""              # type: ignore[attr-defined]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@requires_go
def test_wrapper_executes_and_posts_payload_end_to_end(
    isolated_build_root: Path, tmp_path: Path, local_http_server
) -> None:
    """The canonical evidence path:
    go build → base64 embed → bash wrapper → bash exec → real HTTP POST.
    We verify the httptest server saw the expected payload + header.
    """
    payload = tmp_path / "creds.txt"
    secret = b"AKIAFAKE=deadbeef-integration\n"
    payload.write_bytes(secret)

    host, port = local_http_server.server_address
    rendered = get_binary(
        "honeypot_beacon",
        {
            "HONEYPOT_URL": f"http://{host}:{port}/collect",
            "PAYLOAD_PATH": str(payload),
            "LABEL": "e2e-test",
            "TIMEOUT_S": "5",
        },
    )

    wrapper = tmp_path / "beacon.sh"
    wrapper.write_text(rendered["content"], encoding="utf-8")
    wrapper.chmod(0o755)

    # Give bash up to 10s — wrapper does base64 decode + HTTP round-trip.
    proc = subprocess.run(
        ["bash", str(wrapper)],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"wrapper failed: stderr={proc.stderr}"

    # Allow the server goroutine-equivalent to finish.
    for _ in range(20):
        if getattr(local_http_server, "received_body", b""):
            break
        time.sleep(0.05)

    assert getattr(local_http_server, "received_body", b"") == secret
    assert getattr(local_http_server, "received_label", "") == "e2e-test"
