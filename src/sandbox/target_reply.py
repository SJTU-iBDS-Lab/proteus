"""Pull the target agent's latest assistant reply out of the sandbox.

Lifted out of :mod:`red_team.campaign.runner` because the logic is
sandbox-shaped and has nothing to do with the campaign loop itself.
The runner just wants the string; the parsing details belong here
next to the other sandbox helpers.

This is gray-box observability, not a bypass tip — the reply is fed
to ``_synthesize_runtime_feedback`` so the next round's mutator can
reason about *why* the previous primitive failed (allowlist deny vs.
alignment refusal) without us spoiling the framing.

Two sources, primary first:

1. **OpenClaw container session JSONL**
   ``$root/.openclaw/agents/main/sessions/<id>.jsonl`` — assistant
   message blocks of type ``text``.  Works for targets that emit at
   least one tool call (glm-5 path); breaks for strong alignment
   targets (gpt-5.4-mini) because OpenClaw labels content-only refusal
   responses as ``payloads=0`` and writes ``assistant []`` (zero
   content blocks) into the session log.

2. **YesCode adapter traffic log** (``/tmp/yescode_traffic.jsonl``)
   The adapter sees the raw model response BEFORE OpenClaw filters
   it.  When source #1 returns empty, we fall back to the most recent
   adapter entry whose request prompt contains a fingerprint of the
   chain we're running (host skill name).  This recovers the
   alignment-refusal text that source #1 dropped, so the mutator gets
   actual diagnostic ("I can't reveal credential material...")
   instead of a generic ``no_actuation`` hint.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

__all__ = ["extract_target_reply"]


_TRAFFIC_LOG = os.environ.get(
    "YESCODE_TRAFFIC_LOG", "/tmp/yescode_traffic.jsonl"
)


def _from_openclaw_session(container: str, max_chars: int) -> str:
    """Source #1: read the OpenClaw container's session JSONL."""
    try:
        listing = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             "ls -t /root/.openclaw/agents/main/sessions/*.jsonl 2>/dev/null "
             "| grep -v trajectory | head -1"],
            capture_output=True, text=True, timeout=5,
        )
        latest = (listing.stdout or "").strip()
        if not latest:
            return ""
        catout = subprocess.run(
            ["docker", "exec", container, "sh", "-c", f"cat {latest}"],
            capture_output=True, text=True, timeout=10,
        )
        if catout.returncode != 0:
            return ""
        last_text = ""
        for ln in (catout.stdout or "").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("type") != "message":
                continue
            m = e.get("message", {}) or {}
            if m.get("role") != "assistant":
                continue
            cs = m.get("content")
            if isinstance(cs, list):
                for c in cs:
                    if isinstance(c, dict) and c.get("type") == "text":
                        tx = (c.get("text") or "").strip()
                        if tx:
                            last_text = tx
            elif isinstance(cs, str):
                last_text = cs
        return last_text[:max_chars]
    except Exception:
        return ""


def _from_yescode_traffic(
    chain_fingerprint: list[str] | None,
    since_ts: float | None,
    max_chars: int,
) -> str:
    """Source #2: scan the YesCode adapter's request/response log.

    The adapter logs each POST as one JSONL line with the request
    ``instructions`` + ``input`` (32K-ish system prompt + user prompt)
    AND the model's full ``response_text``.  We pick the most recent
    entry whose request contains every host skill name in the chain
    (chain identity is the cheapest unique fingerprint we have without
    plumbing a session-id header through OpenClaw → LiteLLM → adapter).

    When ``since_ts`` is set, entries older than that are skipped — the
    runner passes the sandbox-start timestamp so a slow-firing call
    from a previous round can't shadow this round's response.
    """
    if not os.path.exists(_TRAFFIC_LOG):
        return ""
    try:
        with open(_TRAFFIC_LOG, "rb") as fh:
            # Read tail (last 256KB ~ ~256 entries) — adapter log is
            # append-only so we don't need the full history; we want
            # the most recent matching entry.
            try:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 262144))
                blob = fh.read().decode("utf-8", errors="replace")
            except Exception:
                fh.seek(0)
                blob = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

    fp = [f.strip().lower() for f in (chain_fingerprint or []) if f]
    best_text = ""
    best_ts = -1.0
    for ln in blob.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        ts = float(obj.get("ts") or 0.0)
        if since_ts is not None and ts < since_ts:
            continue
        # Match: every fingerprint token must appear in instructions+input
        haystack = (
            (obj.get("instructions") or "")
            + "\n"
            + (obj.get("input") or "")
        ).lower()
        if fp and not all(token in haystack for token in fp):
            continue
        text = obj.get("response_text") or ""
        if text and ts > best_ts:
            best_ts = ts
            best_text = text
    return best_text[:max_chars]


def extract_target_reply(
    container: str | None,
    max_chars: int = 8000,
    *,
    chain_fingerprint: list[str] | None = None,
    since_ts: float | None = None,
) -> str:
    """Return the target agent's most recent assistant message text.

    Tries the OpenClaw session log first (works for targets that emit
    at least one tool call).  When that yields nothing — typical for
    strong-alignment targets that refuse with plain text only — falls
    back to the YesCode adapter's traffic log filtered by chain
    fingerprint and the call window.

    Returns ``""`` when neither source has anything; the caller
    treats empty as "no signal" and skips the reply-aware feedback
    path.
    """
    primary = _from_openclaw_session(container or "", max_chars)
    if primary:
        return primary
    if not chain_fingerprint:
        return ""
    return _from_yescode_traffic(chain_fingerprint, since_ts, max_chars)
