"""Shared LLM client for blue-team scanners — routes through LiteLLM proxy.

Configuration sources:

* **Model selection** — read from ``configs/models.yaml`` under the
  ``blue_team.<layer>`` key.  Layer is one of
  ``gatekeeper`` / ``alignment`` / ``flow_sim``; falls back to
  ``blue_team.fallback`` when an unknown layer is requested.
* **Proxy endpoint + auth** — read from env ``LITELLM_URL`` /
  ``LITELLM_KEY`` (defaults: ``http://127.0.0.1:4000`` / ``sk-claw-local``).
* **Upstream credentials** — never read directly here.  ``.env``
  ``LLM_API_KEY`` is consumed by the LiteLLM proxy itself; this module
  only talks to the proxy.

This decoupling means changing the blue-team model is a single
``configs/models.yaml`` edit; rotating upstream keys is a single
``.env`` edit; switching proxies is a single ``LITELLM_URL`` env edit.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

_DEFAULT_LITELLM_URL = "http://127.0.0.1:4000"
_DEFAULT_LITELLM_KEY = "sk-claw-local"
_MODELS_YAML = _PROJECT_ROOT / "configs" / "models.yaml"


@lru_cache(maxsize=1)
def _load_models_yaml() -> dict:
    """Load configs/models.yaml once per process.

    Missing file → empty dict; downstream code falls back to env or
    raises a helpful error.  Import is lazy because not all consumers
    of this module need yaml support installed.
    """
    if not _MODELS_YAML.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _model_for_layer(layer: str) -> str:
    """Return the configured model for a blue-team layer.

    Resolution order: env override (``BLUE_TEAM_<LAYER>_MODEL``) →
    ``configs/models.yaml`` ``blue_team.<layer>`` → ``blue_team.fallback``
    → empty string.
    """
    env_key = f"BLUE_TEAM_{layer.upper()}_MODEL"
    if (override := os.environ.get(env_key, "").strip()):
        return override
    cfg = _load_models_yaml().get("blue_team", {}) or {}
    return cfg.get(layer) or cfg.get("fallback") or ""


def get_config(layer: str = "") -> dict[str, str]:
    """Return ``{model, api_key, base_url}`` for a given layer."""
    return {
        "model": _model_for_layer(layer or "fallback"),
        "api_key": os.environ.get("LITELLM_KEY", _DEFAULT_LITELLM_KEY).strip(),
        "base_url": os.environ.get("LITELLM_URL", _DEFAULT_LITELLM_URL).strip(),
    }


def is_available(layer: str = "") -> bool:
    cfg = get_config(layer)
    return bool(cfg["model"] and cfg["api_key"] and cfg["base_url"])


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    layer: str = "",
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    if not (model and api_key and base_url):
        cfg = get_config(layer)
        model = model or cfg["model"]
        api_key = api_key or cfg["api_key"]
        base_url = base_url or cfg["base_url"]

    if not (model and api_key and base_url):
        raise RuntimeError(
            f"Blue-team LLM not configured for layer={layer!r}. "
            "Check configs/models.yaml blue_team.<layer> and "
            "LITELLM_URL / LITELLM_KEY env (or accept defaults)."
        )

    # Normalize URL → exactly one /v1/chat/completions suffix.
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib_request.Request(url=url, data=data, method="POST", headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {body}")
    return str(choices[0].get("message", {}).get("content", ""))


def chat_json(
    messages: list[dict[str, str]],
    *,
    layer: str = "",
    temperature: float = 0.1,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Chat completion that parses the response as JSON.

    Handles three response shapes:

    1. Pure JSON (OpenAI-style well-behaved).
    2. Fenced JSON (``\\`\\`\\`json { ... } \\`\\`\\```).
    3. Chain-of-thought prefix + JSON somewhere in the body (e.g. GLM-5
       always emits reasoning content first).  We extract the first
       balanced ``{...}`` block and require a JSON object.
    """
    raw = chat_completion(messages, layer=layer, temperature=temperature, timeout=timeout)
    raw = raw.strip()

    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object from the LLM")
        return parsed

    extracted = _extract_first_json(raw)
    if isinstance(extracted, dict):
        return extracted
    if extracted is not None:
        raise ValueError("expected a JSON object from the LLM")
    return json.loads(raw)  # raise JSONDecodeError with line/col


def _extract_first_json(text: str) -> dict | list | None:
    """Scan ``text`` for the first syntactically valid top-level JSON
    object or array and return it; return ``None`` if nothing parses.
    """
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                    continue
                if c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find(open_ch, start + 1)
    return None
