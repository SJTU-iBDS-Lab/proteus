"""Wave 2 — ``runtime._make_client`` delegates to ``llm.litellm``.

Goal
----
The LLM-client construction path has been consolidated into
``src/llm/litellm.py`` (Wave 1).  ``red_team.agent.runtime`` must now
delegate rather than define its own OpenAI-client factory, **while
keeping the module-level symbol ``runtime._make_client`` intact** — the
existing runtime tests (budget/invariant/refusal) monkey-patch that
symbol and must continue to work unchanged.

Three invariants under test:

1. ``runtime._make_client(base, key)`` returns whatever
   ``llm.litellm._make_openai_client`` returns (i.e. the work is
   delegated, not duplicated).
2. ``runtime._DEFAULT_LITELLM_URL`` / ``_DEFAULT_LITELLM_KEY``, if still
   present, are the **same object** as their ``llm.litellm``
   counterparts — guaranteeing no value drift.
3. Importing ``red_team.agent.runtime`` and ``llm.litellm`` must not
   import ``openai`` (the dependency stays lazy, so test environments
   without the package can still load the runtime).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_make_client_delegates_to_llm_litellm(monkeypatch) -> None:
    """``runtime._make_client`` must forward to ``llm.litellm._make_openai_client``.

    We patch the factory inside ``llm.litellm`` to a sentinel and assert
    the runtime wrapper returns *that* sentinel, with the base_url /
    api_key passed through unchanged.
    """
    from llm import litellm as litellm_module
    from red_team.agent import runtime

    captured: dict = {}
    sentinel = object()

    def fake_make_openai_client(base_url: str, api_key: str):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return sentinel

    monkeypatch.setattr(
        litellm_module, "_make_openai_client", fake_make_openai_client
    )

    result = runtime._make_client("http://proxy.example", "sk-unit")

    assert result is sentinel
    assert captured == {
        "base_url": "http://proxy.example",
        "api_key": "sk-unit",
    }


def test_runtime_defaults_match_llm_litellm_constants() -> None:
    """Runtime's default URL/KEY must stay synchronised with ``llm.litellm``.

    If a future refactor drops the runtime-level constants entirely,
    this test must still hold — at minimum the values must match so
    downstream ``os.environ`` fallback semantics do not drift.
    """
    from llm.litellm import (
        _DEFAULT_LITELLM_KEY as LLM_KEY,
        _DEFAULT_LITELLM_URL as LLM_URL,
    )
    from red_team.agent import runtime

    # Runtime either re-exports the same identifiers or has removed
    # them — both are acceptable; if present they must match values.
    runtime_url = getattr(runtime, "_DEFAULT_LITELLM_URL", LLM_URL)
    runtime_key = getattr(runtime, "_DEFAULT_LITELLM_KEY", LLM_KEY)
    assert runtime_url == LLM_URL
    assert runtime_key == LLM_KEY


def test_runtime_imports_without_openai_installed(monkeypatch) -> None:
    """Importing the runtime must not trigger ``import openai``.

    The factory is lazy; nothing in ``red_team.agent.runtime`` nor
    ``llm.litellm`` should import ``openai`` at module load time.  We
    simulate a missing ``openai`` by stubbing ``sys.modules["openai"]``
    to ``None`` and *re-importing* both modules; the import must succeed
    (because ``openai`` is referenced only inside
    ``_make_openai_client``).
    """
    # Ensure a fresh import — drop any cached copies including openai,
    # otherwise the meta-path finder is never consulted (sys.modules hit
    # short-circuits the import machinery).
    purge = [n for n in sys.modules if n == "openai" or n.startswith("openai.")]
    for name in ("red_team.agent.runtime", "llm.litellm", "llm", *purge):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # Simulate "openai not installed": poison sys.modules so any
    # ``import openai`` raises ImportError.  This is more reliable than
    # a meta-path finder because Python consults sys.modules first.
    class _ExplodingModule:
        def __getattr__(self, item):  # pragma: no cover - trivial
            raise ImportError(
                f"openai is not installed (test stub); tried to access {item!r}"
            )

    # Use a loader hook via sys.modules: setting the entry to None makes
    # Python raise ModuleNotFoundError on any subsequent ``import openai``.
    monkeypatch.setitem(sys.modules, "openai", None)
    try:
        # These imports MUST succeed despite openai being unimportable.
        import llm.litellm  # noqa: F401
        import red_team.agent.runtime  # noqa: F401

        # Sanity: the factory symbol exists and is callable.
        from red_team.agent.runtime import _make_client

        assert callable(_make_client)

        # Actually calling it should raise ImportError (openai missing),
        # proving the lazy-import path routes through openai — but the
        # module-load path did not.
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _make_client("http://x", "y")
    finally:
        # monkeypatch restores sys.modules automatically on test exit,
        # so no explicit cleanup is needed here.
        pass
