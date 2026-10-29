"""Unit tests for :mod:`red_team.campaign.seams`.

The runner receives five callable seams; this test verifies the module
exposes their default bindings as ``callable`` symbols so the runner can
fall back to them when a test does not inject a stub.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from red_team.campaign import seams


def test_default_agent_run_is_callable():
    assert callable(seams.default_agent_run)


def test_default_oracle_lethal_is_callable():
    assert callable(seams.default_oracle_lethal)


def test_default_horizontal_expand_is_callable():
    assert callable(seams.default_horizontal_expand)


def test_call_blue_team_honours_explicit_fn():
    """The blue-team seam dispatches to the injected callable first."""
    marker: list = []

    def fake_blue_team(_variant, _repo):
        marker.append(1)
        return []

    out = seams.call_blue_team(
        variant=None, repo=None, method="ours",
        blue_team_fn=fake_blue_team,
        sandbox_container="c",
    )
    assert out == []
    assert marker == [1]


def test_call_blue_team_rejects_unknown_method():
    """Passing an unknown method (without explicit seam) raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        seams.call_blue_team(
            variant=None, repo=None, method="nonsense",
            blue_team_fn=None,
            sandbox_container="c",
        )


def test_sandbox_run_accepts_seam_and_back_compat_tuple():
    """When the seam returns a 2-tuple the third slot must default to None."""
    def two_tuple_seam(_v, _c, _tm, _ct):
        return ("hp_value", [])

    hp, tool_calls, host = seams.sandbox_run(
        variant=None, chain=[], target_model="tm", container="c",
        sandbox_run_fn=two_tuple_seam,
    )
    assert hp == "hp_value"
    assert tool_calls == []
    assert host is None

    def three_tuple_seam(_v, _c, _tm, _ct):
        return ("hp2", [{"tool": "x"}], Path("/tmp/x"))

    hp, tool_calls, host = seams.sandbox_run(
        variant=None, chain=[], target_model="tm", container="c",
        sandbox_run_fn=three_tuple_seam,
    )
    assert hp == "hp2"
    assert tool_calls == [{"tool": "x"}]
    assert host == Path("/tmp/x")
