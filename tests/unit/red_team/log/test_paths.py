"""Path-convention tests for src/red_team/log/paths.py.

The paths module is the single source of truth for
``outputs/evolution/<run_id>/...`` layout. Every log writer must go
through these helpers to avoid hardcoded string concatenation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from red_team.log import paths


# ------------------------------------------------------------ default root

def test_run_root_default() -> None:
    assert paths.run_root("abc") == Path("outputs/evolution/abc")


def test_campaign_log_path_default() -> None:
    assert paths.campaign_log_path("abc") == Path(
        "outputs/evolution/abc/log.jsonl"
    )


def test_attack_log_path_default() -> None:
    assert paths.attack_log_path("abc") == Path(
        "outputs/evolution/abc/attack_log.jsonl"
    )


def test_summary_path_default() -> None:
    assert paths.summary_path("abc") == Path(
        "outputs/evolution/abc/summary.json"
    )


def test_mutation_dir_default() -> None:
    assert paths.mutation_dir("abc") == Path(
        "outputs/evolution/abc/mutations"
    )


# ------------------------------------------------------------ explicit root

def test_run_root_custom_root(tmp_path: Path) -> None:
    assert paths.run_root("abc", root=tmp_path) == tmp_path / "abc"


def test_campaign_log_path_custom_root(tmp_path: Path) -> None:
    assert paths.campaign_log_path("abc", root=tmp_path) == (
        tmp_path / "abc" / "log.jsonl"
    )


def test_attack_log_path_custom_root(tmp_path: Path) -> None:
    assert paths.attack_log_path("abc", root=tmp_path) == (
        tmp_path / "abc" / "attack_log.jsonl"
    )


def test_summary_path_custom_root(tmp_path: Path) -> None:
    assert paths.summary_path("abc", root=tmp_path) == (
        tmp_path / "abc" / "summary.json"
    )


def test_mutation_dir_custom_root(tmp_path: Path) -> None:
    assert paths.mutation_dir("abc", root=tmp_path) == (
        tmp_path / "abc" / "mutations"
    )


# ---------------------------------------------------------- return type Path

@pytest.mark.parametrize(
    "fn",
    [
        paths.run_root,
        paths.campaign_log_path,
        paths.attack_log_path,
        paths.summary_path,
        paths.mutation_dir,
    ],
)
def test_all_path_helpers_return_path_objects(fn) -> None:
    out = fn("abc")
    assert isinstance(out, Path), f"{fn.__name__} must return Path, got {type(out)!r}"
    assert not isinstance(out, str)


@pytest.mark.parametrize(
    "fn",
    [
        paths.run_root,
        paths.campaign_log_path,
        paths.attack_log_path,
        paths.summary_path,
        paths.mutation_dir,
    ],
)
def test_all_path_helpers_return_path_with_custom_root(fn, tmp_path: Path) -> None:
    out = fn("abc", root=tmp_path)
    assert isinstance(out, Path)
