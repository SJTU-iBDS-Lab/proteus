"""Logging bootstrap for the red-team campaign runner.

Two entry points:

* ``bootstrap_from_ini(path)`` — call once at process start.  Reads an
  ini file via :func:`logging.config.fileConfig`.  The default ini at
  ``configs/logging/log.ini`` configures a stderr console handler
  (brief format, INFO+) and a rotating file handler whose path is
  taken from ``$LOG_FILE`` (default ``/tmp/claw_run.log``).

* ``attach_run_file_handler(run_dir)`` — context-manager helper.  Adds
  a ``run.log`` FileHandler to the root logger pointing at
  ``<run_dir>/run.log`` so every campaign also gets its own
  per-run log file alongside the JSONL artifacts.  The handler is
  removed cleanly on exit so concurrent runs don't cross-contaminate.

The runner uses these to emit INFO-level progress events at every
phase boundary (audit start/done, mutator start/done, sandbox
start/done, ★ outcome, expansion start/done).
"""
from __future__ import annotations

import logging
import logging.config
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = [
    "DEFAULT_INI_PATH",
    "bootstrap_from_ini",
    "attach_run_file_handler",
]

DEFAULT_INI_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs" / "logging" / "log.ini"
)


def bootstrap_from_ini(path: str | Path | None = None) -> None:
    """Configure root logging from an ini file.

    Falls back silently to ``logging.basicConfig`` if the ini file is
    missing — keeps unit tests / ad-hoc imports working.
    """
    ini = Path(path) if path else DEFAULT_INI_PATH
    if not ini.is_file():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger(__name__).debug(
            "log.ini not found at %s; using basicConfig fallback", ini
        )
        return
    logging.config.fileConfig(str(ini), disable_existing_loggers=False)


@contextmanager
def attach_run_file_handler(
    run_dir: str | Path,
    *,
    filename: str = "run.log",
    level: int = logging.DEBUG,
) -> Iterator[logging.Handler]:
    """Attach a per-run FileHandler to the root logger for the duration.

    The handler writes to ``<run_dir>/<filename>`` with the same
    ``detailed`` format as the global file handler (module name + level
    + message).  Removed on context exit so a long-lived process can
    drive multiple runs without log cross-contamination.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / filename
    handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def emit_run_banner(
    logger: logging.Logger,
    *,
    run_id: str,
    seed_id: str,
    repo_id: str,
    target_model: str,
    mutator_model: str,
    blue_team: str,
    rounds: int,
    out_dir: str | Path,
) -> None:
    """One-shot 'run starting' banner with the same fields a human would
    grep for in postmortem (run_id, seed, target, blue-team, budget)."""
    bar = "=" * 72
    logger.info(bar)
    logger.info("RUN START  run_id=%s", run_id)
    logger.info("  seed=%s  repo=%s", seed_id, repo_id)
    logger.info(
        "  target=%s  mutator=%s  blue_team=%s",
        target_model, mutator_model, blue_team,
    )
    logger.info("  rounds<=%d  out=%s", rounds, out_dir)
    logger.info(bar)


def emit_run_complete(
    logger: logging.Logger,
    *,
    run_id: str,
    rounds_ran: int,
    first_bypass_round: int | None,
    total_bypass: int,
    total_lethal: int,
    total_cost_usd: float,
    star_round: int | None,
) -> None:
    bar = "=" * 72
    logger.info(bar)
    if star_round is not None:
        logger.info(
            "RUN DONE   ★ at round %d  rounds=%d  cost=$%.4f",
            star_round, rounds_ran, total_cost_usd,
        )
    else:
        logger.info(
            "RUN DONE   no ★  rounds=%d  bypass=%d  lethal=%d  cost=$%.4f",
            rounds_ran, total_bypass, total_lethal, total_cost_usd,
        )
    logger.info(
        "  first_bypass_round=%s  total_bypass=%d  total_lethal=%d",
        first_bypass_round, total_bypass, total_lethal,
    )
    logger.info(bar)
