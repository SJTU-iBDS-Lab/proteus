"""campaign 层日志写入 —— 从 driver.py 抽取。

保持 byte-equal 写出格式（``ensure_ascii=False``, ``default=str``）。
dict-as-row 是主通道（向后兼容），:class:`CampaignLogRow` DTO 是推荐的
新路径但尚未强制（driver.py 仍走超集 dict schema）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from red_team.log.schema import CampaignLogRow

__all__ = ["now_iso", "write_row", "read_rows"]


def now_iso() -> str:
    """UTC ISO-8601 timestamp, seconds resolution.

    Semantically identical to driver.py's original ``_now_iso`` helper.
    Callers use this for the ``timestamp`` field on every log row.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_row(path: Path, row: dict[str, Any] | CampaignLogRow) -> None:
    """Append one JSON line to ``path`` (``log.jsonl``).

    - Creates the parent directory if missing.
    - Writes UTF-8 with ``ensure_ascii=False`` and ``default=str`` — the
      same flags driver.py used inline so log rows remain byte-equal.
    - Accepts either a plain ``dict`` (legacy, the only channel driver.py
      uses today) or a :class:`CampaignLogRow` DTO (typed new callers).
    - Flushes immediately so per-round durability is preserved.
    """
    if is_dataclass(row) and not isinstance(row, type):
        payload: dict[str, Any] = asdict(row)
    else:
        payload = dict(row)  # type: ignore[arg-type]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str))
        fh.write("\n")
        fh.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Parse ``path`` into a list of dicts, skipping blank lines.

    Returns ``[]`` if the file does not exist — matches driver.py's
    "no run yet ⇒ empty" semantics.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
