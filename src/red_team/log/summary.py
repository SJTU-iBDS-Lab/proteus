"""summary.json 终态写入 —— 从 driver.py 抽取。

driver.py 原本的写法是
``path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")``，
本模块保持完全一致的序列化参数，避免论文分析脚本 diff summary.json 时
出现格式噪声。
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from red_team.log.schema import SummaryRow

__all__ = ["write_summary", "read_summary"]


def write_summary(path: Path, summary: dict[str, Any] | SummaryRow) -> None:
    """Write ``summary.json`` (overwrite) with ``indent=2`` + ``ensure_ascii=False``.

    - Creates the parent directory if missing.
    - Accepts either a ``dict`` (legacy, driver.py channel) or a
      :class:`SummaryRow` DTO.
    - Overwrites any existing file — summary is a terminal roll-up.
    """
    if is_dataclass(summary) and not isinstance(summary, type):
        data: dict[str, Any] = asdict(summary)
    else:
        data = dict(summary)  # type: ignore[arg-type]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_summary(path: Path) -> dict[str, Any]:
    """Parse ``summary.json`` into a dict. Raises if file missing."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
