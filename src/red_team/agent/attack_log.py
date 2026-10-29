"""兼容转发 shim — 真实实现在 :mod:`red_team.log.attack_log`.

历史调用方 ``from red_team.agent.attack_log import append, search`` 继续
可用；新代码请直接 import :mod:`red_team.log.attack_log`.
"""
from __future__ import annotations

from red_team.log.attack_log import (
    AttackLogEntry,
    append,
    read_all,
    row_to_document,
    search,
)

__all__ = [
    "AttackLogEntry",
    "append",
    "read_all",
    "row_to_document",
    "search",
]
