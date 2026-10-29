"""Process-wide context handle for the active :class:`MemoryRuntime`.

The runner sets this once per ``run_evolution`` call so deep callers
(``family_enumerate`` / ``surface_expand``) can fetch the live store
+ embedder without threading them through every signature in the
mutation pipeline.

Usage::

    with current_memory_runtime(memory_runtime):
        ...                       # inside this block, get_current() → memory_runtime
    # outside: get_current() → None

The context is implemented as a :class:`contextvars.ContextVar` so it
is safe under nested calls and (more importantly) under the
``ProcessPoolExecutor`` that the campaign runner uses — each child
process gets its own ContextVar instance.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator

from red_team.log.memory.runtime_hook import MemoryRuntime

__all__ = ["current_memory_runtime", "get_current_memory_runtime"]


_CURRENT: ContextVar[MemoryRuntime | None] = ContextVar(
    "current_memory_runtime", default=None
)


def get_current_memory_runtime() -> MemoryRuntime | None:
    return _CURRENT.get()


@contextlib.contextmanager
def current_memory_runtime(rt: MemoryRuntime | None) -> Iterator[None]:
    """Bind ``rt`` as the active memory runtime for the duration of the block."""
    token = _CURRENT.set(rt)
    try:
        yield
    finally:
        _CURRENT.reset(token)
