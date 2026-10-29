"""Minimal entry-script fixture for tests.

Not meant to run — just small enough to fit well under the 4KB read
cap that :func:`red_team.agent.tools._tool_read_skill` applies.
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1:
        print(sys.argv[1])


if __name__ == "__main__":
    main()
