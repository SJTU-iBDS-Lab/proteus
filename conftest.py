"""Root-level pytest configuration.

Registers custom markers used across the integration test suite so pytest
does not emit PytestUnknownMarkWarning (and optionally so collection can
filter with ``-m integration`` / ``-m "not integration"``).
"""

from __future__ import annotations


def pytest_configure(config):
    """Register project-wide pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: requires live external services (LiteLLM providers, "
        "sandbox containers, OpenRouter API). Skipped by default when the "
        "services are unavailable.",
    )
