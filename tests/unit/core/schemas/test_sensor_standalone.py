"""Tests pinning that :class:`core.schemas.SensorReading` is the
authoritative, standalone definition — not a re-export of
``sandbox.honeypot.HoneypotResult``.

These invariants guard the one-way layering contract: ``core.*`` must
never depend on ``sandbox.*``. Historically ``SensorReading`` was an
``as``-import alias of ``HoneypotResult``; that reverse dependency is
now removed and ``HoneypotResult`` is redefined as an alias back to
``SensorReading`` for backward compatibility.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import core.schemas.sensor as _sensor_module
from core.schemas.sensor import SensorReading
from sandbox.honeypot import HoneypotResult


def test_sensor_reading_is_slotted_dataclass() -> None:
    """``SensorReading`` must be a slotted ``@dataclass``.

    Note: this class is intentionally NOT ``frozen`` because the
    legacy ``check_canaries`` pipeline in ``sandbox.honeypot`` mutates
    the record in-place (``result.evidence.append``,
    ``result.attack_succeeded = ...``). The hard backward-compat
    constraint forbids changing that call pattern.
    """
    assert dataclasses.is_dataclass(SensorReading)
    # ``slots=True`` generates ``__slots__`` on the class.
    assert hasattr(SensorReading, "__slots__"), "SensorReading must be slotted"


def test_sensor_reading_has_all_honeypot_fields() -> None:
    """Every field historically defined on ``HoneypotResult`` must
    exist on ``SensorReading`` with the same name and default factory
    semantics."""
    expected_fields = {
        "canary_env_read": False,
        "canary_ssh_read": False,
        "canary_token_leaked": False,
        "network_attempts": False,
        "attack_succeeded": False,
    }
    list_fields = {
        "files_created",
        "files_deleted",
        "files_modified",
        "commands_executed",
        "evidence",
    }

    field_map = {f.name: f for f in dataclasses.fields(SensorReading)}
    for name, default in expected_fields.items():
        assert name in field_map, f"missing field: {name}"
        assert field_map[name].default is default

    for name in list_fields:
        assert name in field_map, f"missing list field: {name}"
        factory = field_map[name].default_factory  # type: ignore[attr-defined]
        assert factory is list, f"{name} must default to list"


def test_sensor_reading_constructs_with_all_fields() -> None:
    """Construction with every field must succeed (byte-equal contract
    with legacy ``HoneypotResult`` keyword arguments)."""
    reading = SensorReading(
        canary_env_read=True,
        canary_ssh_read=True,
        canary_token_leaked=True,
        files_created=["/tmp/a"],
        files_deleted=["/tmp/b"],
        files_modified=["/tmp/c"],
        commands_executed=["bash"],
        network_attempts=True,
        attack_succeeded=True,
        evidence=["canary_token_in_output"],
    )
    assert reading.canary_env_read is True
    assert reading.files_created == ["/tmp/a"]
    assert reading.evidence == ["canary_token_in_output"]


def test_sensor_reading_exposes_to_dict() -> None:
    """The legacy ``to_dict`` method must survive the migration."""
    reading = SensorReading(canary_token_leaked=True, evidence=["x"])
    payload = reading.to_dict()
    assert payload["canary_token_leaked"] is True
    assert payload["evidence"] == ["x"]
    assert payload["files_created"] == []


def test_honeypot_result_is_sensor_reading_alias() -> None:
    """``HoneypotResult`` must be the *same object* as
    ``SensorReading`` so all legacy construction sites (oracle judges,
    driver, tests) keep working byte-equal."""
    assert HoneypotResult is SensorReading


def test_core_schemas_sensor_does_not_import_sandbox() -> None:
    """``core.schemas.sensor`` must not import from ``sandbox.*``.

    This invariant is verified by parsing the module source AST so a
    lazy/conditional ``from sandbox...`` inside a function would still
    be caught.
    """
    module_path = Path(_sensor_module.__file__)
    assert module_path.name == "sensor.py"
    assert module_path.parent.name == "schemas"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "sandbox":
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "sandbox":
                    bad.append(f"import {alias.name}")
    assert not bad, f"core.schemas.sensor leaks sandbox imports: {bad}"
