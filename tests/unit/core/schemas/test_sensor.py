"""Unit tests for :data:`core.schemas.sensor.SensorReading`.

``SensorReading`` is *currently* a semantic alias re-export of
:class:`sandbox.honeypot.HoneypotResult`. The alias lets the L1
judgement layer (``oracle/``) and the L2 mutation layer depend on
``core.schemas`` without importing ``sandbox.*`` directly. These tests
pin the aliasing contract so the upstream name change cannot silently
drift.

The invariants exercised here:

* ``SensorReading`` can be imported from ``core.schemas``.
* ``SensorReading`` is identically the same class object as
  ``sandbox.honeypot.HoneypotResult`` (``is`` comparison, not just
  ``==``).
* A minimally-initialised instance has the expected default state.
"""
from __future__ import annotations

from core.schemas import SensorReading
from sandbox.honeypot import HoneypotResult


def test_sensor_reading_is_honeypot_result_alias() -> None:
    assert SensorReading is HoneypotResult


def test_sensor_reading_constructs_with_defaults() -> None:
    reading = SensorReading()
    assert reading.canary_env_read is False
    assert reading.canary_ssh_read is False
    assert reading.canary_token_leaked is False
    assert reading.attack_succeeded is False
    assert reading.evidence == []
    assert reading.files_created == []
    assert reading.files_deleted == []
    assert reading.commands_executed == []


def test_sensor_reading_accepts_evidence_payload() -> None:
    reading = SensorReading(
        canary_token_leaked=True,
        evidence=["canary_token_in_output"],
        attack_succeeded=True,
    )
    assert reading.canary_token_leaked is True
    assert reading.evidence == ["canary_token_in_output"]
    assert reading.attack_succeeded is True
