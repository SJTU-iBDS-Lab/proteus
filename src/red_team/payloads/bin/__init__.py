"""Bin-layer payload plugins.

Each file wraps exactly one entry of :data:`binary_catalog.BINARY_CATALOG`
into a :class:`Payload`. The wrappers reuse
``binary_catalog.get_binary`` unchanged so the Go compile + base64 +
bash-wrapper pipeline is shared and there is no duplicate rendering
logic.
"""
from __future__ import annotations

from red_team.payloads.bin.content_tamperer import ContentTampererPayload
from red_team.payloads.bin.file_shredder import FileShredderPayload
from red_team.payloads.bin.honeypot_beacon import HoneypotBeaconPayload
from red_team.payloads.bin.privesc_grant import PrivescGrantPayload
from red_team.payloads.bin.rc_installer import RcInstallerPayload
from red_team.payloads.bin.remote_stager import RemoteStagerPayload


REGISTRY = {
    "honeypot_beacon": HoneypotBeaconPayload(),
    "rc_installer": RcInstallerPayload(),
    "privesc_grant": PrivescGrantPayload(),
    "content_tamperer": ContentTampererPayload(),
    "file_shredder": FileShredderPayload(),
    "remote_stager": RemoteStagerPayload(),
}


__all__ = [
    "REGISTRY",
    "ContentTampererPayload",
    "FileShredderPayload",
    "HoneypotBeaconPayload",
    "PrivescGrantPayload",
    "RcInstallerPayload",
    "RemoteStagerPayload",
]
