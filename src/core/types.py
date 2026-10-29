from __future__ import annotations

from typing import Any

JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
JsonDict = dict[str, JsonValue]
StringMap = dict[str, str]
MetadataMap = dict[str, Any]
