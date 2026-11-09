"""Phase 18: allowlisted, deterministic CPU agent-tool execution.

No dynamic imports, shell execution, network or filesystem permissions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    argument_names: tuple[str, ...]


class SafeToolRegistry:
    """Functions are registered by trusted application code, never by model text."""

    def __init__(self, *, max_calls: int = 16, max_payload_bytes: int = 16_384):
        if max_calls <= 0 or max_payload_bytes <= 0:
            raise ValueError("resource limits must be positive")
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}
        self.max_calls = max_calls
        self.max_payload_bytes = max_payload_bytes
        self.calls = 0

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        if (
            not spec.name.isidentifier()
            or spec.name.startswith("_")
            or spec.name in self._tools
            or not callable(fn)
        ):
            raise ValueError("invalid, duplicated or uncallable tool")
        if len(set(spec.argument_names)) != len(spec.argument_names) or any(
            not n.isidentifier() or n.startswith("_") for n in spec.argument_names
        ):
            raise ValueError("argument names must be unique public identifiers")
        self._tools[spec.name] = (spec, fn)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "arguments": list(spec.argument_names),
            }
            for spec, _ in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools or not isinstance(arguments, dict):
            raise ValueError("unknown tool or invalid arguments")
        spec, fn = self._tools[name]
        if set(arguments) != set(spec.argument_names):
            raise ValueError("unexpected or missing tool arguments")
        encoded = json.dumps(arguments, allow_nan=False)
        if len(encoded.encode("utf-8")) > self.max_payload_bytes:
            raise ValueError("tool argument size exceeds limit")
        if self.calls >= self.max_calls:
            raise RuntimeError("tool call budget exhausted")
        self.calls += 1
        result = fn(**arguments)
        payload = json.dumps(result, allow_nan=False)
        if len(payload.encode("utf-8")) > self.max_payload_bytes:
            raise ValueError("tool result exceeds size limit")
        return result


def cpu_math_registry() -> SafeToolRegistry:
    """Provide safe arithmetic demonstrations, without arbitrary code execution."""
    registry = SafeToolRegistry()

    def add(a: float, b: float) -> float:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)):
            raise ValueError("finite numeric values required")
        result = float(a + b)
        if not math.isfinite(result):
            raise ValueError("result is not finite")
        return result

    registry.register(ToolSpec("add", "Add two finite numbers", ("a", "b")), add)
    return registry
