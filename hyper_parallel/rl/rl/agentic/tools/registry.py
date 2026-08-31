# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""User-owned tool definitions and per-environment registration."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


ToolHandler = Callable[..., Any]
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Tool:
    """One callable exposed to an interaction protocol."""

    name: str
    handler: ToolHandler
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stable public tool metadata."""
        if _TOOL_NAME.fullmatch(self.name) is None:
            raise ValueError(
                "Tool name must contain 1-64 letters, digits, underscores, or dashes"
            )
        if not callable(self.handler):
            raise ValueError(f"Tool handler must be callable: {self.name}")
        if not isinstance(self.parameters, Mapping):
            raise ValueError(f"Tool parameters schema must be a mapping: {self.name}")
        object.__setattr__(self, "parameters", dict(self.parameters))


class ToolRegistry:
    """Per-environment registry that avoids global tool side effects."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        *,
        description: str = "",
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Register a handler and return it unchanged for decorator use."""

        def decorator(handler: ToolHandler) -> ToolHandler:
            tool = Tool(
                name=name,
                handler=handler,
                description=description,
                parameters=dict(parameters or {}),
            )
            if tool.name in self._tools:
                raise ValueError(f"Tool is already registered: {tool.name}")
            self._tools[tool.name] = tool
            return handler

        return decorator

    def get(self, name: str) -> Tool:
        """Return a named tool or report the local choices."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown tool '{name}'; available={sorted(self._tools)}"
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return tool names in deterministic order."""
        return tuple(sorted(self._tools))
