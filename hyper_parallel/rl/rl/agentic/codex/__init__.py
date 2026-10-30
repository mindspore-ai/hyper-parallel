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
"""Codex CLI contracts; host-side execution stays standard-library only."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CodexAgentProgram": ("harness", "CodexAgentProgram"),
    "CodexGateway": ("gateway", "CodexGateway"),
    "CodexProgramFactory": ("harness", "CodexProgramFactory"),
    "CodexResponsesProtocol": ("protocol", "CodexResponsesProtocol"),
    "CodexRuntime": ("harness", "CodexRuntime"),
    "build_codex_trajectory": ("harness", "build_codex_trajectory"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Load training dependencies only when a public harness contract is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
