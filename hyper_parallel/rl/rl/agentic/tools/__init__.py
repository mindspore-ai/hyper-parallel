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
"""Tool definitions, registration, and bounded execution."""

import sys

from rl.agentic.tools import executor as _executor
from rl.agentic.tools.executor import (
    INTERACTION_PROTOCOLS,
    InteractionProtocol,
    JsonFunctionCallProtocol,
    OpenAIToolCallProtocol,
    ParsedAction,
    ResponseParser,
    Tool,
    ToolCall,
    ToolExecutor,
    ToolExecutorProtocol,
    ToolHandler,
    ToolRegistry,
    ToolResult,
)


sys.modules.setdefault(f"{__name__}.registry", _executor)
sys.modules.setdefault(f"{__name__}.protocol", _executor)

__all__ = [
    "InteractionProtocol",
    "INTERACTION_PROTOCOLS",
    "JsonFunctionCallProtocol",
    "OpenAIToolCallProtocol",
    "ParsedAction",
    "ResponseParser",
    "Tool",
    "ToolCall",
    "ToolExecutor",
    "ToolExecutorProtocol",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
]
