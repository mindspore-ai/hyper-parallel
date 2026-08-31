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
"""Business-neutral Agentic RL contracts and runtime orchestration."""

from rl.agentic.core.runner import AgentRunner
from rl.agentic.core.session import AgentSession
from rl.agentic.core.types import (
    Action,
    AgentAction,
    EpisodeContext,
    EpisodeResult,
    InteractionMode,
    Observation,
    RewardResult,
    TerminationReason,
    ToolCall,
    ToolResult,
    Transition,
    TurnContext,
    TurnResult,
)
from rl.agentic.envs.base import Environment
from rl.agentic.envs.environment import (
    ENVIRONMENTS,
    RewardFunction,
    ToolEnvironment,
    load_agentic_module,
)
from rl.agentic.core.program_runner import AgentProgram, ProgramAgentRunner
from rl.agentic.tools import Tool, ToolExecutor, ToolHandler, ToolRegistry
from rl.agentic.tools.protocol import (
    INTERACTION_PROTOCOLS,
    InteractionProtocol,
    JsonFunctionCallProtocol,
    OpenAIToolCallProtocol,
    ParsedAction,
    ResponseParser,
    ToolExecutorProtocol,
)
from rl.algorithm.reward import compute_rule_reward, extract_answer


__all__ = [
    "Action",
    "AgentAction",
    "AgentProgram",
    "AgentRunner",
    "AgentSession",
    "ENVIRONMENTS",
    "Environment",
    "EpisodeContext",
    "EpisodeResult",
    "INTERACTION_PROTOCOLS",
    "InteractionMode",
    "InteractionProtocol",
    "JsonFunctionCallProtocol",
    "Observation",
    "OpenAIToolCallProtocol",
    "ParsedAction",
    "ProgramAgentRunner",
    "ResponseParser",
    "RewardFunction",
    "RewardResult",
    "TerminationReason",
    "Tool",
    "ToolCall",
    "ToolEnvironment",
    "ToolExecutor",
    "ToolExecutorProtocol",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
    "Transition",
    "TurnContext",
    "TurnResult",
    "compute_rule_reward",
    "extract_answer",
    "load_agentic_module",
]
