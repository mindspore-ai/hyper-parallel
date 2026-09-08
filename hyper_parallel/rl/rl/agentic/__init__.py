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

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Action": ("rl.agentic.core.types", "Action"),
    "AgentAction": ("rl.agentic.core.types", "AgentAction"),
    "AgentProgram": ("rl.agentic.core.program_runner", "AgentProgram"),
    "AgentRunner": ("rl.agentic.core.runner", "AgentRunner"),
    "AgentSession": ("rl.agentic.core.session", "AgentSession"),
    "ENVIRONMENTS": ("rl.agentic.envs.environment", "ENVIRONMENTS"),
    "Environment": ("rl.agentic.envs.base", "Environment"),
    "EpisodeContext": ("rl.agentic.core.types", "EpisodeContext"),
    "EpisodeResult": ("rl.agentic.core.types", "EpisodeResult"),
    "INTERACTION_PROTOCOLS": ("rl.agentic.tools.protocol", "INTERACTION_PROTOCOLS"),
    "InteractionMode": ("rl.agentic.core.types", "InteractionMode"),
    "InteractionProtocol": ("rl.agentic.tools.protocol", "InteractionProtocol"),
    "JsonFunctionCallProtocol": ("rl.agentic.tools.protocol", "JsonFunctionCallProtocol"),
    "Observation": ("rl.agentic.core.types", "Observation"),
    "OpenAIToolCallProtocol": ("rl.agentic.tools.protocol", "OpenAIToolCallProtocol"),
    "ParsedAction": ("rl.agentic.tools.protocol", "ParsedAction"),
    "ProgramAgentRunner": ("rl.agentic.core.program_runner", "ProgramAgentRunner"),
    "ResponseParser": ("rl.agentic.tools.protocol", "ResponseParser"),
    "RewardFunction": ("rl.agentic.envs.environment", "RewardFunction"),
    "RewardResult": ("rl.agentic.core.types", "RewardResult"),
    "TerminationReason": ("rl.agentic.core.types", "TerminationReason"),
    "Tool": ("rl.agentic.tools", "Tool"),
    "ToolCall": ("rl.agentic.core.types", "ToolCall"),
    "ToolEnvironment": ("rl.agentic.envs.environment", "ToolEnvironment"),
    "ToolExecutor": ("rl.agentic.tools", "ToolExecutor"),
    "ToolExecutorProtocol": ("rl.agentic.tools.protocol", "ToolExecutorProtocol"),
    "ToolHandler": ("rl.agentic.tools", "ToolHandler"),
    "ToolRegistry": ("rl.agentic.tools", "ToolRegistry"),
    "ToolResult": ("rl.agentic.core.types", "ToolResult"),
    "Transition": ("rl.agentic.core.types", "Transition"),
    "TurnContext": ("rl.agentic.core.types", "TurnContext"),
    "TurnResult": ("rl.agentic.core.types", "TurnResult"),
    "compute_rule_reward": ("rl.algorithm.reward", "compute_rule_reward"),
    "extract_answer": ("rl.algorithm.reward", "extract_answer"),
    "load_agentic_module": ("rl.agentic.envs.environment", "load_agentic_module"),
}


def __getattr__(name: str) -> Any:
    """Load public agentic symbols only when a caller requests them."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public symbols to interactive callers."""
    return sorted((*globals(), *_EXPORTS))


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
