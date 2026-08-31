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
"""Environment lifecycle protocol and compatibility type exports."""

from typing import Protocol

from rl.agentic.core.types import (
    Action,
    AgentAction,
    EpisodeContext,
    Observation,
    ObservationEncoder,
    Transition,
    TurnContext,
    TurnResult,
)


class Environment(Protocol):
    """Stateful RL environment owned by exactly one episode."""

    async def reset(self, context: EpisodeContext) -> Observation:
        """Create or reset episode state and return the initial observation."""

    async def step(self, action: AgentAction, context: TurnContext) -> TurnResult:
        """Apply one model action and return reward, state, and termination."""

    async def close(self) -> None:
        """Release resources owned by the episode."""


__all__ = [
    "Action",
    "AgentAction",
    "Environment",
    "EpisodeContext",
    "Observation",
    "ObservationEncoder",
    "Transition",
    "TurnContext",
    "TurnResult",
]
