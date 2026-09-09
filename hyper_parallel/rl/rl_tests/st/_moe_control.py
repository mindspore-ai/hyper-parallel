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
"""Controlled single-turn samples for MoE system-test update verification."""

from decimal import Decimal, InvalidOperation

from rl.agentic.core.types import Action, EpisodeContext, Observation, Transition, TurnContext
from rl.agentic.envs.environment import ENVIRONMENTS

from examples.agents.gsm8k.agent import compute_gsm8k_reward, extract_answer, normalize_answer


def _control_target(ground_truth: str, sample_index: int) -> tuple[str, float]:
    """Return alternating correct and intentionally incorrect numeric targets."""
    answer = normalize_answer(str(ground_truth))
    if sample_index % 2 == 0:
        return answer, 1.0
    try:
        wrong = Decimal(answer) + 1
    except InvalidOperation as error:
        raise ValueError(f"MoE ST requires a numeric ground truth, got {ground_truth!r}") from error
    return format(wrong, "f"), 0.0


class MoEUpdateControlEnvironment:
    """Ask the real model for short positive/negative exact-match controls."""

    def __init__(self, context: EpisodeContext) -> None:
        """Bind one controlled response to an episode identity."""
        self.episode = context
        self.target, self.expected_reward = _control_target(context.ground_truth, context.sample_index)
        self._stepped = False

    def _matches(self, context: EpisodeContext) -> bool:
        """Compare stable episode identifiers without comparing tensor fields."""
        return (
            context.prompt_id == self.episode.prompt_id
            and context.policy_version == self.episode.policy_version
            and context.sample_index == self.episode.sample_index
        )

    async def reset(self, context: EpisodeContext) -> Observation:
        """Render a bounded copy task that preserves real model generation."""
        if not self._matches(context):
            raise ValueError("MoE ST environment context does not match its episode")
        content = (
            "/no_think\nThis is a system-test copy task. Do not solve or explain. "
            "Return exactly the following line and nothing else:\n"
            f"#### {self.target}"
        )
        return context.encode_observation(content, role="user")

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Apply the shipped exact-match scorer to the generated response."""
        if self._stepped or context.turn_index != 0 or not self._matches(context.episode):
            raise ValueError("MoE ST environment accepts one action from its bound episode")
        self._stepped = True
        reward = compute_gsm8k_reward(action.content, str(self.episode.ground_truth))
        return Transition(
            observation=Observation(
                content="",
                token_ids=action.token_ids.new_empty((0,)),
                metadata={"role": "environment"},
            ),
            reward=reward,
            done=True,
            info={
                "reward_components": {"correctness": reward},
                "expected_reward": self.expected_reward,
                "extracted_answer": extract_answer(action.content),
            },
        )

    async def close(self) -> None:
        """Close the stateless control environment."""
        return None


ENVIRONMENTS.register("moe_update_control")(MoEUpdateControlEnvironment)
