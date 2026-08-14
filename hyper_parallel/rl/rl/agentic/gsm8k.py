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
"""Built-in one-step GSM8K agentic environment."""

from rl.algorithm.reward import compute_rule_reward, extract_answer
from rl.agentic.base import Action, Observation, Transition
from rl.agentic.registry import ENVIRONMENTS
from rl.dataset.contracts import PromptRecord

class GSM8KEnvironment:
    """Treat math answer generation as a one-action agent episode."""
    def __init__(self, prompt: PromptRecord) -> None:
        """Initialize the episode for one prompt record."""
        self.prompt = prompt
        self._stepped = False

    async def reset(self, prompt: PromptRecord) -> Observation:
        """Validate the prompt and return its tokenized question."""
        if prompt.prompt_id != self.prompt.prompt_id:
            raise ValueError("Environment prompt does not match its registration record")
        token_ids = prompt.metadata.get("input_ids")
        if token_ids is None or token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError("GSM8K PromptRecord.metadata.input_ids must be a non-empty tensor")
        return Observation(
            content=prompt.messages[-1].content,
            token_ids=token_ids,
            metadata={"role": "user"},
        )
    async def step(self, action: Action) -> Transition:
        """Score the single generated answer and terminate the episode."""
        if self._stepped:
            raise RuntimeError("GSM8KEnvironment accepts exactly one action")
        self._stepped = True
        reward = compute_rule_reward(action.content, str(self.prompt.ground_truth))
        return Transition(
            observation=Observation(
                content="",
                token_ids=action.token_ids.new_empty((0,)),
                metadata={"role": "environment"},
            ),
            reward=reward,
            done=True,
            info={
                "reward_components": {"rule": reward},
                "extracted_answer": extract_answer(action.content),
            },
        )
    async def close(self) -> None:
        """Close the stateless GSM8K environment."""
        return None

@ENVIRONMENTS.register("gsm8k")
def build_gsm8k_environment(prompt: PromptRecord) -> GSM8KEnvironment:
    """Build a GSM8K environment for a registered prompt."""
    return GSM8KEnvironment(prompt)
