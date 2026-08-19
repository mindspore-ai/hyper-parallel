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
"""Role-level facade around the default token-first AgentRunner."""
from typing import Any, Optional, Sequence
from rl.agentic import AgentRunner
from rl.dataset.contracts import ExperienceBatch, PromptRecord
from rl.roles.rollout.base import GenerationEngine, GenerationSettings
class RolloutManager:
    """Configure an AgentRunner for training or evaluation rollout."""
    def __init__(
        self,
        engine: GenerationEngine,
        tokenizer: Any,
        environment_name: str,
        num_return_sequences: int,
        max_turns: int,
        max_observation_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        pad_token_id: int,
        eos_token_id: int,
        do_sample: bool = True,
        collect_old_log_probs: bool = False,
        seed: Optional[int] = None,
        eos_token_ids: Sequence[int] = (),
    ) -> None:
        """Initialize rollout orchestration and generation settings."""
        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            eos_token_ids=tuple(eos_token_ids),
            collect_log_probs=collect_old_log_probs,
            seed=seed,
        )
        self.agent_runner = AgentRunner(
            engine=engine,
            tokenizer=tokenizer,
            environment_name=environment_name,
            num_samples=num_return_sequences,
            max_turns=max_turns,
            max_observation_tokens=max_observation_tokens,
            settings=settings,
        )
    def generate(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int = 0,
    ) -> ExperienceBatch:
        """Run agent episodes and return their padded training batch."""
        return self.agent_runner.rollout(prompt_records, policy_version)
