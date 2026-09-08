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
from rl.agentic import AgentRunner, ProgramAgentRunner
from rl.agentic.codex import CodexProgramFactory, CodexRuntime
from rl.agentic.deepseek import DeepSeekProgramFactory, DeepSeekRuntime
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
        ignore_eos: bool = False,
        max_episode_tokens: Optional[int] = None,
        environment_settings: Optional[dict[str, Any]] = None,
        interaction_mode: Optional[str] = None,
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
            ignore_eos=ignore_eos,
        )
        self.agent_runner = AgentRunner(
            engine=engine,
            tokenizer=tokenizer,
            environment_name=environment_name,
            num_samples=num_return_sequences,
            max_turns=max_turns,
            max_observation_tokens=max_observation_tokens,
            settings=settings,
            max_episode_tokens=max_episode_tokens,
            environment_settings=environment_settings,
            interaction_mode=interaction_mode,
        )

    def generate(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int = 0,
    ) -> ExperienceBatch:
        """Run agent episodes and return their padded training batch."""
        return self.agent_runner.rollout(prompt_records, policy_version)


class CodexRolloutManager:
    """Run complete Codex CLI programs against the shared vLLM policy."""

    def __init__(
        self,
        runtime: CodexRuntime,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        pad_token_id: int,
        eos_token_id: int,
        do_sample: bool = True,
        collect_old_log_probs: bool = True,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
    ) -> None:
        """Build the contract-only ProgramAgentRunner around Codex."""
        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            eos_token_ids=(eos_token_id,),
            collect_log_probs=collect_old_log_probs,
            seed=seed,
            ignore_eos=ignore_eos,
        )
        self.runtime = runtime
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "ignore_eos": ignore_eos,
        }
        self.agent_runner = ProgramAgentRunner(
            program_factory=CodexProgramFactory(
                runtime,
                eos_token_id,
                generation_config,
            ),
            num_samples=num_return_sequences,
            settings=settings,
            engine=runtime.engine,
        )

    def generate(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int = 0,
    ) -> ExperienceBatch:
        """Start the shared gateway and execute complete Codex episodes."""
        self.runtime.ensure_started()
        identity_before = self.runtime.engine.generation_policy_identity()
        if identity_before[0] != policy_version:
            raise RuntimeError(
                "Codex requested policy version does not match the served policy: "
                f"requested={policy_version}, served={identity_before[0]}"
            )
        self.runtime.bind_episode_identity(identity_before)
        try:
            result = self.agent_runner.rollout(prompt_records, policy_version)
        finally:
            self.runtime.clear_episode_identity()
        identity_after = self.runtime.engine.generation_policy_identity()
        if identity_after != identity_before:
            raise RuntimeError(
                "vLLM policy identity changed while Codex was executing an episode: "
                f"before={identity_before}, after={identity_after}"
            )
        return result

    def close(self) -> None:
        """Release the node-level gateway when this manager owns it."""
        self.runtime.close()


class DeepSeekRolloutManager:
    """Run DeepSeek Harness programs against the shared vLLM policy."""

    def __init__(
        self,
        runtime: DeepSeekRuntime,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        pad_token_id: int,
        eos_token_id: int,
        do_sample: bool = True,
        collect_old_log_probs: bool = True,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
    ) -> None:
        """Build the contract-only ProgramAgentRunner around DeepSeek Harness."""
        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            eos_token_ids=(eos_token_id,),
            collect_log_probs=collect_old_log_probs,
            seed=seed,
            ignore_eos=ignore_eos,
        )
        self.runtime = runtime
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "ignore_eos": ignore_eos,
        }
        self.agent_runner = ProgramAgentRunner(
            program_factory=DeepSeekProgramFactory(
                runtime,
                eos_token_id,
                generation_config,
            ),
            num_samples=num_return_sequences,
            settings=settings,
            engine=runtime.engine,
        )

    def generate(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int = 0,
    ) -> ExperienceBatch:
        """Start the DeepSeek gateway and execute complete Harness episodes."""
        self.runtime.ensure_started()
        identity_before = self.runtime.engine.generation_policy_identity()
        if identity_before[0] != policy_version:
            raise RuntimeError(
                "DeepSeek requested policy version does not match the served policy: "
                f"requested={policy_version}, served={identity_before[0]}"
            )
        self.runtime.bind_episode_identity(identity_before)
        try:
            result = self.agent_runner.rollout(prompt_records, policy_version)
        finally:
            self.runtime.clear_episode_identity()
        identity_after = self.runtime.engine.generation_policy_identity()
        if identity_after != identity_before:
            raise RuntimeError(
                "vLLM policy identity changed while DeepSeek Harness was executing: "
                f"before={identity_before}, after={identity_after}"
            )
        return result

    def close(self) -> None:
        """Release only the DeepSeek node-level gateway."""
        self.runtime.close()
