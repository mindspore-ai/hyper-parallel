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
"""Role-level facades for internal and external Agentic rollout."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from rl.agentic import AgentRunner, ProgramAgentRunner
from rl.agentic.codex import CodexProgramFactory, CodexRuntime
from rl.agentic.ds_harness import DeepSeekProgramFactory, DeepSeekRuntime
from rl.dataset.contracts import ExperienceBatch, PromptRecord
from rl.roles.rollout.base import GenerationEngine, GenerationSettings


def _generation_settings(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    pad_token_id: int,
    eos_token_id: int,
    do_sample: bool,
    collect_old_log_probs: bool,
    seed: Optional[int],
    ignore_eos: bool,
    eos_token_ids: Optional[Sequence[int]] = None,
) -> GenerationSettings:
    """Normalize sampling options into backend-neutral generation settings."""
    return GenerationSettings(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        eos_token_ids=(eos_token_id,) if eos_token_ids is None else tuple(eos_token_ids),
        collect_log_probs=collect_old_log_probs,
        seed=seed,
        ignore_eos=ignore_eos,
    )


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
        settings = _generation_settings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            collect_old_log_probs=collect_old_log_probs,
            seed=seed,
            ignore_eos=ignore_eos,
            eos_token_ids=eos_token_ids,
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


class _ProgramRolloutManager:
    """Share policy binding and batch construction across external harnesses."""

    program_factory: Any
    display_name: str

    def __init__(
        self,
        runtime: CodexRuntime | DeepSeekRuntime,
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
        """Build a contract-only program runner around an external harness."""
        settings = _generation_settings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            collect_old_log_probs=collect_old_log_probs,
            seed=seed,
            ignore_eos=ignore_eos,
        )
        self.runtime = runtime
        generation = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "ignore_eos": ignore_eos,
        }
        self.agent_runner = ProgramAgentRunner(
            program_factory=self.program_factory(runtime, eos_token_id, generation),
            num_samples=num_return_sequences,
            settings=settings,
            engine=runtime.engine,
        )

    def generate(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int = 0,
    ) -> ExperienceBatch:
        """Run external programs against one stable served policy version."""
        self.runtime.ensure_started()
        version_before = self.runtime.engine.generation_policy_version()
        if version_before != policy_version:
            raise RuntimeError(
                f"{self.display_name} requested policy version does not match the served policy: "
                f"requested={policy_version}, served={version_before}"
            )
        self.runtime.bind_episode_version(version_before)
        try:
            result = self.agent_runner.rollout(prompt_records, policy_version)
        finally:
            self.runtime.clear_episode_version()
        version_after = self.runtime.engine.generation_policy_version()
        if version_after != version_before:
            raise RuntimeError(
                f"vLLM policy version changed while {self.display_name} was executing: "
                f"before={version_before}, after={version_after}"
            )
        return result

    def close(self) -> None:
        """Release the node-level gateway owned by this manager."""
        self.runtime.close()


class CodexRolloutManager(_ProgramRolloutManager):
    """Run complete Codex CLI programs against the shared vLLM policy."""

    program_factory = CodexProgramFactory
    display_name = "Codex"


class DeepSeekRolloutManager(_ProgramRolloutManager):
    """Run DeepSeek Harness programs against the shared vLLM policy."""

    program_factory = DeepSeekProgramFactory
    display_name = "DeepSeek Harness"


__all__ = ["CodexRolloutManager", "DeepSeekRolloutManager", "RolloutManager"]
