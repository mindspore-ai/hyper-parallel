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
"""Synchronous batched engine driver for token-first AgentSessions."""
import asyncio
from typing import Any, Optional, Sequence
from rl.agentic.base import Action
from rl.agentic.registry import ENVIRONMENTS
from rl.agentic.session import AgentSession
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory
from rl.dataset.batch_builder import build_experience_batch
from rl.roles.rollout.base import GenerationEngine, GenerationRequest, GenerationSettings
class AgentRunner:
    """Drive N sampled environments through one batched generation engine.

    The runner executes exactly ``max_turns`` generation calls on every rank.
    Finished sessions receive discarded dummy generations, preserving FSDP
    collective ordering without introducing a distributed scheduler.
    """
    def __init__(
        self,
        engine: GenerationEngine,
        tokenizer: Any,
        environment_name: str,
        num_samples: int,
        max_turns: int,
        max_observation_tokens: Optional[int],
        settings: GenerationSettings,
    ) -> None:
        """Initialize batched agent rollout settings and dependencies."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if max_observation_tokens is not None and max_observation_tokens < 0:
            raise ValueError("max_observation_tokens must be non-negative")
        self.engine = engine
        self.tokenizer = tokenizer
        self.environment_name = environment_name
        self.num_samples = num_samples
        self.max_turns = max_turns
        self.max_observation_tokens = max_observation_tokens
        self.settings = settings
    def _left_pad_sessions(
        self,
        sessions: Sequence[AgentSession],
    ) -> tuple[Any, Any]:
        """Left-pad active and completed session histories into one batch."""
        max_length = max(int(session.token_ids.numel()) for session in sessions)
        first = sessions[0].token_ids
        input_ids = first.new_full(
            (len(sessions), max_length),
            self.settings.pad_token_id,
        )
        attention_mask = first.new_zeros((len(sessions), max_length))
        for row, session in enumerate(sessions):
            tokens = session.token_ids
            input_ids[row, -tokens.numel() :] = tokens
            attention_mask[row, -tokens.numel() :] = 1
        return input_ids, attention_mask
    def _response_mask(self, response_ids: Any, explicit_mask: Any) -> Any:
        terminal_mask = response_ids.eq(self.settings.eos_token_ids[0])
        for token_id in self.settings.eos_token_ids[1:]:
            terminal_mask = terminal_mask | response_ids.eq(token_id)
        eos_prefix_mask = terminal_mask.cumsum(dim=-1).eq(0)
        if explicit_mask is None:
            return eos_prefix_mask
        if tuple(explicit_mask.shape) != tuple(response_ids.shape):
            raise ValueError("Generation response_mask must align with response IDs")
        return explicit_mask.bool() & eos_prefix_mask
    def _build_batch(
        self,
        trajectories: tuple[Trajectory, ...],
        generation_seconds: float,
    ) -> ExperienceBatch:
        return build_experience_batch(
            trajectories=trajectories,
            generation_seconds=generation_seconds,
            settings=self.settings,
            metadata={
                "environment": self.environment_name,
                "max_turns": self.max_turns,
            },
        )
    def _synchronize_error(self, local_error: Optional[Exception], operation: str) -> None:
        """Use the generation engine to make local orchestration failures global."""
        synchronize = getattr(self.engine, "synchronize_error", None)
        if synchronize is not None:
            synchronize(local_error, operation)
        elif local_error is not None:
            raise local_error
    @staticmethod
    async def _gather_session_operations(operation: str, *coroutines: Any) -> None:
        """Wait for every session coroutine and report all local failures together."""
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        errors = [str(result) for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"{operation} failed: {errors}")
    async def _run_sessions(
        self,
        sessions: Sequence[AgentSession],
    ) -> tuple[tuple[Trajectory, ...], float]:
        """Keep reset, every step, and close on one environment event loop."""
        generation_seconds = 0.0
        try:
            local_error = None
            try:
                await self._gather_session_operations(
                    "session start",
                    *(session.start() for session in sessions),
                )
            except Exception as error:  # pylint: disable=W0718
                local_error = error
            self._synchronize_error(local_error, "agent session start")
            for _ in range(self.max_turns):
                active_before_generation = [session.active for session in sessions]
                request = None
                local_error = None
                try:
                    input_ids, attention_mask = self._left_pad_sessions(sessions)
                    request = GenerationRequest(input_ids, attention_mask, self.settings)
                except Exception as error:  # pylint: disable=W0718
                    local_error = error
                self._synchronize_error(local_error, "generation request preparation")
                if request is None:
                    raise RuntimeError("Generation request preparation failed without a synchronized error")
                prompt_length = int(request.input_ids.shape[1])
                result = self.engine.generate(request)
                local_error = None
                try:
                    generation_seconds += result.generation_seconds
                    if (
                        result.sequences.ndim != 2
                        or result.sequences.shape[0] != len(sessions)
                        or result.sequences.shape[1] < prompt_length
                    ):
                        raise ValueError(
                            "Generation sequences must have shape "
                            f"[{len(sessions)}, >= {prompt_length}]"
                        )
                    response_ids = result.sequences[:, prompt_length:]
                    response_mask = self._response_mask(
                        response_ids,
                        result.response_mask,
                    )
                    if self.settings.collect_log_probs and result.rollout_log_probs is None:
                        raise ValueError(
                            "Generation engine did not return rollout log-probabilities"
                        )
                    if (
                        result.rollout_log_probs is not None
                        and tuple(result.rollout_log_probs.shape)
                        != tuple(response_ids.shape)
                    ):
                        raise ValueError(
                            "Generation rollout_log_probs must align with response IDs"
                        )
                    for session, was_active in zip(sessions, active_before_generation):
                        if was_active:
                            session.record_worker_policy_identity(
                                result.worker_policy_version,
                                result.worker_policy_fingerprint,
                            )
                    active_sessions: list[AgentSession] = []
                    actions: list[Action] = []
                    for row, (session, was_active) in enumerate(
                        zip(sessions, active_before_generation)
                    ):
                        if not was_active:
                            continue
                        token_mask = response_mask[row]
                        action_tokens = response_ids[row][token_mask]
                        action_text = self.tokenizer.decode(
                            action_tokens.detach().cpu().tolist(),
                            skip_special_tokens=True,
                        )
                        action_log_probs = (
                            None
                            if result.rollout_log_probs is None
                            else result.rollout_log_probs[row][token_mask]
                        )
                        active_sessions.append(session)
                        actions.append(
                            Action(
                                content=action_text,
                                token_ids=action_tokens,
                                rollout_log_probs=action_log_probs,
                            )
                        )
                    await self._gather_session_operations(
                        "session apply",
                        *(
                            session.apply(action)
                            for session, action in zip(active_sessions, actions)
                        ),
                    )
                except Exception as error:  # pylint: disable=W0718
                    local_error = error
                self._synchronize_error(local_error, "generation postprocessing")
            local_error = None
            try:
                for session in sessions:
                    session.finish_max_turns()
            except Exception as error:  # pylint: disable=W0718
                local_error = error
            self._synchronize_error(local_error, "agent session finalization")
        except Exception as error:  # pylint: disable=W0718
            primary_error = error
        else:
            primary_error = None
        finally:
            local_error = None
            try:
                await self._gather_session_operations(
                    "session close",
                    *(session.close() for session in sessions),
                )
            except Exception as error:  # pylint: disable=W0718
                local_error = error
            combined_error = primary_error or local_error
            if primary_error is not None and local_error is not None:
                combined_error = RuntimeError(
                    f"Agent rollout failed: {primary_error}; session close also failed: {local_error}"
                )
            self._synchronize_error(combined_error, "agent rollout and session close")
        trajectories = None
        local_error = None
        try:
            trajectories = tuple(session.build() for session in sessions)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        self._synchronize_error(local_error, "trajectory construction")
        if trajectories is None:
            raise RuntimeError("Trajectory construction failed without a synchronized error")
        return trajectories, generation_seconds
    def rollout(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> ExperienceBatch:
        """Run sampled agent episodes for one published policy version."""
        sessions: list[AgentSession] = []
        local_error = None
        try:
            if not prompt_records:
                raise ValueError("AgentRunner requires at least one PromptRecord")
            engine_version = getattr(self.engine, "policy_version", policy_version)
            if engine_version != policy_version:
                raise RuntimeError(
                    "Rollout requested a stale or unpublished policy snapshot: "
                    f"engine={engine_version}, requested={policy_version}"
                )
            for prompt in prompt_records:
                for sample_index in range(self.num_samples):
                    sessions.append(
                        AgentSession(
                            prompt=prompt,
                            environment=ENVIRONMENTS.build(self.environment_name, prompt),
                            policy_version=policy_version,
                            sample_index=sample_index,
                            max_observation_tokens=self.max_observation_tokens,
                        )
                    )
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        setup_error = None
        try:
            self._synchronize_error(local_error, "rollout setup")
        except Exception as error:  # pylint: disable=W0718
            setup_error = error
        if setup_error is not None:
            cleanup_error = None
            try:
                asyncio.run(
                    self._gather_session_operations(
                        "rollout setup cleanup",
                        *(session.close() for session in sessions),
                    )
                )
            except Exception as error:  # pylint: disable=W0718
                cleanup_error = error
            try:
                self._synchronize_error(cleanup_error, "rollout setup cleanup")
            except Exception as error:  # pylint: disable=W0718
                raise RuntimeError(
                    f"{setup_error}; rollout setup cleanup also failed: {error}"
                ) from setup_error
            raise setup_error
        trajectories, generation_seconds = asyncio.run(self._run_sessions(sessions))
        batch = None
        local_error = None
        try:
            batch = self._build_batch(trajectories, generation_seconds)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        self._synchronize_error(local_error, "experience batch construction")
        if batch is None:
            raise RuntimeError("Experience batch construction failed without a synchronized error")
        return batch
