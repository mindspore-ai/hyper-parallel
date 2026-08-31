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
from dataclasses import replace
import hashlib
from typing import Any, Optional, Sequence

from rl.agentic.core.chat_template import TokenizerChatTemplateEncoder
from rl.agentic.core.session import AgentSession
from rl.agentic.core.types import (
    Action,
    EpisodeContext,
    InteractionMode,
    Observation,
    ObservationEncoder,
)
from rl.agentic.envs.environment import ENVIRONMENTS
from rl.dataset.batch_builder import build_experience_batch
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory
from rl.roles.rollout.base import GenerationEngine, GenerationRequest, GenerationSettings


def _canonical_row_seed(
    base_seed: int,
    prompt_id: str,
    sample_index: int,
    samples_per_prompt: int,
) -> int:
    """Derive a stable sampling seed independent of physical DP rank."""
    try:
        prompt_index = int(prompt_id)
    except ValueError:
        digest = hashlib.sha256(prompt_id.encode("utf-8")).digest()
        prompt_index = int.from_bytes(digest[:4], byteorder="big", signed=False)
    if prompt_index < 0:
        raise ValueError(
            "Prompt IDs used for seeded rollout must be non-negative, "
            f"got {prompt_id!r}"
        )
    return base_seed + prompt_index * samples_per_prompt + sample_index


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
        max_episode_tokens: Optional[int] = None,
        environment_settings: Optional[dict[str, Any]] = None,
        interaction_mode: InteractionMode | str | None = None,
    ) -> None:
        """Initialize batched agent rollout settings and dependencies."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if max_observation_tokens is not None and max_observation_tokens < 0:
            raise ValueError("max_observation_tokens must be non-negative")
        if max_episode_tokens is not None and max_episode_tokens <= 0:
            raise ValueError("max_episode_tokens must be positive when configured")
        self.engine = engine
        self.tokenizer = tokenizer
        self.environment_name = environment_name
        self.num_samples = num_samples
        self.max_turns = max_turns
        self.max_observation_tokens = max_observation_tokens
        self.max_episode_tokens = max_episode_tokens
        self.environment_settings = dict(environment_settings or {})
        configured_mode = interaction_mode or self.environment_settings.get("interaction_mode")
        if configured_mode is None:
            configured_mode = (
                InteractionMode.SINGLE_TURN
                if max_turns == 1
                else InteractionMode.MULTI_TURN
            )
        self.interaction_mode = InteractionMode.parse(configured_mode)
        if self.interaction_mode is InteractionMode.SINGLE_TURN and max_turns != 1:
            raise ValueError("single_turn interaction requires max_turns=1")
        self._batch_decode = getattr(tokenizer, "batch_decode", None)
        self.settings = settings

    def _encode_observation(
        self,
        content: str,
        role: str,
        metadata: dict[str, Any],
    ) -> Observation:
        """Tokenize exact incremental environment text without re-rendering history."""
        encoded = self.tokenizer(
            text=content,
            add_special_tokens=False,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"][0]
        observation_metadata = dict(metadata)
        observation_metadata["role"] = role
        return Observation(
            content=content,
            token_ids=token_ids,
            metadata=observation_metadata,
        )

    def _build_observation_encoder(self, prompt: PromptRecord) -> ObservationEncoder:
        """Build one isolated raw or native-template encoder per episode."""
        environment_settings = getattr(self, "environment_settings", {})
        if not bool(environment_settings.get("apply_chat_template", False)):
            return self._encode_observation
        input_ids = prompt.metadata.get("input_ids")
        device = None if input_ids is None else input_ids.device
        return TokenizerChatTemplateEncoder(self.tokenizer, device=device)

    def _left_pad_sessions(
        self,
        sessions: Sequence[AgentSession],
    ) -> tuple[Any, Any]:
        """Left-pad active and completed session histories into one batch."""
        # Completed sessions still occupy a batch row so every distributed rank
        # executes the same generation collectives. Their output is discarded.
        session_tokens = [
            session.token_ids if session.active else session.token_ids[:1]
            for session in sessions
        ]
        max_length = max(int(tokens.numel()) for tokens in session_tokens)
        first = sessions[0].token_ids
        input_ids = first.new_full(
            (len(sessions), max_length),
            self.settings.pad_token_id,
        )
        attention_mask = first.new_zeros((len(sessions), max_length))
        for row, tokens in enumerate(session_tokens):
            input_ids[row, -tokens.numel() :] = tokens
            attention_mask[row, -tokens.numel() :] = 1
        return input_ids, attention_mask

    def _settings_for_turn(
        self,
        sessions: Sequence[AgentSession],
    ) -> GenerationSettings:
        """Clamp one batched generation to every active episode's context budget."""
        for session in sessions:
            if not session.active:
                continue
            remaining = session.remaining_token_budget
            if remaining is None or remaining > 0:
                continue
            if session.turn_count == 0:
                raise ValueError(
                    "Initial observation leaves no token budget for an agent action: "
                    f"prompt_id={session.prompt.prompt_id!r}, "
                    f"max_episode_tokens={session.max_episode_tokens}"
                )
            session.finish_context_limit()
        active_sessions = [session for session in sessions if session.active]
        if not active_sessions:
            return replace(self.settings, max_new_tokens=1)
        finite_budgets = []
        for session in active_sessions:
            remaining = session.remaining_token_budget
            if remaining is not None:
                finite_budgets.append(remaining)
        if not finite_budgets:
            return self.settings
        max_new_tokens = min(self.settings.max_new_tokens, *finite_budgets)
        return replace(self.settings, max_new_tokens=max_new_tokens)

    def _response_mask(self, response_ids: Any, explicit_mask: Any) -> Any:
        """Keep the first EOS trainable and mask every later token."""
        if explicit_mask is not None and tuple(explicit_mask.shape) != tuple(response_ids.shape):
            raise ValueError("Generation response_mask must align with response IDs")
        if self.settings.ignore_eos:
            return response_ids.eq(response_ids) if explicit_mask is None else explicit_mask.bool()
        terminal_mask = response_ids.eq(self.settings.eos_token_ids[0])
        for token_id in self.settings.eos_token_ids[1:]:
            terminal_mask = terminal_mask | response_ids.eq(token_id)
        terminal_count = terminal_mask.to(dtype=response_ids.dtype)
        eos_prefix_mask = (terminal_count.cumsum(dim=-1) - terminal_count).eq(0)
        if explicit_mask is None:
            return eos_prefix_mask
        return explicit_mask.bool() & eos_prefix_mask

    def _prepare_generation_request(
        self,
        sessions: Sequence[AgentSession],
    ) -> tuple[list[bool], GenerationRequest]:
        """Build one synchronized generation request and snapshot active rows."""
        turn_settings = self._settings_for_turn(sessions)
        active_sessions = [session.active for session in sessions]
        input_ids, attention_mask = self._left_pad_sessions(sessions)
        row_seeds = None
        if turn_settings.seed is not None:
            row_seeds = tuple(
                _canonical_row_seed(
                    turn_settings.seed,
                    session.prompt.prompt_id,
                    session.sample_index,
                    self.num_samples,
                )
                for session in sessions
            )
        return active_sessions, GenerationRequest(
            input_ids,
            attention_mask,
            turn_settings,
            row_seeds,
        )

    def _validate_generation_result(
        self,
        result: Any,
        session_count: int,
        prompt_length: int,
    ) -> tuple[Any, Any]:
        """Validate engine output shapes and return response IDs with their mask."""
        if (
            result.sequences.ndim != 2
            or result.sequences.shape[0] != session_count
            or result.sequences.shape[1] < prompt_length
        ):
            raise ValueError(
                "Generation sequences must have shape "
                f"[{session_count}, >= {prompt_length}]"
            )
        response_ids = result.sequences[:, prompt_length:]
        if self.settings.collect_log_probs and result.rollout_log_probs is None:
            raise ValueError("Generation engine did not return rollout log-probabilities")
        if (
            result.rollout_log_probs is not None
            and tuple(result.rollout_log_probs.shape) != tuple(response_ids.shape)
        ):
            raise ValueError("Generation rollout_log_probs must align with response IDs")
        return response_ids, self._response_mask(response_ids, result.response_mask)

    def _decode_actions(
        self,
        sessions: Sequence[AgentSession],
        active_before_generation: Sequence[bool],
        result: Any,
        response_ids: Any,
        response_mask: Any,
    ) -> tuple[list[AgentSession], list[Action]]:
        """Decode engine rows that belonged to active sessions before generation."""
        active_sessions = []
        actions = []
        for session, was_active in zip(sessions, active_before_generation):
            if was_active:
                session.record_worker_policy_identity(
                    result.worker_policy_version,
                    result.worker_policy_fingerprint,
                )
        active_rows = [
            row for row, was_active in enumerate(active_before_generation) if was_active
        ]
        response_values = response_ids[active_rows].detach().cpu().tolist()
        mask_values = response_mask[active_rows].detach().cpu().tolist()
        decoded_ids = [
            [token_id for token_id, keep in zip(row_ids, row_mask) if keep]
            for row_ids, row_mask in zip(response_values, mask_values)
        ]
        if self._batch_decode is None:
            action_texts = [
                self.tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in decoded_ids
            ]
        else:
            action_texts = self._batch_decode(decoded_ids, skip_special_tokens=True)
        for action_index, row in enumerate(active_rows):
            session = sessions[row]
            token_mask = response_mask[row]
            action_tokens = response_ids[row][token_mask]
            action_log_probs = (
                None
                if result.rollout_log_probs is None
                else result.rollout_log_probs[row][token_mask]
            )
            active_sessions.append(session)
            actions.append(
                Action(action_texts[action_index], action_tokens, action_log_probs)
            )
        return active_sessions, actions

    async def _apply_generation_result(
        self,
        sessions: Sequence[AgentSession],
        active_before_generation: Sequence[bool],
        result: Any,
        prompt_length: int,
    ) -> None:
        """Validate, decode, and apply one batched generation result."""
        response_ids, response_mask = self._validate_generation_result(
            result,
            len(sessions),
            prompt_length,
        )
        active_sessions, actions = self._decode_actions(
            sessions,
            active_before_generation,
            result,
            response_ids,
            response_mask,
        )
        await self._gather_session_operations(
            "session apply",
            *(session.apply(action) for session, action in zip(active_sessions, actions)),
        )

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
                "interaction_mode": self.interaction_mode.value,
                "max_turns": self.max_turns,
                "max_episode_tokens": self.max_episode_tokens,
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

    async def _run_session_turns(self, sessions: Sequence[AgentSession]) -> float:
        """Start sessions, execute synchronized turns, and finalize turn limits."""
        local_error = None
        try:
            await self._gather_session_operations(
                "session start",
                *(session.start() for session in sessions),
            )
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        self._synchronize_error(local_error, "agent session start")
        generation_seconds = 0.0
        # A fixed loop count keeps collective ordering identical across ranks.
        for _ in range(self.max_turns):
            active_before_generation = []
            request = None
            local_error = None
            try:
                active_before_generation, request = self._prepare_generation_request(sessions)
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
                await self._apply_generation_result(
                    sessions,
                    active_before_generation,
                    result,
                    prompt_length,
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
        return generation_seconds

    async def _run_sessions(
        self,
        sessions: Sequence[AgentSession],
    ) -> tuple[tuple[Trajectory, ...], float]:
        """Keep reset, every step, and close on one environment event loop."""
        generation_seconds = 0.0
        try:
            generation_seconds = await self._run_session_turns(sessions)
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

    def _populate_sessions(
        self,
        sessions: list[AgentSession],
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> None:
        """Validate a rollout request and append every sampled episode session."""
        if not prompt_records:
            raise ValueError("AgentRunner requires at least one PromptRecord")
        max_turns = getattr(self, "max_turns", 1)
        max_episode_tokens = getattr(self, "max_episode_tokens", None)
        environment_settings = dict(getattr(self, "environment_settings", {}))
        interaction_mode = getattr(self, "interaction_mode", None)
        engine_version = getattr(self.engine, "policy_version", policy_version)
        if engine_version != policy_version:
            raise RuntimeError(
                "Rollout requested a stale or unpublished policy snapshot: "
                f"engine={engine_version}, requested={policy_version}"
            )
        for prompt in prompt_records:
            for sample_index in range(self.num_samples):
                observation_encoder = self._build_observation_encoder(prompt)
                episode_context = EpisodeContext(
                    prompt=prompt,
                    policy_version=policy_version,
                    sample_index=sample_index,
                    max_turns=max_turns,
                    interaction_mode=interaction_mode,
                    settings=environment_settings,
                    observation_encoder=observation_encoder,
                )
                sessions.append(
                    AgentSession(
                        prompt=prompt,
                        environment=ENVIRONMENTS.build(self.environment_name, episode_context),
                        policy_version=policy_version,
                        sample_index=sample_index,
                        max_turns=max_turns,
                        max_observation_tokens=self.max_observation_tokens,
                        max_episode_tokens=max_episode_tokens,
                        episode_context=episode_context,
                    )
                )

    async def _synchronize_setup(
        self,
        sessions: Sequence[AgentSession],
        local_error: Optional[Exception],
    ) -> None:
        """Synchronize setup failure and close every environment created so far."""
        try:
            self._synchronize_error(local_error, "rollout setup")
        except Exception as setup_error:  # pylint: disable=W0718
            cleanup_error = None
            try:
                await self._gather_session_operations(
                    "rollout setup cleanup",
                    *(session.close() for session in sessions),
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

    def _build_synchronized_batch(
        self,
        trajectories: tuple[Trajectory, ...],
        generation_seconds: float,
    ) -> ExperienceBatch:
        """Build an experience batch and synchronize construction failures."""
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

    async def rollout_async(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> ExperienceBatch:
        """Asynchronously run sampled episodes for one policy version."""
        sessions: list[AgentSession] = []
        local_error = None
        try:
            self._populate_sessions(sessions, prompt_records, policy_version)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        await self._synchronize_setup(sessions, local_error)
        trajectories, generation_seconds = await self._run_sessions(sessions)
        return self._build_synchronized_batch(trajectories, generation_seconds)

    def rollout(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> ExperienceBatch:
        """Synchronously run sampled episodes for one policy version."""
        return asyncio.run(self.rollout_async(prompt_records, policy_version))
