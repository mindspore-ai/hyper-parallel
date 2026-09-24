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
"""User-owned agent control flow with the canonical Hyper-RL trajectory output."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable, Optional, Protocol, Sequence

import torch
import torch.distributed as dist

from rl.dataset.batch_builder import build_experience_batch
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory, Turn
from rl.dataset.episodes import episode_rows
from rl.roles.rollout.base import GenerationSettings
from rl.tool_protocol import validate_trainability

_NATURAL_STOPS = frozenset({"stop", "tool_calls", "stop_sequence"})
StatusResolver = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], tuple[bool, str]]


def _int_list(value: Any, label: str, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} completion omitted {field}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} completion {field} must contain integer token IDs") from error


def _sampled_logprobs(content, response_ids, label) -> list[float]:
    """Validate finite sampled probabilities against their captured action token IDs."""
    if not isinstance(content, list) or len(content) != len(response_ids):
        raise ValueError(f"{label} completion logprobs must align with response token IDs")
    sampled_logprobs = []
    for token_id, item in zip(response_ids, content):
        if not isinstance(item, Mapping) or item.get("logprob") is None:
            raise ValueError(f"{label} completion contains an incomplete sampled-token logprob")
        if item.get("token_id") is not None and int(item["token_id"]) != token_id:
            raise ValueError(f"{label} completion token ID and logprob token ID differ")
        value = float(item["logprob"])
        if not math.isfinite(value):
            raise ValueError(f"{label} sampled-token logprob must be finite")
        sampled_logprobs.append(value)
    return sampled_logprobs


def _trace(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate a captured completion and extract its normalized token trace."""
    request = record.get("request")
    response = record.get("response")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        raise ValueError(f"{label} completion requires normalized request and response objects")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError(f"{label} completion omitted its first vLLM choice")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    prompt_value = choice.get(
        "input_token_ids", choice.get("prompt_token_ids", response.get("prompt_token_ids"))
    )
    response_value = choice.get("token_ids", response.get("token_ids"))
    if response_value is None and isinstance(content, list):
        response_value = [item.get("token_id") if isinstance(item, Mapping) else None for item in content]
    prompt_ids = _int_list(prompt_value, label, "prompt token IDs")
    response_ids = _int_list(response_value, label, "response token IDs")
    sampled_logprobs = _sampled_logprobs(content, response_ids, label)
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": sampled_logprobs,
        "finish_reason": choice.get("finish_reason"),
    }


def _end_of_turn_id(
    traces: Sequence[Mapping[str, Any]], configured: int | None, label: str
) -> int:
    if configured is not None:
        return int(configured)
    for trace in traces:
        response_ids = trace["response_ids"]
        if trace.get("finish_reason") in _NATURAL_STOPS and response_ids:
            return int(response_ids[-1])
    raise ValueError(f"{label} trajectory cannot determine its end-of-turn token ID")


def _interstitial(
    next_prompt: list[int],
    previous_prompt: list[int],
    previous_response: list[int],
    end_of_turn_token_id: int,
    label: str,
    max_prefix_rewrite: int,
) -> list[int]:
    """Extract observations only when history preserves the exact sampled action."""
    del end_of_turn_token_id, max_prefix_rewrite
    expected = previous_prompt + previous_response
    if next_prompt[:len(expected)] != expected:
        raise ValueError(f"{label} completion history rewrote its exact sampled-action prefix")
    return next_prompt[len(expected):]


def _trajectory_turns(
    prompt_length: int,
    action_mask: list[int],
    count: int,
    label: str,
    runner_name: str,
) -> tuple[Turn, ...]:
    """Build turn boundaries from contiguous policy-action token spans."""
    turns = [Turn(
        role="system",
        content="DeepSeek Harness prompt" if label == "DeepSeek" else f"{label} harness prompt",
        token_start=0,
        token_end=prompt_length,
        trainable=False,
        metadata={"source": "gateway_prompt", "completion_count": count},
    )]
    start = prompt_length
    while start < len(action_mask):
        trainable = bool(action_mask[start])
        end = start + 1
        while end < len(action_mask) and bool(action_mask[end]) == trainable:
            end += 1
        turns.append(Turn(
            role="assistant" if trainable else "tool",
            content="",
            token_start=start,
            token_end=end,
            trainable=trainable,
            metadata={"source": "sampled" if trainable else f"{runner_name}_interstitial"},
        ))
        start = end
    return tuple(turns)


def _ordered_completions(records: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    """Require complete, unique call ordinals before producing any training row."""
    if not records:
        raise ValueError(f"{label} trajectory requires at least one captured completion")
    ordinals = [record.get("ordinal", 0) for record in records]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ordinals):
        raise ValueError(f"{label} completion ordinals must be integers")
    if sorted(ordinals) != list(range(len(records))):
        raise ValueError(f"{label} completion ordinals must be complete and unique starting at zero")
    return sorted(records, key=lambda record: record.get("ordinal", 0))


def build_harness_trajectory(
    *,
    label: str,
    runner_name: str,
    prompt: PromptRecord,
    policy_version: int,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    end_of_turn_token_id: int | None,
    max_episode_tokens: int | None,
    metadata: Mapping[str, Any] | None,
    max_prefix_rewrite: int = 0,
    tool_history_field: str = "input",
    status_resolver: StatusResolver | None = None,
) -> Trajectory:
    """Merge one external harness trace without decoding or re-tokenizing."""
    if not completion_records:
        raise ValueError(f"{label} trajectory requires at least one captured completion")
    ordered = _ordered_completions(completion_records, label)
    traces = [_trace(record, label) for record in ordered]
    eot_id = _end_of_turn_id(traces, end_of_turn_token_id, label)
    prompt_ids = list(traces[0]["prompt_ids"])
    token_ids = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    token_logprobs: list[float | None] = [None] * len(prompt_ids)
    previous_prompt = list(prompt_ids)
    previous_response: list[int] = []
    for index, trace in enumerate(traces):
        if index:
            inserted = _interstitial(
                list(trace["prompt_ids"]), previous_prompt, previous_response,
                eot_id, label, max_prefix_rewrite,
            )
            token_ids.extend(inserted)
            action_mask.extend([0] * len(inserted))
            token_logprobs.extend([None] * len(inserted))
        response_ids = list(trace["response_ids"])
        token_ids.extend(response_ids)
        action_mask.extend([1] * len(response_ids))
        token_logprobs.extend(trace["response_logprobs"])
        previous_prompt = list(trace["prompt_ids"])
        previous_response = response_ids
    _validate_harness_tokens(label, token_ids, action_mask, token_logprobs, max_episode_tokens)
    prototype = prompt.metadata.get("input_ids")
    if prototype is None or not torch.is_tensor(prototype):
        raise ValueError(f"{label} trajectory requires PromptRecord metadata input_ids")
    trajectory_metadata = dict(metadata or {})
    trajectory_metadata.update({
        "runner": runner_name,
        f"{runner_name}_completion_count": len(ordered),
        "tool_history": [
            record.get("original_request", {}).get(tool_history_field, []) for record in ordered
        ],
        "gateway_records": json.loads(json.dumps(ordered)),
    })
    truncated, terminal_reason = (
        status_resolver(traces, trajectory_metadata) if status_resolver else (False, "completed")
    )
    return Trajectory(
        trajectory_id=f"{prompt.prompt_id}:{policy_version}:{sample_index}",
        prompt_id=prompt.prompt_id,
        group_id=prompt.prompt_id,
        policy_version=policy_version,
        turns=_trajectory_turns(len(prompt_ids), action_mask, len(ordered), label, runner_name),
        token_ids=prototype.new_tensor(token_ids),
        attention_mask=prototype.new_ones((len(token_ids),)),
        action_mask=prototype.new_tensor(action_mask, dtype=torch.bool),
        rollout_log_probs=prototype.new_tensor(
            [float(value) if value is not None else 0.0 for value in token_logprobs[1:]],
            dtype=torch.float32,
        ),
        reward=float(reward),
        reward_components={str(name): float(value) for name, value in reward_components.items()},
        done=True,
        truncated=truncated,
        terminal_reason=terminal_reason,
        metadata=trajectory_metadata,
        worker_policy_version=policy_version,
    )


def _call_status(trace: Mapping[str, Any], episode_status: Optional[tuple[bool, str]]) -> tuple[bool, str]:
    """Prefer a harness-owned episode outcome over the individual call's stop reason."""
    if episode_status is not None:
        return episode_status
    truncated = trace.get("finish_reason") in {"length", "max_tokens"}
    return truncated, "max_tokens" if truncated else "completed"


def build_harness_call_trajectories(
    *,
    label: str,
    runner_name: str,
    prompt: PromptRecord,
    policy_version: int,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    max_episode_tokens: int | None,
    metadata: Mapping[str, Any] | None,
    tool_history_field: str = "input",
    status_resolver: StatusResolver | None = None,
) -> tuple[Trajectory, ...]:
    """Train each sampled action under the exact prompt used by its model call.

    The episode remains the reward/GRPO unit; rows are merely independent
    teacher-forced contexts. No rewritten assistant history is spliced in.
    """
    if not completion_records:
        raise ValueError(f"{label} trajectory requires at least one captured completion")
    prototype = prompt.metadata.get("input_ids")
    if prototype is None or not torch.is_tensor(prototype):
        raise ValueError(f"{label} trajectory requires PromptRecord metadata input_ids")
    ordered = _ordered_completions(completion_records, label)
    traces = [_trace(record, label) for record in ordered]
    episode_id = f"{prompt.prompt_id}:{policy_version}:{sample_index}"
    episode_metadata = dict(metadata or {})
    count = len(traces)
    episode_status = status_resolver(traces, episode_metadata) if status_resolver else None
    results = []
    for index, (record, trace) in enumerate(zip(ordered, traces)):
        prompt_ids = list(trace["prompt_ids"])
        response_ids = list(trace["response_ids"])
        token_ids = prompt_ids + response_ids
        action_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        token_logprobs = [None] * len(prompt_ids) + list(trace["response_logprobs"])
        _validate_harness_tokens(label, token_ids, action_mask, token_logprobs, max_episode_tokens)
        row_metadata = dict(episode_metadata)
        row_metadata.update({
            "runner": runner_name,
            "episode_id": episode_id,
            "call_index": index,
            "call_count": count,
            f"{runner_name}_completion_count": count,
            "finish_reason": trace.get("finish_reason"),
            "tool_history": record.get("original_request", {}).get(tool_history_field, []),
            "gateway_record": json.loads(json.dumps(record)),
        })
        message = record["response"]["choices"][0].get("message", {})
        action_text = message.get("content") if isinstance(message, Mapping) else None
        turns = _trajectory_turns(len(prompt_ids), action_mask, 1, label, runner_name)
        if isinstance(action_text, str):
            turns = tuple(replace(turn, content=action_text) if turn.trainable else turn for turn in turns)
        truncated, terminal_reason = _call_status(trace, episode_status)
        results.append(Trajectory(
            trajectory_id=f"{episode_id}:call:{index}",
            prompt_id=prompt.prompt_id,
            group_id=prompt.prompt_id,
            policy_version=policy_version,
            turns=turns,
            token_ids=prototype.new_tensor(token_ids),
            attention_mask=prototype.new_ones((len(token_ids),)),
            action_mask=prototype.new_tensor(action_mask, dtype=torch.bool),
            rollout_log_probs=prototype.new_tensor(
                [float(value) if value is not None else 0.0 for value in token_logprobs[1:]],
                dtype=torch.float32,
            ),
            reward=float(reward),
            reward_components={str(name): float(value) for name, value in reward_components.items()},
            done=True,
            truncated=truncated,
            terminal_reason=terminal_reason,
            metadata=row_metadata,
            worker_policy_version=policy_version,
        ))
    return tuple(results)


def request_gateway_json(
    label: str,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    """Exchange one JSON object with an external harness gateway."""
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} gateway HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{label} gateway request failed: {error.reason}") from error
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} gateway returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{label} gateway returned a non-object response")
    return decoded


def load_reward_callable(value: Any, config_path: str, label: str) -> Callable[..., Any]:
    """Load a configured external-harness reward callback."""
    if not isinstance(value, str) or ":" not in value:
        raise ValueError(f"{config_path} must use 'module:function' syntax")
    module_name, attribute = value.rsplit(":", 1)
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise ValueError(f"{label} reward callable is not callable: {value}")
    return callback


def harness_generation_settings(
    config: Mapping[str, Any],
    prompt_id: str,
    policy_version: int,
    sample_index: int,
    **extra: Any,
) -> dict[str, Any]:
    """Build deterministic sampling settings shared by external harnesses."""
    top_k = int(config["top_k"])
    settings = {
        "max_tokens": int(config["max_new_tokens"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "top_k": top_k if top_k > 0 else -1,
        **extra,
    }
    seed = config.get("seed")
    if seed is not None:
        identity = f"{prompt_id}:{policy_version}:{sample_index}"
        offset = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
        settings["seed"] = (int(seed) + offset) % (2**31)
    return settings


class HarnessRuntime:
    """Own one rank-zero protocol gateway backed by the shared rollout engine."""

    def __init__(
        self,
        engine: Any,
        config: Mapping[str, Any],
        gateway_factory: Callable[..., Any],
        label: str,
        default_port: int,
        api_prefix: str = "",
        gateway_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a protocol gateway to the shared, policy-versioned rollout engine."""
        self.engine = engine
        self.config = dict(config)
        self._gateway_factory = gateway_factory
        self._label = label
        self._default_port = default_port
        self._gateway_options = dict(gateway_options or {})
        self._gateway: Any | None = None
        self._episode_version: int | None = None
        host = str(self.config.get("gateway_host", "127.0.0.1"))
        public_host = str(self.config.get("gateway_public_host", host))
        port = int(self.config.get("gateway_port", default_port))
        self.admin_url = f"http://{public_host}:{port}"
        self.gateway_url = f"{self.admin_url}{api_prefix}"

    def ensure_started(self) -> None:
        """Materialize vLLM and start the protocol gateway on rank zero."""
        # Client startup contains distributed collectives, so every Trainer rank
        # must finish it before the coordinator enters gateway-only work.
        backend_url = self.engine.inference_base_url
        model_name = self.engine.inference_model_name
        local_error = None
        if dist.get_rank() == 0 and self._gateway is None:
            try:
                self._gateway = self._gateway_factory(
                    host=str(self.config.get("gateway_host", "127.0.0.1")),
                    port=int(self.config.get("gateway_port", self._default_port)),
                    backend_url=backend_url,
                    model_name=model_name,
                    request_timeout=float(self.config.get("request_timeout", 600.0)),
                    **self._gateway_options,
                )
                self._gateway.start()
            except Exception as error:  # pylint: disable=W0718
                local_error = error
        self.engine.synchronize_error(local_error, f"{self._label} gateway startup")
        dist.barrier()

    def close(self) -> None:
        """Stop the gateway on its sole owning rank."""
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    def bind_episode_version(self, version: int) -> None:
        """Publish an all-rank-verified policy version."""
        if version < 0:
            raise ValueError(f"{self._label} episode requires a valid policy version")
        self._episode_version = int(version)

    def clear_episode_version(self) -> None:
        """Clear the policy version after an episode batch."""
        self._episode_version = None

    @property
    def episode_version(self) -> int:
        """Return the collectively established policy version."""
        if self._episode_version is None:
            raise RuntimeError(f"{self._label} episode policy version has not been established")
        return self._episode_version


class HarnessProgramFactory:
    """Create policy-bound external harness programs."""

    program_type: Callable[..., AgentProgram]
    label: str
    include_admin_url = False

    def __init__(
        self,
        runtime: HarnessRuntime,
        end_of_turn_token_id: int | None,
        generation_config: Mapping[str, Any],
    ) -> None:
        """Retain the runtime identity and generation settings for program construction."""
        self.runtime = runtime
        self.end_of_turn_token_id = end_of_turn_token_id
        self.config = {**runtime.config, **dict(generation_config)}

    def __call__(
        self, prompt: PromptRecord, policy_version: int, sample_index: int
    ) -> AgentProgram:
        """Construct one program after verifying the served policy identity."""
        served_version = self.runtime.episode_version
        if served_version != policy_version:
            raise RuntimeError(
                f"{self.label} requested policy version does not match the served policy: "
                f"requested={policy_version}, served={served_version}"
            )
        urls = {"gateway_url": self.runtime.gateway_url}
        if self.include_admin_url:
            urls["admin_url"] = self.runtime.admin_url
        return self.program_type(
            prompt=prompt,
            policy_version=policy_version,
            sample_index=sample_index,
            config=self.config,
            end_of_turn_token_id=self.end_of_turn_token_id,
            **urls,
        )


class AgentProgram(Protocol):
    """One user-defined episode, including its own tools and model calls."""

    async def run(self) -> Trajectory | tuple[Trajectory, ...]:
        """Execute one episode and return a continuous trajectory or its per-call rows."""

AgentProgramFactory = Callable[[PromptRecord, int, int], AgentProgram]


class ProgramAgentRunner:
    """Run user-owned agent programs and enforce only the data-plane contract.

    The factory receives ``(prompt, policy_version, sample_index)``.  User code
    owns the semantic loop and may call an external inference service, tools,
    or a sandbox.  Hyper-RL owns validation, batching, and downstream learning.
    """

    def __init__(
        self,
        program_factory: AgentProgramFactory,
        num_samples: int,
        settings: GenerationSettings,
        engine: Optional[Any] = None,
    ) -> None:
        """Initialize the runner with a program factory and sampling policy."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.program_factory = program_factory
        self.num_samples = num_samples
        self.settings = settings
        self.engine = engine

    async def _run(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> tuple[Trajectory, ...]:
        """Run every sampled user program concurrently on one event loop."""
        programs = [
            self.program_factory(prompt, policy_version, sample_index)
            for prompt in prompt_records
            for sample_index in range(self.num_samples)
        ]
        results = await asyncio.gather(*(program.run() for program in programs), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            details = [f"{type(error).__name__}: {error}" for error in errors]
            raise RuntimeError(f"Agent program group failed after draining all samples: {details}")
        expected_prompts = [prompt.prompt_id for prompt in prompt_records for _ in range(self.num_samples)]
        trajectories = self._program_rows(results, expected_prompts)
        self._validate_trajectories(trajectories, prompt_records, policy_version)
        episode_rows(trajectories)
        validate_trainability(trajectories)
        return tuple(trajectories)

    @staticmethod
    def _program_rows(results, expected_prompts: Sequence[str]) -> tuple[Trajectory, ...]:
        """Flatten program results only after validating each complete episode."""
        trajectories = []
        for result, prompt_id in zip(results, expected_prompts):
            rows = (result,) if isinstance(result, Trajectory) else result
            if not isinstance(rows, tuple) or not rows or not all(isinstance(row, Trajectory) for row in rows):
                raise TypeError("Agent program must return a Trajectory or a nonempty tuple of Trajectories")
            if any(row.prompt_id != prompt_id for row in rows):
                raise ValueError("Agent program returned rows for a different submitted prompt")
            if isinstance(result, tuple):
                if not all(row.metadata.get("episode_id") for row in rows) or len(episode_rows(rows)) != 1:
                    raise ValueError("Agent program call rows must describe exactly one complete episode")
            trajectories.extend(rows)
        return tuple(trajectories)

    def rollout(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> ExperienceBatch:
        """Run all user programs and batch their validated trajectories."""
        if not prompt_records:
            raise ValueError("ProgramAgentRunner requires at least one PromptRecord")
        started = time.perf_counter()
        trajectories = None
        payload = None
        local_error = None
        synchronize_payload = getattr(self.engine, "synchronize_agent_payload", None)
        if bool(getattr(self.engine, "is_request_owner", True)):
            try:
                trajectories = asyncio.run(self._run(prompt_records, policy_version))
                if callable(synchronize_payload):
                    payload = self._serialize_trajectories(trajectories)
            except Exception as error:  # pylint: disable=broad-exception-caught
                local_error = error
        self._synchronize_failure(local_error, "agent program rollout and serialization")
        batch = None
        local_error = None
        try:
            if callable(synchronize_payload):
                payload = synchronize_payload(payload)  # pylint: disable=not-callable
                trajectories = self._deserialize_trajectories(payload, prompt_records)
            if trajectories is None:
                raise RuntimeError("AgentProgram rollout produced no trajectories")
            self._validate_trajectories(trajectories, prompt_records, policy_version)
            episode_rows(trajectories)
            validate_trainability(trajectories)
            batch = build_experience_batch(
                trajectories=trajectories,
                generation_seconds=time.perf_counter() - started,
                settings=self.settings,
                metadata={"runner": "program"},
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            local_error = error
        self._synchronize_failure(local_error, "agent program reconstruction and batching")
        if batch is None:
            raise RuntimeError("AgentProgram batching produced no experience")
        return batch

    def _synchronize_failure(self, error: Optional[Exception], operation: str) -> None:
        """Keep every TP/DP rank on the same error boundary before subsequent collectives."""
        synchronize = getattr(self.engine, "synchronize_error", None)
        if callable(synchronize):
            synchronize(error, operation)  # pylint: disable=not-callable
        elif error is not None:
            raise error

    @staticmethod
    def _validate_trajectories(trajectories, prompt_records, policy_version) -> None:
        """Validate identity before entering any all-rank synchronization."""
        allowed_prompt_ids = {prompt.prompt_id for prompt in prompt_records}
        if len({trajectory.trajectory_id for trajectory in trajectories}) != len(trajectories):
            raise ValueError("Agent programs returned duplicate trajectory IDs")
        for trajectory in trajectories:
            if trajectory.metadata.get("dp_padding", False):
                raise ValueError("Agent programs must return real calls, not DP padding")
            if not math.isfinite(trajectory.reward):
                raise ValueError("Agent program trajectory reward must be finite")
            if trajectory.prompt_id not in allowed_prompt_ids:
                raise ValueError("AgentProgram returned a trajectory for an unknown prompt")
            if trajectory.policy_version != policy_version:
                raise ValueError(
                    "AgentProgram trajectory policy_version does not match the requested snapshot"
                )

    @staticmethod
    def _serialize_trajectories(trajectories: tuple[Trajectory, ...]) -> list[dict[str, Any]]:
        """Convert device tensors to an object-collective-safe payload."""
        return [
            {
                "trajectory_id": item.trajectory_id,
                "prompt_id": item.prompt_id,
                "group_id": item.group_id,
                "policy_version": item.policy_version,
                "turns": [
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "token_start": turn.token_start,
                        "token_end": turn.token_end,
                        "trainable": turn.trainable,
                        "metadata": turn.metadata,
                    }
                    for turn in item.turns
                ],
                "token_ids": item.token_ids.detach().cpu().tolist(),
                "attention_mask": item.attention_mask.detach().cpu().tolist(),
                "action_mask": item.action_mask.detach().cpu().tolist(),
                "rollout_log_probs": (
                    None
                    if item.rollout_log_probs is None
                    else item.rollout_log_probs.detach().cpu().tolist()
                ),
                "reward": item.reward,
                "reward_components": item.reward_components,
                "done": item.done,
                "truncated": item.truncated,
                "terminal_reason": item.terminal_reason,
                "metadata": item.metadata,
                "worker_policy_version": item.worker_policy_version,
            }
            for item in trajectories
        ]

    @staticmethod
    def _deserialize_trajectories(
        payload: Any,
        prompt_records: Sequence[PromptRecord],
    ) -> tuple[Trajectory, ...]:
        """Restore synchronized trajectory payloads on each Trainer TP sibling."""
        if not isinstance(payload, list):
            raise ValueError("Synchronized agent trajectory payload must be a list")
        prompts = {prompt.prompt_id: prompt for prompt in prompt_records}
        trajectories = []
        for item in payload:
            if not isinstance(item, dict) or item.get("prompt_id") not in prompts:
                raise ValueError("Synchronized agent trajectory references an unknown prompt")
            prototype = prompts[item["prompt_id"]].metadata.get("input_ids")
            if prototype is None or not torch.is_tensor(prototype):
                raise ValueError("Synchronized agent trajectory requires prompt input_ids")
            logprobs = item.get("rollout_log_probs")
            trajectories.append(
                Trajectory(
                    trajectory_id=str(item["trajectory_id"]),
                    prompt_id=str(item["prompt_id"]),
                    group_id=item.get("group_id"),
                    policy_version=int(item["policy_version"]),
                    turns=tuple(Turn(**turn) for turn in item["turns"]),
                    token_ids=prototype.new_tensor(item["token_ids"]),
                    attention_mask=prototype.new_tensor(item["attention_mask"]),
                    action_mask=prototype.new_tensor(
                        item["action_mask"],
                        dtype=torch.bool,
                    ),
                    rollout_log_probs=(
                        None
                        if logprobs is None
                        else prototype.new_tensor(logprobs, dtype=torch.float32)
                    ),
                    reward=float(item["reward"]),
                    reward_components=dict(item["reward_components"]),
                    done=bool(item["done"]),
                    truncated=bool(item["truncated"]),
                    terminal_reason=str(item["terminal_reason"]),
                    metadata=dict(item["metadata"]),
                    worker_policy_version=item.get("worker_policy_version"),
                )
            )
        return tuple(trajectories)


def _validate_harness_tokens(label, token_ids, action_mask, token_logprobs, max_episode_tokens):
    """Reject empty, oversized or incomplete sampled-action trajectories."""
    if len(token_ids) < 2 or not any(action_mask):
        raise ValueError(f"{label} trajectory contains no trainable action tokens")
    if max_episode_tokens is not None and len(token_ids) > max_episode_tokens:
        raise ValueError(
            f"{label} trajectory exceeded max_episode_tokens: tokens={len(token_ids)}, "
            f"limit={max_episode_tokens}"
        )
    for index, selected in enumerate(action_mask[1:]):
        if selected and token_logprobs[index + 1] is None:
            raise ValueError(f"{label} trainable token is missing its sampled logprob")
