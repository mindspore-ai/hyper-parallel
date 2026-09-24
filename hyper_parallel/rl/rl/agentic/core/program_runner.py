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
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

import torch
import torch.distributed as dist

from rl.dataset.batch_builder import build_experience_batch
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory, Turn
from rl.roles.rollout.base import GenerationSettings

_NATURAL_STOPS = frozenset({"stop", "tool_calls", "stop_sequence"})
StatusResolver = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], tuple[bool, str]]


def _int_list(value: Any, label: str, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} completion omitted {field}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} completion {field} must contain integer token IDs") from error


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
    if not isinstance(content, list) or len(content) != len(response_ids):
        raise ValueError(f"{label} completion logprobs must align with response token IDs")
    sampled_logprobs = []
    for token_id, item in zip(response_ids, content):
        if not isinstance(item, Mapping) or item.get("logprob") is None:
            raise ValueError(f"{label} completion contains an incomplete sampled-token logprob")
        if item.get("token_id") is not None and int(item["token_id"]) != token_id:
            raise ValueError(f"{label} completion token ID and logprob token ID differ")
        sampled_logprobs.append(float(item["logprob"]))
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
    """Extract observations between completions while validating prefix continuity."""
    common_prefix = 0
    for previous_token, next_token in zip(previous_prompt, next_prompt):
        if previous_token != next_token:
            break
        common_prefix += 1
    if len(previous_prompt) - common_prefix > max_prefix_rewrite:
        detail = "canonical prompt prefix" if max_prefix_rewrite == 0 else "canonical prompt body"
        raise ValueError(f"{label} completion history rewrote its {detail}")
    tail = next_prompt[common_prefix:]
    try:
        boundary = tail.index(end_of_turn_token_id)
    except ValueError as error:
        raise ValueError(f"{label} completion history omitted the end-of-turn boundary") from error
    if previous_response and previous_response[-1] == end_of_turn_token_id:
        return tail[boundary + 1:]
    return tail[boundary:]


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
    fields = _harness_trajectory_fields(
        label, runner_name, prompt, completion_records, end_of_turn_token_id,
        max_episode_tokens, metadata, max_prefix_rewrite, tool_history_field, status_resolver,
    )
    return Trajectory(
        trajectory_id=f"{prompt.prompt_id}:{policy_version}:{sample_index}",
        prompt_id=prompt.prompt_id,
        group_id=prompt.prompt_id,
        policy_version=policy_version,
        **fields,
        reward=float(reward),
        reward_components={str(name): float(value) for name, value in reward_components.items()},
        done=True,
        worker_policy_version=policy_version,
    )


def _merge_harness_traces(
    traces: Sequence[Mapping[str, Any]], eot_id: int | None, label: str, max_prefix_rewrite: int,
) -> tuple[list[int], list[int], list[float | None]]:
    """Merge completion tokens and interstitial observations in trace order."""
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
    return token_ids, action_mask, token_logprobs


@dataclass(frozen=True)
class _PreparedHarnessTrace:
    """Validated token trace and the source records needed for metadata."""
    ordered: list[Mapping[str, Any]]
    traces: list[dict[str, Any]]
    prompt_length: int
    token_ids: list[int]
    action_mask: list[int]
    token_logprobs: list[float | None]
    prototype: Any


def _prepare_harness_trace(
    label: str, prompt: PromptRecord, completion_records: Sequence[Mapping[str, Any]],
    end_of_turn_token_id: int | None, max_episode_tokens: int | None, max_prefix_rewrite: int,
) -> _PreparedHarnessTrace:
    """Validate and merge captured token traces before constructing tensors."""
    if not completion_records:
        raise ValueError(f"{label} trajectory requires at least one captured completion")
    ordered = sorted(completion_records, key=lambda record: int(record.get("ordinal", 0)))
    traces = [_trace(record, label) for record in ordered]
    eot_id = _end_of_turn_id(traces, end_of_turn_token_id, label)
    token_ids, action_mask, token_logprobs = _merge_harness_traces(traces, eot_id, label, max_prefix_rewrite)
    _validate_harness_tokens(label, token_ids, action_mask, token_logprobs, max_episode_tokens)
    prototype = prompt.metadata.get("input_ids")
    if prototype is None or not torch.is_tensor(prototype):
        raise ValueError(f"{label} trajectory requires PromptRecord metadata input_ids")
    return _PreparedHarnessTrace(
        ordered=ordered,
        traces=traces,
        prompt_length=len(traces[0]["prompt_ids"]),
        token_ids=token_ids,
        action_mask=action_mask,
        token_logprobs=token_logprobs,
        prototype=prototype,
    )


def _harness_trajectory_fields(
    label: str, runner_name: str, prompt: PromptRecord,
    completion_records: Sequence[Mapping[str, Any]], end_of_turn_token_id: int | None,
    max_episode_tokens: int | None, metadata: Mapping[str, Any] | None,
    max_prefix_rewrite: int, tool_history_field: str, status_resolver: StatusResolver | None,
) -> dict[str, Any]:
    """Build token tensors and metadata shared by external harness trajectories."""
    trace = _prepare_harness_trace(
        label, prompt, completion_records, end_of_turn_token_id, max_episode_tokens, max_prefix_rewrite
    )
    trajectory_metadata = dict(metadata or {})
    trajectory_metadata.update({
        "runner": runner_name,
        f"{runner_name}_completion_count": len(trace.ordered),
        "tool_history": [
            record.get("original_request", {}).get(tool_history_field, []) for record in trace.ordered
        ],
        "gateway_records": json.loads(json.dumps(trace.ordered)),
    })
    truncated, terminal_reason = (
        status_resolver(trace.traces, trajectory_metadata) if status_resolver else (False, "completed")
    )
    return {
        "turns": _trajectory_turns(
            trace.prompt_length, trace.action_mask, len(trace.ordered), label, runner_name
        ),
        "token_ids": trace.prototype.new_tensor(trace.token_ids),
        "attention_mask": trace.prototype.new_ones((len(trace.token_ids),)),
        "action_mask": trace.prototype.new_tensor(trace.action_mask, dtype=torch.bool),
        "rollout_log_probs": trace.prototype.new_tensor(
            [float(value) if value is not None else 0.0 for value in trace.token_logprobs[1:]],
            dtype=torch.float32,
        ),
        "truncated": truncated,
        "terminal_reason": terminal_reason,
        "metadata": trajectory_metadata,
    }


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
    ) -> None:
        self.engine = engine
        self.config = dict(config)
        self._gateway_factory = gateway_factory
        self._label = label
        self._default_port = default_port
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

    async def run(self) -> Trajectory:
        """Execute one user-owned episode and return its trajectory."""

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
        return tuple(await asyncio.gather(*(program.run() for program in programs)))

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
        local_error = None
        is_owner = bool(getattr(self.engine, "is_request_owner", True))
        if is_owner:
            try:
                trajectories = asyncio.run(self._run(prompt_records, policy_version))
            except Exception as error:  # pylint: disable=W0718
                local_error = error
        synchronize_error = getattr(self.engine, "synchronize_error", None)
        if callable(synchronize_error):
            # Optional engine hooks are installed dynamically; the callable guard validates them.
            synchronize_error(local_error, "agent program rollout")  # pylint: disable=not-callable
        elif local_error is not None:
            raise local_error
        synchronize_payload = getattr(self.engine, "synchronize_agent_payload", None)
        if callable(synchronize_payload):
            payload = None if trajectories is None else self._serialize_trajectories(trajectories)
            # Pylint infers the getattr default rather than the dynamically installed callback.
            payload = synchronize_payload(payload)  # pylint: disable=not-callable
            trajectories = self._deserialize_trajectories(payload, prompt_records)
        if trajectories is None:
            raise RuntimeError("AgentProgram rollout produced no trajectories")
        allowed_prompt_ids = {prompt.prompt_id for prompt in prompt_records}
        for trajectory in trajectories:
            if trajectory.prompt_id not in allowed_prompt_ids:
                raise ValueError("AgentProgram returned a trajectory for an unknown prompt")
            if trajectory.policy_version != policy_version:
                raise ValueError(
                    "AgentProgram trajectory policy_version does not match the requested snapshot"
                )
        return build_experience_batch(
            trajectories=trajectories,
            generation_seconds=time.perf_counter() - started,
            settings=self.settings,
            metadata={"runner": "program"},
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
