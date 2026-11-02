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
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Protocol, Sequence

import torch
import torch.distributed as dist

from rl.agentic.core.prepared_batch import PreparedProgramBatch
from rl.dataset.batch_builder import build_experience_batch
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory, Turn
from rl.dataset.episodes import episode_rows
# Preserve the public imports while host executors use the dependency-free transport module.
from rl.agentic.core.gateway_transport import (  # pylint: disable=unused-import,useless-import-alias
    GatewayTransportError as GatewayTransportError, request_gateway_json as request_gateway_json,
)
from rl.roles.rollout.base import GenerationSettings
from rl.roles.rollout.router_capture import router_trace_from_capture, router_trace_from_wire, router_trace_to_wire
from rl.tool_protocol import validate_trainability

_NATURAL_STOPS = frozenset({"stop", "tool_calls", "stop_sequence"})
_LOGGER = logging.getLogger(__name__)
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
    ordered = _ordered_completions(completion_records, label)
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


def _call_status(trace: Mapping[str, Any], episode_status: Optional[tuple[bool, str]]) -> tuple[bool, str]:
    """Prefer a harness-owned episode outcome over the individual call's stop reason."""
    if episode_status is not None:
        return episode_status
    truncated = trace.get("finish_reason") in {"length", "max_tokens"}
    return truncated, "max_tokens" if truncated else "completed"


def _harness_call_versions(records, policy_version):
    """Validate complete asynchronous evidence before building any call rows."""
    asynchronous = any(record.get("metadata", {}).get("async_training") is True for record in records)
    if asynchronous and not all(record.get("metadata", {}).get("async_training") is True for record in records):
        raise ValueError("An async episode requires version evidence for every model call")
    versions = []
    for record in records:
        call_version = record["metadata"]["policy_version"] if asynchronous else policy_version
        if asynchronous and (isinstance(call_version, bool) or not isinstance(call_version, int)
                             or call_version < max(1, policy_version)):
            raise ValueError("Invalid asynchronous model-call policy version")
        versions.append(call_version)
    return versions


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
    versions = _harness_call_versions(ordered, policy_version)
    for index, (record, trace) in enumerate(zip(ordered, traces)):
        call_version = versions[index]
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
            "gateway_record": json.loads(json.dumps({
                key: value for key, value in record.items() if key not in ("routed_experts", "router_capture_spec")
            })),
        })
        if record.get("metadata", {}).get("async_training") is True:
            row_metadata.update({
                "async_training": True,
                "behavior_segments": ({"start": 0, "end": len(response_ids),
                                       "policy_version": call_version, "source_update": call_version - 1},),
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
            policy_version=call_version,
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
            worker_policy_version=call_version,
            router_trace=router_trace_from_capture(record, len(token_ids)),
        ))
    return tuple(results)


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
        self.admin_token: str | None = None
        host = str(self.config.get("gateway_host", "127.0.0.1"))
        public_host = str(self.config.get("gateway_public_host", host))
        port = int(self.config.get("gateway_port", default_port))
        admin_host = str(self.config.get("gateway_admin_host", public_host))
        self.admin_url = f"http://{admin_host}:{port}"
        self.gateway_url = f"http://{public_host}:{port}{api_prefix}"

    def ensure_started(self) -> None:
        """Materialize vLLM and start the protocol gateway on rank zero."""
        # Client startup contains distributed collectives, so every Trainer rank
        # must finish it before the coordinator enters gateway-only work.
        backend_url = self.engine.inference_base_url
        model_name = self.engine.inference_model_name
        self._ensure_gateway(backend_url, model_name)

    def _ensure_gateway(self, backend_url: str, model_name: str) -> None:
        """Start the CPU gateway collectively, independently of backend readiness."""
        local_error = None
        asynchronous = getattr(self.engine, "asynchronous", False) is True
        owner = asynchronous or dist.get_rank() == 0
        if owner and self._gateway is None:
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
        if asynchronous:
            self.admin_token = self._gateway.admin_token
            return
        if "admin_token" in self._gateway_options:
            credential = [self._gateway.admin_token if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(credential, src=0)
            self.admin_token = credential[0]
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
        engine = getattr(self.runtime, "engine", None)
        served_version = (policy_version if getattr(engine, "asynchronous", False) is True
                          else self.runtime.episode_version)
        if served_version != policy_version:
            raise RuntimeError(
                f"{self.label} requested policy version does not match the served policy: "
                f"requested={policy_version}, served={served_version}"
            )
        return self._build_program(prompt, policy_version, sample_index)

    def _build_program(self, prompt: PromptRecord, policy_version: int, sample_index: int) -> AgentProgram:
        """Construct a session whose actual model admission is owned by the runtime."""
        urls = {"gateway_url": self.runtime.gateway_url}
        if self.include_admin_url:
            urls["admin_url"] = self.runtime.admin_url
            if getattr(self.runtime, "admin_token", None) is not None:
                urls["admin_token"] = self.runtime.admin_token
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
        sample_index_offset: int = 0,

        max_concurrent_programs: int = 8,
    ) -> None:
        """Initialize the runner with a program factory and sampling policy."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.sample_index_offset = sample_index_offset

        if (isinstance(max_concurrent_programs, bool) or not isinstance(max_concurrent_programs, int)
                or max_concurrent_programs <= 0):
            raise ValueError("max_concurrent_programs must be a positive integer")
        self.program_factory = program_factory
        self.num_samples = num_samples
        self.settings = settings
        self.engine = engine
        self.max_concurrent_programs = max_concurrent_programs
        self._prepared: PreparedProgramBatch | None = None

    async def _run(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
        programs: Sequence[AgentProgram] | None = None,
    ) -> tuple[Trajectory, ...]:
        """Bound live programs, stop admitting after failure, and retain submitted row order."""
        jobs = [
            (prompt, self.sample_index_offset + sample_index)
            for prompt in prompt_records
            for sample_index in range(self.num_samples)
        ]
        results = await self._run_bounded(jobs, policy_version, programs)
        expected_prompts = [prompt.prompt_id for prompt in prompt_records for _ in range(self.num_samples)]
        trajectories = self._program_rows(results, expected_prompts)
        self._validate_trajectories(trajectories, prompt_records, policy_version)
        episode_rows(trajectories)
        validate_trainability(trajectories)
        return tuple(trajectories)

    async def _watch_shutdown(self, tasks: list[asyncio.Task]) -> None:
        """Cancel external programs so their workspace/process finally blocks can run."""
        while getattr(self.engine, "closed", False) is not True:
            await asyncio.sleep(.05)
        for task in tasks:
            task.cancel()

    async def _run_bounded(self, jobs: Sequence[tuple[PromptRecord, int]], policy_version: int,
                           programs: Sequence[AgentProgram] | None = None) -> list[Any]:
        """Drain only admitted programs after a failure, before the rank error collective."""
        pending = iter(enumerate(jobs))
        results: list[Any] = [None] * len(jobs)
        failures: list[Exception] = []

        async def worker() -> None:
            """Pull the next job only while the group remains valid."""
            while not failures:
                item = next(pending, None)
                if item is None:
                    return
                index, (prompt, sample_index) = item
                try:
                    program = (programs[index] if programs is not None
                               else self.program_factory(prompt, policy_version, sample_index))
                    results[index] = await program.run()
                except Exception as error:  # pylint: disable=broad-exception-caught
                    failures.append(error)
                    _LOGGER.exception("Agent program failed: prompt=%s sample=%d policy_version=%d; "
                                      "stopping local admission", prompt.prompt_id, sample_index, policy_version)

        tasks = [asyncio.create_task(worker()) for _ in range(min(len(jobs), self.max_concurrent_programs))]
        cancellation = None
        if getattr(self.engine, "asynchronous", False) is True:
            cancellation = asyncio.create_task(self._watch_shutdown(tasks))
        try:
            await asyncio.gather(*tasks)
        except BaseException as error:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if (isinstance(error, asyncio.CancelledError)
                    and getattr(self.engine, "asynchronous", False) is True
                    and getattr(self.engine, "closed", False) is True):
                raise RuntimeError("Agent producer stopped after draining admitted samples") from error
            raise
        finally:
            if cancellation is not None:
                cancellation.cancel()
                await asyncio.gather(cancellation, return_exceptions=True)
        if failures:
            details = [f"{type(error).__name__}: {error}" for error in failures]
            if getattr(self.engine, "asynchronous", False) is True:
                contracts = [error for error in failures if isinstance(error, (ValueError, TypeError))]
                if contracts:
                    message = f"Agent program contract failed after draining admitted samples: {details}"
                    raise ValueError(message) from contracts[0]
            raise RuntimeError(
                f"Agent program group failed after draining admitted samples: {details}"
            ) from failures[0]
        return results

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

    def prepare(self, prompt_records: Sequence[PromptRecord], policy_version: int,
                factory: AgentProgramFactory, *, cpu_fraction: float | None = None,
                max_cpus: int | None = None) -> None:
        """Start only CPU episode work; the caller must keep model admission closed."""
        if self._prepared is not None:
            raise RuntimeError("An agent batch is already prepared")
        if not bool(getattr(self.engine, "is_request_owner", True)):
            return
        programs = [factory(prompt, policy_version, index)
                    for prompt in prompt_records for index in range(self.num_samples)]
        self._prepared = PreparedProgramBatch(
            prompt_records, policy_version, lambda: self._run(prompt_records, policy_version, programs),
            cpu_fraction=cpu_fraction, max_cpus=max_cpus,
        )

    def close(self) -> None:
        """Cancel and join any CPU preparation before closing its gateway."""
        if self._prepared is not None:
            try:
                self._prepared.close()
            finally:
                self._prepared = None

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
                if self._prepared is None:
                    trajectories = asyncio.run(self._run(prompt_records, policy_version))
                else:
                    trajectories = self._prepared.result(prompt_records, policy_version)
                    self._prepared.close()
                    self._prepared = None
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
                mixed_policy_versions=getattr(self.engine, "asynchronous", False) is True,
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

    def _validate_trajectories(self, trajectories, prompt_records, policy_version) -> None:
        """Validate identity before entering any all-rank synchronization."""
        allowed_prompt_ids = {prompt.prompt_id for prompt in prompt_records}
        if len({trajectory.trajectory_id for trajectory in trajectories}) != len(trajectories):
            raise ValueError("Agent programs returned duplicate trajectory IDs")
        for trajectory in trajectories:
            if self.settings.collect_router_trace and trajectory.router_trace is None:
                raise ValueError("R3 agent programs require router traces for every real call")
            if trajectory.metadata.get("dp_padding", False):
                raise ValueError("Agent programs must return real calls, not DP padding")
            if not math.isfinite(trajectory.reward):
                raise ValueError("Agent program trajectory reward must be finite")
            if trajectory.prompt_id not in allowed_prompt_ids:
                raise ValueError("AgentProgram returned a trajectory for an unknown prompt")
            asynchronous = getattr(self.engine, "asynchronous", False) is True
            if asynchronous:
                if trajectory.metadata.get("async_training") is not True or trajectory.policy_version < policy_version:
                    raise ValueError("Async programs require captured per-call policy identities")
            elif trajectory.policy_version != policy_version:
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
                **({"router_trace": router_trace_to_wire(item.router_trace)} if item.router_trace is not None else {}),
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
                    router_trace=(None if item.get("router_trace") is None else router_trace_from_wire(
                        item["router_trace"],
                    )),
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


def _audit_training_row(row) -> tuple[list[int], list[int]]:
    """Verify a training row against the sampled gateway token sequence."""
    response = row.metadata["gateway_record"]["response"]
    choice = response["choices"][0]
    prompt = choice.get("input_token_ids", choice.get("prompt_token_ids", response.get("prompt_token_ids")))
    content = choice["logprobs"]["content"]
    action = choice.get("token_ids", response.get("token_ids"))
    if action is None:
        action = [item["token_id"] for item in content]
    if prompt is None or not action:
        raise ValueError("Missing real prompt/action token IDs")
    prompt, action = list(prompt), list(action)
    if row.token_ids.tolist() != prompt + action:
        raise ValueError("Training context differs from sampled context")
    if row.action_mask.tolist() != [False] * len(prompt) + [True] * len(action):
        raise ValueError("Training action mask differs from sampled actions")
    actual = row.rollout_log_probs[len(prompt) - 1:].tolist()
    expected = [item["logprob"] for item in content]
    if len(actual) != len(expected) or not all(
        math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6) for left, right in zip(actual, expected)
    ):
        raise ValueError("Training rollout logprobs differ from sampled logprobs")
    return prompt, action


def audit_training_rows(trajectories: tuple) -> dict:
    """Fail closed if a training row differs from its actual sampled context/action."""
    prompts, actions = [], []
    for row in trajectories:
        prompt, action = _audit_training_row(row)
        prompts.append(prompt)
        actions.append(action)
    prefix_samples = []
    for index in range(len(prompts) - 1):
        previous = prompts[index] + actions[index]
        current = prompts[index + 1]
        if current[:len(previous)] != previous and len(prefix_samples) < 3:
            mismatch = next((position for position, pair in enumerate(zip(previous, current))
                             if pair[0] != pair[1]), min(len(previous), len(current)))
            prefix_samples.append({"call_index": index + 1, "first_mismatch": mismatch,
                                   "previous_ids": previous[mismatch:mismatch + 8],
                                   "current_ids": current[mismatch:mismatch + 8]})
    return {
        "model_calls": len(trajectories), "training_context_mismatches": 0,
        "raw_prefix_mismatch_samples": prefix_samples,
        "prompt_tokens": [len(item) for item in prompts], "action_tokens": [len(item) for item in actions],
        "training_tokens": sum(len(prompt) + len(action) for prompt, action in zip(prompts, actions)),
        "raw_prefix_mismatches": sum(
            prompts[index + 1][:len(prompts[index]) + len(actions[index])] != prompts[index] + actions[index]
            for index in range(len(prompts) - 1)
        ),
    }
