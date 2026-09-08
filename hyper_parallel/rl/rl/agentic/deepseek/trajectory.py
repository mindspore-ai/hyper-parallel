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
"""Build token-exact Hyper-RL trajectories from DeepSeek Harness model calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel import get_platform
from rl.dataset.contracts import PromptRecord, Trajectory, Turn

platform = get_platform()
_NATURAL_STOPS = frozenset({"stop", "tool_calls", "stop_sequence"})
_MAX_GENERATION_PREFIX_REWRITE = 16


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"DeepSeek completion omitted {label}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"DeepSeek completion {label} must contain integer token IDs"
        ) from error


def _trace(record: Mapping[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    response = record.get("response")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        raise ValueError(
            "DeepSeek completion requires normalized request and response objects"
        )
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise ValueError("DeepSeek completion omitted its first vLLM choice")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    prompt_value = choice.get(
        "input_token_ids",
        choice.get("prompt_token_ids", response.get("prompt_token_ids")),
    )
    response_value = choice.get("token_ids", response.get("token_ids"))
    if response_value is None and isinstance(content, list):
        response_value = [
            item.get("token_id") if isinstance(item, Mapping) else None
            for item in content
        ]
    prompt_ids = _int_list(prompt_value, "prompt token IDs")
    response_ids = _int_list(response_value, "response token IDs")
    if not isinstance(content, list) or len(content) != len(response_ids):
        raise ValueError(
            "DeepSeek completion logprobs must align with response token IDs"
        )
    sampled_logprobs: list[float] = []
    for token_id, item in zip(response_ids, content):
        if not isinstance(item, Mapping) or item.get("logprob") is None:
            raise ValueError(
                "DeepSeek completion contains an incomplete sampled-token logprob"
            )
        if item.get("token_id") is not None and int(item["token_id"]) != token_id:
            raise ValueError("DeepSeek completion token ID and logprob token ID differ")
        sampled_logprobs.append(float(item["logprob"]))
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": sampled_logprobs,
        "finish_reason": choice.get("finish_reason"),
    }


def _eot_id(traces: Sequence[Mapping[str, Any]], configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    for trace in traces:
        response_ids = trace["response_ids"]
        if trace.get("finish_reason") in _NATURAL_STOPS and response_ids:
            return int(response_ids[-1])
    raise ValueError("DeepSeek trajectory cannot determine its end-of-turn token ID")


def _interstitial(
    next_prompt: list[int],
    previous_prompt: list[int],
    previous_response: list[int],
    end_of_turn_token_id: int,
) -> list[int]:
    common_prefix = 0
    for previous_token, next_token in zip(previous_prompt, next_prompt):
        if previous_token != next_token:
            break
        common_prefix += 1
    rewritten_suffix = len(previous_prompt) - common_prefix
    if rewritten_suffix > _MAX_GENERATION_PREFIX_REWRITE:
        raise ValueError(
            "DeepSeek completion history rewrote its canonical prompt body: "
            f"common_prefix={common_prefix}, previous_prompt={len(previous_prompt)}, "
            f"rewritten_suffix={rewritten_suffix}"
        )
    # Qwen's non-thinking chat template appends a short assistant-generation
    # prefill to a live prompt. Once that response becomes history, the template
    # canonicalizes the prefill. Keep the exact sampled response and discard only
    # that bounded, template-owned suffix from the next canonical prompt.
    tail = next_prompt[common_prefix:]
    try:
        boundary = tail.index(end_of_turn_token_id)
    except ValueError as error:
        raise ValueError(
            "DeepSeek completion history omitted the end-of-turn boundary"
        ) from error
    if previous_response and previous_response[-1] == end_of_turn_token_id:
        return tail[boundary + 1 :]
    return tail[boundary:]


def _turns(prompt_length: int, action_mask: list[int], count: int) -> tuple[Turn, ...]:
    turns = [
        Turn(
            role="system",
            content="DeepSeek Harness prompt",
            token_start=0,
            token_end=prompt_length,
            trainable=False,
            metadata={"source": "gateway_prompt", "completion_count": count},
        )
    ]
    start = prompt_length
    while start < len(action_mask):
        trainable = bool(action_mask[start])
        end = start + 1
        while end < len(action_mask) and bool(action_mask[end]) == trainable:
            end += 1
        turns.append(
            Turn(
                role="assistant" if trainable else "tool",
                content="",
                token_start=start,
                token_end=end,
                trainable=trainable,
                metadata={
                    "source": "sampled" if trainable else "deepseek_interstitial"
                },
            )
        )
        start = end
    return tuple(turns)


def build_deepseek_trajectory(
    *,
    prompt: PromptRecord,
    policy_version: int,
    policy_fingerprint: str,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    end_of_turn_token_id: int | None = None,
    max_episode_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Merge captured DeepSeek calls without decoding or re-tokenizing tokens."""
    if not completion_records:
        raise ValueError(
            "DeepSeek trajectory requires at least one captured completion"
        )
    ordered = sorted(
        completion_records, key=lambda record: int(record.get("ordinal", 0))
    )
    traces = [_trace(record) for record in ordered]
    eot_id = _eot_id(traces, end_of_turn_token_id)
    prompt_ids = list(traces[0]["prompt_ids"])
    token_ids = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    token_logprobs: list[float | None] = [None] * len(prompt_ids)
    previous_prompt = list(prompt_ids)
    previous_response: list[int] = []
    for index, trace in enumerate(traces):
        if index:
            inserted = _interstitial(
                list(trace["prompt_ids"]), previous_prompt, previous_response, eot_id
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
    if len(token_ids) < 2 or not any(action_mask):
        raise ValueError("DeepSeek trajectory contains no trainable action tokens")
    if max_episode_tokens is not None and len(token_ids) > max_episode_tokens:
        raise ValueError(
            "DeepSeek trajectory exceeded max_episode_tokens: "
            f"tokens={len(token_ids)}, limit={max_episode_tokens}"
        )
    for index, selected in enumerate(action_mask[1:]):
        if selected and token_logprobs[index + 1] is None:
            raise ValueError("DeepSeek trainable token is missing its sampled logprob")
    prototype = prompt.metadata.get("input_ids")
    if prototype is None or not platform.is_tensor(prototype):
        raise ValueError("DeepSeek trajectory requires PromptRecord metadata input_ids")
    trajectory_metadata = dict(metadata or {})
    trajectory_metadata.update(
        {
            "runner": "deepseek",
            "deepseek_completion_count": len(ordered),
            "tool_history": [
                record.get("original_request", {}).get("messages", [])
                for record in ordered
            ],
            "gateway_records": json.loads(json.dumps(ordered)),
        }
    )
    finish_reasons = [str(trace.get("finish_reason") or "") for trace in traces]
    backend_limited = any(
        reason in {"length", "max_tokens"} for reason in finish_reasons
    )
    harness_reason = str(trajectory_metadata.get("finish_reason") or "")
    harness_failed = harness_reason in {"error", "aborted"}
    truncated = backend_limited or harness_failed
    if backend_limited:
        terminal_reason = "max_tokens"
    elif harness_failed:
        terminal_reason = f"harness_{harness_reason}"
    else:
        terminal_reason = "completed"
    return Trajectory(
        trajectory_id=f"{prompt.prompt_id}:{policy_version}:{sample_index}",
        prompt_id=prompt.prompt_id,
        group_id=prompt.prompt_id,
        policy_version=policy_version,
        turns=_turns(len(prompt_ids), action_mask, len(ordered)),
        token_ids=prototype.new_tensor(token_ids),
        attention_mask=prototype.new_ones((len(token_ids),)),
        action_mask=prototype.new_tensor(action_mask, dtype=platform.tensor_dtype.bool),
        rollout_log_probs=prototype.new_tensor(
            [
                float(value) if value is not None else 0.0
                for value in token_logprobs[1:]
            ],
            dtype=platform.tensor_dtype.float32,
        ),
        reward=float(reward),
        reward_components={
            str(name): float(value) for name, value in reward_components.items()
        },
        done=True,
        truncated=truncated,
        terminal_reason=terminal_reason,
        metadata=trajectory_metadata,
        worker_policy_version=policy_version,
        worker_policy_fingerprint=policy_fingerprint,
    )
