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
"""Build a strict token-exact HyperParallel-RL trajectory from captured Codex turns."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from hyper_parallel import get_platform
from rl.dataset.contracts import PromptRecord, Trajectory, Turn


platform = get_platform()
_NATURAL_STOPS = frozenset({"stop", "tool_calls", "stop_sequence"})


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Codex completion omitted {label}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Codex completion {label} must contain integer token IDs") from error


def _trace(record: Mapping[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    response = record.get("response")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        raise ValueError("Codex completion requires normalized request and response objects")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("Codex completion omitted its first vLLM choice")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    prompt_ids = choice.get(
        "input_token_ids",
        choice.get("prompt_token_ids", response.get("prompt_token_ids")),
    )
    response_id_value = choice.get("token_ids", response.get("token_ids"))
    if response_id_value is None and isinstance(content, list):
        response_id_value = [
            item.get("token_id") if isinstance(item, Mapping) else None
            for item in content
        ]
    response_ids = _int_list(response_id_value, "response token IDs")
    prompt_ids = _int_list(prompt_ids, "prompt token IDs")
    if not isinstance(content, list) or len(content) != len(response_ids):
        raise ValueError("Codex completion logprobs must align with response token IDs")
    sampled_logprobs: list[float] = []
    for token_id, item in zip(response_ids, content):
        if not isinstance(item, Mapping) or item.get("logprob") is None:
            raise ValueError("Codex completion contains an incomplete sampled-token logprob")
        if item.get("token_id") is not None and int(item["token_id"]) != token_id:
            raise ValueError("Codex completion token ID and logprob token ID differ")
        sampled_logprobs.append(float(item["logprob"]))
    message = choice.get("message")
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": sampled_logprobs,
        "finish_reason": choice.get("finish_reason"),
        "request": dict(request),
        "message": dict(message) if isinstance(message, Mapping) else {},
    }


def _eot_id(traces: Sequence[Mapping[str, Any]], configured: Optional[int]) -> int:
    if configured is not None:
        return int(configured)
    for trace in traces:
        response_ids = trace["response_ids"]
        if trace.get("finish_reason") in _NATURAL_STOPS and response_ids:
            return int(response_ids[-1])
    raise ValueError("Codex trajectory cannot determine its end-of-turn token ID")


def _interstitial(
    next_prompt: list[int],
    previous_prompt: list[int],
    previous_response: list[int],
    end_of_turn_token_id: int,
) -> list[int]:
    if (
        len(next_prompt) < len(previous_prompt)
        or next_prompt[:len(previous_prompt)] != previous_prompt
    ):
        raise ValueError("Codex completion history rewrote its canonical prompt prefix")
    tail = next_prompt[len(previous_prompt):]
    try:
        boundary = tail.index(end_of_turn_token_id)
    except ValueError as error:
        raise ValueError("Codex completion history omitted the end-of-turn boundary") from error
    if previous_response and previous_response[-1] == end_of_turn_token_id:
        return tail[boundary + 1:]
    return tail[boundary:]


def _turns(
    prompt_length: int,
    action_mask: list[int],
    records: Sequence[Mapping[str, Any]],
) -> tuple[Turn, ...]:
    prompt_metadata = {
        "source": "gateway_prompt",
        "completion_count": len(records),
    }
    turns = [
        Turn(
            role="system",
            content="Codex harness prompt",
            token_start=0,
            token_end=prompt_length,
            trainable=False,
            metadata=prompt_metadata,
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
                metadata={"source": "sampled" if trainable else "codex_interstitial"},
            )
        )
        start = end
    return tuple(turns)


def build_codex_trajectory(
    *,
    prompt: PromptRecord,
    policy_version: int,
    policy_fingerprint: str,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    end_of_turn_token_id: Optional[int] = None,
    max_episode_tokens: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Trajectory:
    """Merge every captured completion without decoding or re-tokenizing."""
    if not completion_records:
        raise ValueError("Codex trajectory requires at least one captured completion")
    ordered = sorted(completion_records, key=lambda record: int(record.get("ordinal", 0)))
    traces = [_trace(record) for record in ordered]
    eot_id = _eot_id(traces, end_of_turn_token_id)
    prompt_ids = list(traces[0]["prompt_ids"])
    token_ids = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    token_logprobs: list[Optional[float]] = [None] * len(prompt_ids)
    previous_prompt = list(prompt_ids)
    previous_response: list[int] = []
    for index, trace in enumerate(traces):
        if index:
            inserted = _interstitial(
                list(trace["prompt_ids"]),
                previous_prompt,
                previous_response,
                eot_id,
            )
            token_ids.extend(inserted)
            action_mask.extend([0] * len(inserted))
            token_logprobs.extend([None] * len(inserted))
        response_ids = list(trace["response_ids"])
        response_logprobs = list(trace["response_logprobs"])
        token_ids.extend(response_ids)
        action_mask.extend([1] * len(response_ids))
        token_logprobs.extend(response_logprobs)
        previous_prompt = list(trace["prompt_ids"])
        previous_response = response_ids
    if len(token_ids) < 2 or not any(action_mask):
        raise ValueError("Codex trajectory contains no trainable action tokens")
    if max_episode_tokens is not None and len(token_ids) > max_episode_tokens:
        raise ValueError(
            "Codex trajectory exceeded max_episode_tokens: "
            f"tokens={len(token_ids)}, limit={max_episode_tokens}"
        )
    shifted_logprobs = [
        float(value) if value is not None else 0.0
        for value in token_logprobs[1:]
    ]
    for index, selected in enumerate(action_mask[1:]):
        if selected and token_logprobs[index + 1] is None:
            raise ValueError("Codex trainable token is missing its sampled logprob")
    prototype = prompt.metadata.get("input_ids")
    if prototype is None or not platform.is_tensor(prototype):
        raise ValueError("Codex trajectory requires PromptRecord metadata input_ids")
    ids_tensor = prototype.new_tensor(token_ids)
    attention_tensor = prototype.new_ones((len(token_ids),))
    action_tensor = prototype.new_tensor(action_mask, dtype=platform.tensor_dtype.bool)
    logprob_tensor = prototype.new_tensor(shifted_logprobs, dtype=platform.tensor_dtype.float32)
    trajectory_metadata = dict(metadata or {})
    trajectory_metadata.update(
        {
            "runner": "codex",
            "codex_completion_count": len(ordered),
            "tool_history": [
                record.get("original_request", {}).get("input", [])
                for record in ordered
            ],
            "gateway_records": json.loads(json.dumps(ordered)),
        }
    )
    return Trajectory(
        trajectory_id=f"{prompt.prompt_id}:{policy_version}:{sample_index}",
        prompt_id=prompt.prompt_id,
        group_id=prompt.prompt_id,
        policy_version=policy_version,
        turns=_turns(len(prompt_ids), action_mask, ordered),
        token_ids=ids_tensor,
        attention_mask=attention_tensor,
        action_mask=action_tensor,
        rollout_log_probs=logprob_tensor,
        reward=float(reward),
        reward_components={str(name): float(value) for name, value in reward_components.items()},
        done=True,
        truncated=False,
        terminal_reason="completed",
        metadata=trajectory_metadata,
        worker_policy_version=policy_version,
        worker_policy_fingerprint=policy_fingerprint,
    )
