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
"""Run a task-owned model reward function after colocated rollout has slept."""

import asyncio
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import math
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from rl.agentic.core.types import RewardResult
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory
from rl.reward_model.client import RewardModelClient
from rl.weight_sync.sync import synchronized_call

ModelRewardFunction = Callable[..., Awaitable[RewardResult | float]]


def load_reward_function(path: str) -> ModelRewardFunction:
    """Load one user scorer, leaving task prompts and score semantics in examples."""
    if not isinstance(path, str) or ":" not in path:
        raise ValueError("reward_model.scorer must use module:function syntax")
    module_name, function_name = path.rsplit(":", 1)
    if not module_name or not function_name:
        raise ValueError("reward_model.scorer must use module:function syntax")
    scorer = getattr(importlib.import_module(module_name), function_name)
    if not inspect.iscoroutinefunction(scorer):
        raise ValueError("reward_model.scorer must be an async function")
    signature = inspect.signature(scorer)
    if "interaction_mode" in signature.parameters:
        signature.bind(None, None, None, interaction_mode="single_turn")
    else:
        signature.bind(None, None, None)
    return scorer


def scorer_fingerprint(config: Mapping[str, Any]) -> str:
    """Identify score semantics independently of deployment ports and timeouts."""
    excluded = {
        "port", "server_hccl_if_base_port", "server_hccl_npu_socket_port_range",
        "startup_timeout", "request_timeout", "max_retries", "log_path",
        "max_concurrency", "gpu_memory_utilization", "kv_cache_memory_bytes",
    }
    semantics = {key: value for key, value in config.items() if key not in excluded}
    encoded = json.dumps(semantics, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_pending(prompts: Sequence[PromptRecord], batch: ExperienceBatch, *,
                      require_interaction_mode: bool = False) -> dict[str, PromptRecord]:
    """Reject pre-scored or misaligned trajectories before contacting the RM."""
    if batch.metadata.get("reward_status") != "pending":
        raise ValueError("Model reward input must be pending")
    if require_interaction_mode and batch.metadata.get("interaction_mode") not in ("single_turn", "multi_turn"):
        raise ValueError("Environment reward scorer requires a valid interaction_mode")
    if any(getattr(batch, name) is not None for name in ("advantages", "returns", "values")):
        raise ValueError("Cannot score a batch with training targets")
    if len(batch.trajectories) != len(batch.responses):
        raise ValueError("Model reward needs one trajectory per response")
    if not bool(batch.rewards.eq(0).all().item()):
        raise ValueError("Pending reward tensor must contain only placeholders")
    records = {item.prompt_id: item for item in prompts}
    if len(records) != len(prompts):
        raise ValueError("Model reward prompts must have unique IDs")
    identities = set()
    for trajectory in batch.trajectories:
        if trajectory.trajectory_id in identities or trajectory.prompt_id not in records:
            raise ValueError("Duplicate trajectory or unknown prompt in model reward batch")
        identities.add(trajectory.trajectory_id)
        if trajectory.reward != 0 or any(value != 0 for value in trajectory.reward_components.values()):
            raise ValueError("Pending trajectory has a computed reward")
    return records


async def _score_all(
    records: Mapping[str, PromptRecord], trajectories: Sequence[Trajectory],
    scorer: ModelRewardFunction, client: RewardModelClient, max_concurrency: int, *,
    interaction_mode: Optional[str] = None,
) -> list[RewardResult]:
    """Run bounded scoring requests and close the HTTP session on its event loop."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score(trajectory: Trajectory) -> RewardResult:
        """Evaluate and validate one task-owned model reward."""
        async with semaphore:
            if interaction_mode is None:
                raw = await scorer(records[trajectory.prompt_id], trajectory, client)
            else:
                raw = await scorer(
                    records[trajectory.prompt_id], trajectory, client, interaction_mode=interaction_mode,
                )
            result = raw if isinstance(raw, RewardResult) else RewardResult(float(raw))
            if not math.isfinite(result.value):
                raise ValueError("Model reward must be finite")
            if any(not math.isfinite(value) for value in result.components.values()):
                raise ValueError("Model reward components must be finite")
            json.dumps(dict(result.metadata), allow_nan=False)
            return result

    tasks = [asyncio.create_task(score(trajectory)) for trajectory in trajectories]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close_connection()


def _commit(batch: ExperienceBatch, results: Sequence[RewardResult], scorer_id: str,
            request_count: int, retry_count: int) -> ExperienceBatch:
    """Replace rewards atomically after all results pass tensor-dtype validation."""
    if len(results) != len(batch.trajectories):
        raise ValueError("Model reward results do not align with trajectories")
    rewards = batch.rewards.new_tensor([result.value for result in results])
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("Model rewards must remain finite in training dtype")
    trajectories = tuple(
        replace(trajectory, reward=result.value, reward_components=dict(result.components), metadata={
            **trajectory.metadata,
            "reward_scorer": scorer_id,
            "reward_evidence": dict(result.metadata),
        })
        for trajectory, result in zip(batch.trajectories, results)
    )
    return replace(batch, rewards=rewards, trajectories=trajectories, metadata={
        **batch.metadata,
        "reward_status": "scored",
        "reward_scorer": scorer_id,
        "reward_request_count": request_count,
        "reward_retry_count": retry_count,
    })


def score_model_batch(
    prompts: Sequence[PromptRecord], batch: ExperienceBatch, *,
    scorer: ModelRewardFunction, client: RewardModelClient, scorer_id: str,
    tp_rank: int, tp_size: int, tp_group: Any, max_concurrency: int,
) -> ExperienceBatch:
    """Score each logical TP candidate once and return a new, complete batch."""
    uses_interaction_mode = "interaction_mode" in inspect.signature(scorer).parameters
    records = synchronized_call(
        "model reward input",
        lambda: _validate_pending(prompts, batch, require_interaction_mode=uses_interaction_mode),
    )
    if tp_size > 1:
        signatures = [
            (batch.metadata.get("interaction_mode"), item.trajectory_id, item.prompt_id, item.policy_version,
             tuple((turn.role, turn.content) for turn in item.turns),
             tuple((message.role, message.content) for message in records[item.prompt_id].messages),
             records[item.prompt_id].ground_truth)
            for item in batch.trajectories
        ]
        gathered: list[Any] = [None] * tp_size
        dist.all_gather_object(gathered, signatures, group=tp_group)
        synchronized_call("model reward TP identity", lambda: _check_signatures(gathered))
    before_requests, before_retries = client.request_count, client.retry_count
    results = synchronized_call(
        "model reward computation",
        lambda: asyncio.run(_score_all(
            records, batch.trajectories, scorer, client, max_concurrency,
            interaction_mode=batch.metadata["interaction_mode"] if uses_interaction_mode else None,
        ))
        if tp_rank == 0 else None,
    )
    if tp_size > 1:
        gathered = [None] * tp_size
        dist.all_gather_object(gathered, results, group=tp_group)
        results = gathered[0]
    requests = client.request_count - before_requests
    retries = client.retry_count - before_retries
    return synchronized_call(
        "model reward commit", lambda: _commit(batch, results, scorer_id, requests, retries),
    )


def _check_signatures(signatures: Sequence[Any]) -> None:
    """Reject TP copies that disagree on the trajectories being scored."""
    if any(value != signatures[0] for value in signatures[1:]):
        raise ValueError("TP ranks received different reward trajectories")
