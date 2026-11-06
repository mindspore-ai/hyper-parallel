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
"""All-rank colocated reporting; stage scheduling stays with the trainer."""
from typing import Any, Callable

from rl.async_runtime.protocol import behavior_segments, decode_group
from rl.utils.monitoring.metrics import build_training_metrics, enforce_learning_gate, summarize_rollout


def summarize_colocate_payloads(
    payloads: tuple[bytes, ...], candidates: int, max_group_bytes: int, step: int,
) -> dict[str, float]:
    """Summarize immutable behavior identities without advancing sampling or training."""
    segments = [segment for payload in payloads for group in [decode_group(payload, candidates, max_group_bytes)]
                for row in group.trajectories for segment in behavior_segments(row, group.task)]
    return {"colocate/sample_age_max": max(step - 1 - part["source_update"] for part in segments),
            "colocate/behavior_versions": len({part["policy_version"] for part in segments})}


def report_colocate_update(roles: Any, rollout: Any, update: Any, diagnostics: Callable[[], dict[str, float]]) -> None:
    """Run reporting collectives and learning gates on every rank before logging."""
    step = roles.state.global_step
    rollout_metrics, samples = summarize_rollout(
        rollout, {"sample_indices": []}, step=step, sample_limit=0,
        is_request_owner=roles.parallel_dims.tp_rank == 0,
    )
    metrics = build_training_metrics(step=step, actor_update=update, rollout_metrics=rollout_metrics,
                                     policy=roles.rollout_engine, diagnostic_metrics=diagnostics())
    enforce_learning_gate(metrics, step=step, config=roles.config["train"].get("learning_gate", {}),
                          run_synchronized=roles.run_synchronized)
    roles.tracker.log(metrics, step=step, samples=samples)
