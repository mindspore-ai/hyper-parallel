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
"""All-rank async reporting; scheduling and learning state remain owned by the trainer."""
import time
from typing import Any

from rl.async_training.protocol import behavior_segments, decode_group
from rl.utils.monitoring.metrics import build_training_metrics, enforce_learning_gate, summarize_rollout
from rl.weight_sync.sync import coordinator_call


def report_async_update(trainer: Any, rollout: Any, update: Any, payloads: tuple[bytes, ...],
                        started: float) -> None:
    """Preserve metric names, learning gates and collective order; call on every training rank."""
    step = trainer.roles.state.global_step
    metrics, samples = summarize_rollout(
        rollout, {"sample_indices": []}, step=trainer.roles.state.global_step, sample_limit=0,
        is_request_owner=trainer.roles.parallel_dims.tp_rank == 0,
    )
    metrics = build_training_metrics(
        step=trainer.roles.state.global_step, actor_update=update, rollout_metrics=metrics,
        policy=trainer.roles.rollout_engine,
        diagnostic_metrics={"async/update_seconds": time.monotonic() - started,
                            "async/policy_lag_updates": trainer.roles.state.global_step - trainer.source_update,
                            "async/optimizer_steps": trainer.optimizer_steps},
    )
    finished = time.monotonic()
    counts = coordinator_call("async buffer metrics", lambda: {
        "async/rollout_update_overlap_seconds": trainer.buffer.overlap_seconds(started, finished),
        "async/ready_groups": len(trainer.buffer.ready),
        "async/inflight_groups": len(trainer.buffer.reserved),
        "async/dropped_groups": trainer.buffer.dropped_groups,
    })
    metrics.update(counts)
    if trainer.rank == 0:
        metrics["reward/zero_std_group_ratio"] = metrics["reward/zero_std_groups"] / len(payloads)
    versions = [segment for group_payload in payloads
                for group in [decode_group(group_payload, trainer.buffer.candidates, trainer.settings.max_group_bytes)]
                for row in group.trajectories for segment in behavior_segments(row, group.task)]
    token_ages = [(step - 1 - segment["source_update"], segment["end"] - segment["start"])
                  for segment in versions]
    metrics.update({
        "async/sample_age_max": max(age for age, _ in token_ages),
        "async/sample_age_token_mean": (sum(age * count for age, count in token_ages)
                                        / sum(count for _, count in token_ages)),
        "async/behavior_versions": len({segment["policy_version"] for segment in versions}),
    })
    enforce_learning_gate(
        metrics, step=trainer.roles.state.global_step,
        config=trainer.config["train"].get("learning_gate", {}),
        run_synchronized=trainer.roles.run_synchronized,
    )
    trainer.roles.tracker.log(metrics, step=trainer.roles.state.global_step, samples=samples)
