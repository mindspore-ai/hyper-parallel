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
"""Real DDP/GRPO/IPC integration; inference and device residency remain CPU doubles."""
import os
from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from rl.async_runtime.producer import AsyncRuntime
from rl.trainer.colocate_async import ColocateAsyncTrainer, _ColocateRoleRuntime
from hyper_parallel.rl.tests.st._async_train import TinyRuntime, producer


class ColocateTinyRuntime(TinyRuntime):
    """Reuse the existing real Actor fixture, checking only the new residency boundaries."""

    def __init__(self, config: dict) -> None:
        """Build real CPU training roles and residency transition doubles."""
        super().__init__(config)
        self.resolved_config = config
        self.parallel_dims.dp_cp_mesh = SimpleNamespace(get_group=lambda: dist.group.WORLD)
        # Ascend's default optimizer heuristic queries a device even for CPU parameters.
        self.actor.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, foreach=False)
        self.rollout_engine.phase = "rollout"
        self.rollout_engine.prepare_for_training = self.enter_training
        self.rollout_engine.prepare_for_rollout = self.wake

    @property
    def evaluation_needs_gateway_handoff(self) -> bool:
        """Exercise the production ownership check with CPU runtime dependencies."""
        return _ColocateRoleRuntime.evaluation_needs_gateway_handoff.fget(self)

    def evaluate_policy(self, version: int) -> None:
        """Exercise the production evaluator wrapper without model construction."""
        _ColocateRoleRuntime.evaluate_policy(self, version)

    def enter_training(self) -> None:
        """Require a rollout-to-training transition on every rank."""
        if self.rollout_engine.phase != "rollout":
            raise RuntimeError("Invalid CPU sleep transition")
        dist.barrier()
        self.rollout_engine.phase = "training"

    def wake(self) -> None:
        """Restore the same policy after a periodic checkpoint."""
        if self.rollout_engine.phase != "training":
            raise RuntimeError("Invalid CPU wake transition")
        dist.barrier()
        self.rollout_engine.phase = "rollout"

    def release_training_residency(self) -> None:
        """CPU fixture has no NPU allocations to release."""

    def publish_policy(self, version: int) -> None:
        """Verify publication occurs only after training-side sleep."""
        if self.rollout_engine.phase != "training":
            raise RuntimeError("Publishing without sleeping the colocated runtime")
        super().publish_policy(version)
        self.rollout_engine.phase = "rollout"


def test_colocate_worker():
    """Exercise actual optimizer changes, replica agreement and fresh-process-style restore."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    output = os.environ["HYPER_RL_COLOCATE_ST_OUTPUT"]
    config = {
        "model": {"name": "qwen3"}, "data": {},
        "algorithm": {"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"},
        "rollout": {"engine": "vllm", "num_return_sequences": 2, "max_new_tokens": 2,
                    "vllm": {"deployment": "colocated", "data_parallel_size": 1,
                             "tensor_parallel_size": 2, "enforce_eager": True}},
        "agentic": {"max_turns": 1, "interaction_mode": "single_turn"},
        "train": {"max_steps": 3, "micro_batch_size": 1, "response_mini_batch_size": 2,
                  "accelerator": {"dp_shard": 2, "cpu_offload": True},
                  "checkpoint": {"load_path": None, "output_dir": output, "save_final": True}},
        "colocate_async": {"groups_per_dp_rank": 1, "max_group_bytes": 65536,
                           "max_buffer_bytes": 524288, "operation_timeout": 60},
    }
    mode = os.environ.get("HYPER_RL_COLOCATE_ST_CORRECTION", "off")
    if mode != "off":
        config["colocate_async"]["off_policy"] = {"mode": mode}

    def sampler_factory(*args: Any) -> AsyncRuntime:
        """Keep real bounded IPC while replacing the expensive model server."""
        return AsyncRuntime(*args, target=producer, control_client=Mock())

    with patch("rl.trainer.colocate_async._ColocateRoleRuntime", ColocateTinyRuntime), patch(
            "rl.trainer.colocate_async.AsyncGenerationClient.from_config", return_value=Mock()), patch(
            "rl.async_runtime.colocated.AsyncRuntime", sampler_factory):
        trainer = ColocateAsyncTrainer(config)
        trainer.train()
        first_roles = cast(ColocateTinyRuntime, trainer.roles)
        first = first_roles.model.module.table.detach().clone()
        if not bool(first.abs().sum() > 0) or first_roles.publications != [1, 2, 3, 4]:
            raise RuntimeError("Colocated Actor did not update/publish once per step")
        config["train"]["max_steps"] = 5
        resumed = ColocateAsyncTrainer(config)
        resumed.train()
        resumed_roles = cast(ColocateTinyRuntime, resumed.roles)
        final = resumed_roles.model.module.table.detach()
        if resumed_roles.state.global_step != 5 or torch.equal(first, final):
            raise RuntimeError("Colocated checkpoint restore did not continue real updates")
        if resumed_roles.publications != [4, 5, 6]:
            raise RuntimeError("Restored Actor was not republished before sampling")
        replicas = [torch.empty_like(final) for _ in range(dist.get_world_size())]
        dist.all_gather(replicas, final)
        if not all(torch.equal(replicas[0], value) for value in replicas):
            raise RuntimeError("Colocated DP replicas diverged")
        if not any("colocate/ready_groups" in metrics for metrics in resumed_roles.metrics):
            raise RuntimeError("Colocated training lost its metrics")
    if dist.get_rank() == 0:
        print("COLOCATE_GLOO_OK: real updates, IPC, stage transitions, checkpoint and restore")
    dist.destroy_process_group()


if __name__ == "__main__":
    test_colocate_worker()
