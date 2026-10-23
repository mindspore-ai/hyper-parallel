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
"""Two-rank CPU integration: production async trainer with a tiny DDP Actor and spawned producer."""
import json
import os
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rl.algorithm.loss import build_algorithm
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.fully_async_trainer import FullyAsyncTrainer
from rl.async_training.protocol import SampleGroup
from rl.async_training.fully_async_rollouter import AsyncRuntime
from rl.roles.rollout.base import GenerationSettings
from rl.dataset.batch_builder import ExperiencePreparer, pad_agent_call_batch_for_dp
from rl.dataset.contracts import Trajectory, Turn
from rl.roles.policy.actor import Actor


class TinyPolicy(torch.nn.Module):
    """A real trainable categorical policy; no model server is implied by this test."""

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(8, 8))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                use_cache: bool = False) -> dict[str, torch.Tensor]:
        """Return real categorical logits for the tiny trainable policy."""
        del attention_mask, use_cache
        return {"logits": self.table[input_ids]}


def producer(config: dict, connection: Any, results: Any, source: dict | None) -> None:
    """Maintain the real CPU producer protocol with deterministic complete trajectories."""
    connection.send(("ready", 1))
    cursor = 0 if source is None else source["cursor"]
    while True:
        command, value = connection.recv()
        if command == "stop":
            break
        if command == "task":
            started = time.monotonic()
            time.sleep(0.03)
            rows = tuple(Trajectory(
                trajectory_id=f"{value.identity}:{index}", prompt_id=str(value.identity),
                group_id=str(value.identity), policy_version=value.policy_version,
                worker_policy_version=value.policy_version,
                turns=(Turn("user", "q", 0, 1, False), Turn("assistant", "a", 1, 3, True)),
                token_ids=torch.tensor([1, 2 + index, 4]), attention_mask=torch.ones(3, dtype=torch.bool),
                action_mask=torch.tensor([False, True, True]),
                rollout_log_probs=torch.full((2,), -math.log(8)),
                reward=float(index), reward_components={"task": float(index)},
                done=True, truncated=False, terminal_reason="completed",
            ) for index in range(2))
            if config["agentic"].get("runner") == "codex":
                calls = []
                for candidate, row in enumerate(rows):
                    count = 1 + (value.identity + candidate) % 3
                    for index in range(count):
                        calls.append(replace(row, trajectory_id=f"{row.trajectory_id}:call:{index}", metadata={
                            "episode_id": row.trajectory_id, "call_index": index, "call_count": count,
                            "async_training": True,
                            "behavior_segments": ({"start": 0, "end": 2, "policy_version": row.policy_version,
                                                   "source_update": row.policy_version - 1},),
                        }))
                rows = tuple(calls)
            payload = SampleGroup(value, rows, 0.03, started, time.monotonic()).encode(
                2, config["async_training"]["max_group_bytes"],
            )
            results.put(payload)
            cursor += 1
        elif command == "pause":
            connection.send(("paused", None))
        elif command == "drain":
            connection.send(("drained", None))
        elif command == "resume":
            connection.send(("resumed", value))
        elif command == "snapshot":
            connection.send(("snapshot", {"cursor": cursor}))
    connection.close()


class Checkpoints:
    """Small persistent model contract, exercising the production async commit layer."""

    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.root = Path(config["output_dir"])

    @property
    def load_path(self) -> str | None:
        """Return the selected checkpoint source."""
        return self.config.get("load_path")

    def directory(self, step: int) -> Path:
        """Locate one committed training step."""
        return self.root / f"step_{step}"

    def validate_resume(self) -> None:
        """Reject incomplete test checkpoints."""
        if self.load_path and not (Path(self.load_path) / "checkpoint_complete.json").exists():
            raise RuntimeError("Incomplete test model checkpoint")

    def begin(self, state: Any) -> None:
        """Restore actual model and optimizer state."""
        if self.load_path:
            data = torch.load(Path(self.load_path) / f"rank{dist.get_rank()}.pt", weights_only=False)
            self.runtime.model.load_state_dict(data["model"])
            self.runtime.actor.optimizer.load_state_dict(data["optimizer"])
            state.global_step = data["step"]

    def will_save(self, step: int) -> bool:
        """Save at the final boundary for this integration test."""
        return step == self.runtime.state.max_steps

    def save(self, state: Any) -> None:
        """Persist each replica before publishing the completion marker."""
        directory = self.directory(state.global_step)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.runtime.model.state_dict(),
                    "optimizer": self.runtime.actor.optimizer.state_dict(),
                    "step": state.global_step}, directory / f"rank{dist.get_rank()}.pt")
        dist.barrier()
        if dist.get_rank() == 0:
            (directory / "checkpoint_complete.json").write_text("{}")
        dist.barrier()

    def finalize(self, state: Any) -> None:
        """Finalize through the same model persistence boundary."""
        self.save(state)


class TinyRuntime:
    """Use actual GRPO, Actor and DDP, replacing only the expensive model/server boundary."""

    def __init__(self, config):
        self.state = SimpleNamespace(global_step=0, max_steps=config["train"]["max_steps"],
                                     consumed_samples=0, consumed_tokens=0)
        self.parallel_dims = SimpleNamespace(dp_rank=dist.get_rank(), dp_size=dist.get_world_size(), tp_rank=0)
        self.device = torch.device("cpu")
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=4)
        self.model = DistributedDataParallel(TinyPolicy())
        self.algorithm = build_algorithm(config["algorithm"])
        group_info = SimpleNamespace(group=dist.group.WORLD, rank_size=dist.get_world_size())
        self.actor = Actor(
            self.model, self.algorithm, micro_batch_size=1,
            optimizer=torch.optim.SGD(self.model.parameters(), lr=0.05),
            device=self.device, dp_group_info=group_info, dp_size=dist.get_world_size(),
            response_mini_batch_size=2, update_epochs=1,
        )
        original_update = self.actor.update

        def delayed_update(batch: Any) -> Any:
            """Keep a measurable overlap window around actual Actor updates."""
            # Leave a deterministic window for producer work while exercising actual Actor/DDP updates.
            time.sleep(0.15)
            return original_update(batch)

        self.actor.update = delayed_update
        self.preparer = ExperiencePreparer(self.algorithm, group_info)
        self.checkpoints = Checkpoints(self, config["train"]["checkpoint"])
        self.evaluator = None
        self.metrics = []
        self.tracker = SimpleNamespace(log=lambda metrics, **kwargs: self.metrics.append(metrics))
        self.rollout_engine = SimpleNamespace(start_service=lambda: None, policy_version=0)
        self.rollout_manager = SimpleNamespace(agent_runner=SimpleNamespace(
            settings=GenerationSettings(2, 1., 1., 0, True, 0, 4, collect_log_probs=True)))
        self.publications = []

    @staticmethod
    def run_synchronized(operation: str, callback: Callable[[], Any]) -> Any:
        """Execute the already coordinated CPU role operation."""
        del operation
        return callback()

    def align_rollout(self, batch: Any) -> Any:
        """Use production DP call padding before model forwards."""
        group_info = SimpleNamespace(group=dist.group.WORLD, rank_size=dist.get_world_size())
        return pad_agent_call_batch_for_dp(batch, group_info)

    def prepare_experience(self, batch: Any) -> Any:
        """Build real episode-level GRPO targets."""
        return self.preparer.prepare(batch)

    def publish_policy(self, version: int) -> None:
        """Coordinate the scripted inference version on every rank."""
        dist.barrier()
        self.rollout_engine.policy_version = version
        self.publications.append(version)

    def close(self) -> None:
        """The parent owns process-group cleanup in this CPU fixture."""


def run(output: str) -> None:
    """Verify actual updates, publication, complete checkpoint and resume on two ranks."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    config = {
        "model": {"name": "qwen3"},
        "algorithm": {"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"},
        "rollout": {"engine": "vllm", "num_return_sequences": 2, "max_new_tokens": 2,
                    "vllm": {"deployment": "disjoint"}},
        "agentic": {"max_turns": 1, "interaction_mode": "single_turn"},
        "train": {"max_steps": 3, "micro_batch_size": 1, "response_mini_batch_size": 2,
                  "accelerator": {"dp_shard": dist.get_world_size()}, "checkpoint": {
                      "load_path": None, "output_dir": output, "save_final": True}},
        "async_training": asdict(AsyncTrainingConfig(
            groups_per_dp_rank=1, max_inflight_groups=4, max_buffer_groups=8,
            max_group_bytes=65536, max_buffer_bytes=524288, publish_every_updates=2,
            operation_timeout=60,
        )),
    }

    if os.environ.get("HYPER_RL_ASYNC_ST_MULTITURN") == "1":
        config["agentic"] = {"runner": "codex", "max_turns": 3, "codex": {}}

    def sampler_factory(*args: Any) -> AsyncRuntime:
        """Keep production IPC while substituting a deterministic producer."""
        return AsyncRuntime(*args, target=producer, control_client=Mock())

    with patch("rl.async_training.fully_async_trainer.RLRoleRuntime", TinyRuntime), patch(
        "rl.async_training.fully_async_trainer.AsyncRuntime", sampler_factory,
    ):
        trainer = FullyAsyncTrainer(config)
        trainer.train()
        first = trainer.roles.model.module.table.detach().clone()
        if not any(item.get("async/rollout_update_overlap_seconds", 0) > 0 for item in trainer.roles.metrics):
            raise RuntimeError("No producer/update overlap was observed")
        if not bool(first.abs().sum() > 0):
            raise RuntimeError("Actor did not update parameters")
        if trainer.roles.publications != [1, 3, 4]:
            raise RuntimeError(f"Unexpected publication schedule: {trainer.roles.publications}")
        config["train"]["checkpoint"]["load_path"] = None
        config["train"]["max_steps"] = 5
        resumed = FullyAsyncTrainer(config)
        resumed.train()
        if resumed.roles.state.global_step != 5:
            raise RuntimeError("Resume did not advance the real trainer")
        if torch.equal(first, resumed.roles.model.module.table):
            raise RuntimeError("Resumed Actor did not update")
        replicas = [torch.empty_like(first) for _ in range(dist.get_world_size())]
        dist.all_gather(replicas, resumed.roles.model.module.table.detach())
        if not all(torch.equal(replicas[0], item) for item in replicas):
            raise RuntimeError("DP replicas diverged")
        for step in range(1, 6):
            path = Path(output).parent / "rollouts" / f"step_{step}_rank_{dist.get_rank()}.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(records) < 2 or not all(record["group_nonzero_advantage"] for record in records):
                raise RuntimeError("Consumed rollout diagnostics lost a DP group or its advantage signal")
    if dist.get_rank() == 0:
        print("ASYNC_GLOO_OK: updates, publication, checkpoint, resume and DP agreement")
    dist.destroy_process_group()


def test_async_worker():
    """Repository torchrun_case target; only workers import the model runtime."""
    run(os.environ["HYPER_RL_ASYNC_ST_OUTPUT"])


if __name__ == "__main__":
    test_async_worker()
