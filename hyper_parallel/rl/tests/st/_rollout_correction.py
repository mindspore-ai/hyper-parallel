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
"""Real CPU/DDP arithmetic, failure propagation and bounded sampler replenishment."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rl.algorithm import build_algorithm
from rl.algorithm.rollout_correction import correction_metrics, prepare_rollout_correction
from rl.async_runtime.producer import AsyncRuntime
from rl.trainer.colocate_async import ColocateAsyncTrainer
from rl.config.schema import RolloutCorrectionConfig
from rl.data.contracts import ExperienceBatch
from rl.workers.actor import Actor
from hyper_parallel.rl.tests.st._async_train import TinyPolicy, producer
from hyper_parallel.rl.tests.st._colocate_train import ColocateTinyRuntime


def test_correction_ddp() -> None:
    """An empty rank still participates; a bad forward stops both ranks before any step."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    model = DistributedDataParallel(TinyPolicy())
    algorithm = build_algorithm({"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"})
    actor = Actor(model, algorithm, 1, optimizer=torch.optim.SGD(model.parameters(), lr=.05, foreach=False),
                  device=torch.device("cpu"), dp_group_info=SimpleNamespace(group=dist.group.WORLD),
                  dp_size=2, response_mini_batch_size=2, max_grad_norm=0.)
    sequences = torch.tensor([[1, 2, 4], [1, 3, 4]])
    mask = torch.tensor([[False, True, True], [False, True, True]])
    if dist.get_rank() == 1:
        mask.zero_()
    behavior = torch.tensor([[-math.log(8)-math.log(2)]*2, [-math.log(8)+math.log(2)]*2])
    batch = ExperienceBatch((), sequences, torch.ones_like(sequences).bool(), mask, torch.zeros(2),
                            behavior, ("", ""), 0., advantages=torch.tensor([[1., 1.], [-1., -1.]]),
                            rollout_log_probs=behavior)
    config = RolloutCorrectionConfig(mode="decoupled")
    corrected, analysis = prepare_rollout_correction(batch, actor.compute_log_probs(batch), config, 1)
    anchor, weights = corrected.old_log_probs.clone(), corrected.rollout_is_weights.clone()
    metrics = correction_metrics(analysis, corrected.loss_action_mask, config,
                                 group=dist.group.WORLD, world_size=2)
    actor.update(corrected)
    reference = TinyPolicy()
    reference_actor = Actor(reference, algorithm, 1,
                            optimizer=torch.optim.SGD(reference.parameters(), lr=.05, foreach=False),
                            device=torch.device("cpu"), response_mini_batch_size=2, max_grad_norm=0.)
    complete = replace(batch, action_mask=torch.tensor([[False, True, True], [False, True, True]]))
    full, _ = prepare_rollout_correction(complete, reference_actor.compute_log_probs(complete), config, 1)
    reference_actor.update(full)
    torch.testing.assert_close(model.module.table, reference.table, rtol=1.e-6, atol=1.e-7)
    assert metrics["off_policy/action_tokens"] == 4
    torch.testing.assert_close(corrected.old_log_probs, anchor)
    torch.testing.assert_close(corrected.rollout_is_weights, weights)
    before = model.module.table.detach().clone()
    original = model.module.forward

    def bad_forward(*args: Any, **kwargs: Any) -> Any:
        """Inject one rank's invalid model output after collective forward completes."""
        values = original(*args, **kwargs)
        return {**values, "logits": values["logits"] + float("nan")} if dist.get_rank() == 0 else values

    failed = False
    with patch.object(model.module, "forward", side_effect=bad_forward):
        try:
            actor.update(corrected)
        except RuntimeError as error:
            failed = "Non-finite" in str(error)
    flags = torch.tensor(int(failed))
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    assert flags.item() == 1, f"Expected every rank to reject before backward, got={flags.item()}"
    torch.testing.assert_close(before, model.module.table)
    dist.destroy_process_group()


class OutlierQueue:
    """Give only the first global batch a deliberately mismatched behavior distribution."""

    def __init__(self, target: Any) -> None:
        """Wrap the actual multiprocessing result queue."""
        self.target = target

    def put(self, payload: bytes) -> None:
        """Retain complete identities while injecting controlled finite outlier scores."""
        group = pickle.loads(payload)
        if group.task.identity < 2:
            rows = tuple(replace(row, rollout_log_probs=torch.full_like(row.rollout_log_probs, -10.))
                         for row in group.trajectories)
            payload = replace(group, trajectories=rows).encode(2, 65536)
        self.target.put(payload)


def outlier_producer(config: dict, connection: Any, results: Any, source: Any) -> None:
    """Use the production IPC protocol with the controlled test distribution."""
    producer(config, connection, OutlierQueue(results), source)


def test_rejection_refill() -> None:
    """Reject once, preserve the Actor/version, and consume a replenished complete batch."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    output = os.environ["HYPER_RL_COLOCATE_ST_OUTPUT"]
    config = {
        "model": {"name": "qwen3"}, "data": {},
        "algorithm": {"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"},
        "rollout": {"engine": "vllm", "num_return_sequences": 2,
                    "vllm": {"deployment": "colocated", "data_parallel_size": 1,
                             "tensor_parallel_size": 2, "enforce_eager": True}},
        "agentic": {"max_turns": 1},
        "train": {"max_steps": 1, "micro_batch_size": 1, "response_mini_batch_size": 2,
                  "accelerator": {"dp_shard": 2, "cpu_offload": True},
                  "checkpoint": {"load_path": None, "output_dir": output, "save_final": True}},
        "colocate_async": {"groups_per_dp_rank": 1, "max_group_bytes": 65536,
                           "max_buffer_bytes": 524288, "operation_timeout": 60,
                           "off_policy": {"mode": "decoupled", "rejection": "group",
                                          "max_abs_log_ratio": 1., "max_refill_rounds": 1}},
    }
    client = Mock()

    def sampler(*args: Any) -> AsyncRuntime:
        """Keep the real spawned producer and coordinator control path."""
        return AsyncRuntime(*args, target=outlier_producer, control_client=client)

    with patch("rl.trainer.colocate_async._ColocateRoleRuntime", ColocateTinyRuntime), patch(
            "rl.trainer.colocate_async.AsyncGenerationClient.from_config", return_value=Mock()), patch(
            "rl.async_runtime.colocated.AsyncRuntime", sampler):
        trainer = ColocateAsyncTrainer(config)
        trainer.train()
    assert trainer.roles.publications == [1, 2]
    assert trainer.roles.state.global_step == 1 and trainer.optimizer_steps == 1
    if dist.get_rank() == 0:
        assert [call.args[0] for call in client.resume_sampling.call_args_list] == [1, 1, 2]
        state = pickle.loads((Path(output)/"step_1/async_state.pkl").read_bytes())
        assert state["buffer"]["dropped_groups"] == 2
        assert state["buffer"]["consumed_groups"] == 2
        manifest = json.loads((Path(output)/"step_1/async_complete.json").read_text(encoding="utf-8"))
        assert manifest["learner_update"] == 1
    dist.destroy_process_group()
