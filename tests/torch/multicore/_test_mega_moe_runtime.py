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
"""Execute large task graphs and expert counts with isolated runtime scratch."""

import os
import struct
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.scheduler.runtime import RUNTIME_HEADER_BYTES
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    memory_sample,
    start_shmem_lifetime,
    write_evidence,
)

FORMER_TASK_LIMIT = 25600


def _runtime_metadata(tensor: torch.Tensor) -> dict:
    """Inspect the actual uploaded image, including the last queued descriptor."""
    image = tensor.cpu().numpy().tobytes()
    tasks, _, capacity, events = struct.unpack_from("<4I", image)
    counts_offset = RUNTIME_HEADER_BYTES + events * 20 + capacity * 576
    counts = struct.unpack_from("<3i", image, counts_offset)
    queues = [
        struct.unpack_from(f"<{count}i", image, counts_offset + 16 + index * capacity * 4)
        for index, count in enumerate(counts)
    ]
    last_id = max(task_id for queue in queues for task_id in queue)
    assert tasks > FORMER_TASK_LIMIT, f"actual task count {tasks} must exceed {FORMER_TASK_LIMIT}."
    assert last_id > FORMER_TASK_LIMIT, f"queue must execute past {FORMER_TASK_LIMIT}, got {last_id}."
    task_types = [struct.unpack_from("<I", image, RUNTIME_HEADER_BYTES + events * 4 + task_id * 576)[0]
                  for task_id in range(FORMER_TASK_LIMIT, last_id)]
    assert any(task_types), "queued descriptors above the old bound must include compute work."
    return {"task_num": tasks, "capacity": capacity, "event_capacity": events,
            "last_queued_task_id": last_id, "runtime_bytes": len(image), "queue_counts": counts}


def test_mega_moe_large_runtime() -> None:
    """Compare complete F+B and SGD with common MoE for an unfused large graph."""
    shape = baseline.MoeShape(local_num_tokens=327680, hidden_size=128, intermediate_size=128)
    start_shmem_lifetime()
    hidden, upstream = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    mega, common = baseline.new_layers(shape)
    try:
        expected = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
        actual = baseline.run_layer(mega, hidden, ids, weights, counts, upstream)
        baseline.assert_results_close(expected, actual)
        # Inspect private plan ownership only to prove which uploaded queues ran.
        plan = mega._resource_group.resources.plan  # pylint: disable=protected-access
        result = {"forward": _runtime_metadata(plan.fwd_runtime_config),
                  "backward": _runtime_metadata(plan.bwd_runtime_config),
                  "all_outputs_gradients_updates_match": True, "memory": memory_sample()}
    finally:
        mega.close()
    write_evidence(result)


def test_mega_moe_group_list_isolation() -> None:
    """Compare repeated routes across dynamic per-worker group-list boundaries."""
    start_shmem_lifetime()
    dispatch_mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    records = []
    patterns = ("balanced", "zero_token_experts", "skew", "single_destination") * 2
    for local_experts in (10, 11, 12, 13, 16, 17, 31, 32, 33, 64, 128):
        shape = baseline.MoeShape(local_num_tokens=128 * local_experts, num_experts=2 * local_experts)
        hidden, upstream = baseline.make_data(shape)
        mega, common = baseline.new_layers(shape, dispatch_mode=dispatch_mode)
        try:
            for pattern in patterns:
                ids, weights, counts = baseline._make_fixed_route(shape, pattern)  # pylint: disable=protected-access
                expected = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
                actual = baseline.run_layer(mega, hidden, ids, weights, counts, upstream)
                baseline.assert_results_close(expected, actual)
            records.append({"shape": asdict(shape), "patterns": patterns,
                            "all_outputs_gradients_updates_match": True})
        finally:
            mega.close()
    write_evidence({"dispatch_mode": dispatch_mode, "cases": records, "memory": memory_sample()})


def _subgroup_route(mode: str, step: int) -> torch.Tensor:
    """Make only the first noncontiguous subgroup overflow on its second step."""
    ids = (torch.arange(512, device=baseline.DEVICE).reshape(256, 2) + step).remainder(4).int()
    if mode == "grow" and step == 1 and baseline.RANK % 2 == 0:
        ids.remainder_(2)
    return ids


def test_mega_moe_subgroups() -> None:
    """Feature: Independent noncontiguous EP groups with externally owned weights.

    Description: Compare both transports against an unsharded oracle with fresh weights each step.
    Expectation: Group-local routing, outputs and input/router/expert gradients agree.
    """
    groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    group = groups[baseline.RANK % 2]
    rank = dist.get_rank(group)
    for mode in ("push", "pull", "grow"):
        layer = MegaMoeExperts(local_num_tokens=128, hidden_size=512, intermediate_size=128,
                               num_experts=4, top_k=2, ep_size=2, ep_group=group,
                               create_parameters=False, dispatch_mode="push" if mode == "grow" else mode,
                               capacity_policy="grow" if mode == "grow" else "static",
                               expert_capacity_factor=1.0 if mode == "grow" else None)
        try:
            for step in range(2):
                torch.manual_seed(123 + step + baseline.RANK % 2)
                tensors = [torch.randn(shape).to(baseline.DEVICE, torch.bfloat16).mul_(0.02).requires_grad_()
                           for shape in ((256, 512), (4, 512, 256), (4, 128, 512))]
                hidden, gate_up, down = tensors
                ids = _subgroup_route(mode, step)
                probs = torch.full((256, 2), 0.5, device=baseline.DEVICE, requires_grad=True)
                permuted, mapping = torch_npu.npu_moe_token_permute(hidden, ids)
                counts = torch.bincount(ids.flatten().long(), minlength=4).cumsum(0)
                activation = torch_npu.npu_swiglu(grouped_matmul(permuted, gate_up, group_list=counts))
                expected = torch_npu.npu_moe_token_unpermute(
                    grouped_matmul(activation, down, group_list=counts), mapping, probs=probs)
                expected.sum().backward()
                token_slice, expert_slice = slice(rank * 128, (rank + 1) * 128), slice(rank * 2, (rank + 1) * 2)
                local = [tensor[index].detach().clone().requires_grad_()
                         for tensor, index in ((hidden, token_slice), (probs, token_slice),
                                               (gate_up, expert_slice), (down, expert_slice))]
                actual = layer(local[0], ids[token_slice], local[1], expert_weights=tuple(local[2:]))
                actual.sum().backward()
                baseline.assert_close("subgroup output", actual, expected[token_slice])
                for observed, reference, index in zip(local, (hidden, probs, gate_up, down),
                                                       (token_slice, token_slice, expert_slice, expert_slice)):
                    baseline.assert_close("subgroup gradient", observed.grad, reference.grad[index])
                assert not list(layer.parameters())
                if mode == "grow":
                    expected_epoch = int(step == 1 and baseline.RANK % 2 == 0)
                    assert layer._resource_group.resources.heap_manager.epoch == expected_epoch
        finally:
            layer.close()
    write_evidence({"noncontiguous_subgroups": True, "external_weights": True, "both_transports": True})
