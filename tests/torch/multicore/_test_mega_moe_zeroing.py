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
"""Verify that MegaMoe never consumes unwritten activation or communication rows."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_moe import function as function_module
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


@contextmanager
def poison_overwritten_buffers(value: float) -> Iterator[None]:
    """Fill candidate outputs before their producers, preserving required dW zeros."""
    allocate_forward = function_module._allocate_forward_intermediates  # pylint: disable=protected-access
    allocate_backward = function_module._allocate_backward_intermediates  # pylint: disable=protected-access
    claim = MegaMoeWorkspace.claim

    def _forward(*args):
        tensors = allocate_forward(*args)
        for tensor in tensors:
            tensor.fill_(value)
        return tensors

    def _backward(*args):
        tensors = allocate_backward(*args)
        for tensor in tensors[2:]:
            tensor.fill_(value)
        return tensors

    def _poisoned_claim(workspace):
        claim(workspace)
        # The existing stream lease must finish before overwriting shared storage.
        workspace.expert_buffer.fill_(value)
        workspace.routed_buffer.fill_(value)

    with (
        patch.object(function_module, "_allocate_forward_intermediates", side_effect=_forward),
        patch.object(function_module, "_allocate_backward_intermediates", side_effect=_backward),
        patch.object(MegaMoeWorkspace, "claim", _poisoned_claim),
    ):
        yield


def _route(shape: baseline.MoeShape, pattern: str) -> tuple:
    """Keep distinct Top-K IDs while changing empty experts and destination ranks."""
    slots = torch.arange(shape.local_num_tokens * shape.top_k).reshape(-1, shape.top_k)
    ids = (slots + baseline.RANK * shape.top_k).remainder(shape.num_experts)
    if pattern == "tail" and baseline.RANK == 0:
        ids[0, 0] = shape.local_experts
    elif pattern == "empty_experts":
        ids = slots.remainder(shape.ep_size) * shape.local_experts
    elif pattern.startswith("destination"):
        ids = slots.remainder(shape.local_experts) + int(pattern[-1]) * shape.local_experts
    counts = torch.bincount(ids.reshape(-1), minlength=shape.num_experts).to(torch.int32)
    weights = torch.full(ids.shape, 1.0 / shape.top_k, dtype=torch.float32, device=baseline.DEVICE)
    return ids.to(device=baseline.DEVICE, dtype=torch.int32), weights, counts.to(baseline.DEVICE)


def _check_empty_gradients(shape: baseline.MoeShape, counts: torch.Tensor, result) -> None:
    """Require exact zero dW for experts skipped by the compute kernels."""
    global_counts = counts.clone()
    dist.all_reduce(global_counts)
    start = baseline.RANK * shape.local_experts
    for expert in range(shape.local_experts):
        if global_counts[start + expert].item() != 0:
            continue
        for gradient in (result.gate_up_weight_grad, result.down_weight_grad):
            assert torch.count_nonzero(gradient[expert]).item() == 0, (
                f"rank={baseline.RANK}: empty expert {expert} has nonzero or NaN dW."
            )


def test_mega_moe_poisoned_buffers() -> None:
    """Compare F+B and updates across poisoned reuse, empty ranks and odd tails."""
    shape = baseline.MoeShape(local_num_tokens=128, num_experts=8)
    source, upstream = baseline.make_data(shape)
    stream = torch.npu.Stream(device=baseline.DEVICE)
    results = []
    for factor in (None, 1.5):
        start_shmem_lifetime()
        mega, common = baseline.new_layers(shape, initial_capacity_factor=factor)
        patterns = ["balanced", "tail", "empty_experts", "balanced"]
        if factor is None:
            patterns.extend(["destination0", "destination1", "tail"])
        try:
            for step, pattern in enumerate(patterns):
                ids, weights, counts = _route(shape, pattern)
                expected = baseline.run_layer(common, source, ids, weights, counts, upstream)
                value = float("nan") if step % 2 == 0 else 1234.0
                with torch.npu.stream(stream), poison_overwritten_buffers(value):
                    actual = baseline.run_layer(mega, source, ids, weights, counts, upstream)
                baseline.assert_results_close(expected, actual)
                _check_empty_gradients(shape, counts, actual)
                results.append({"factor": factor, "pattern": pattern, "poison": str(value), "passed": True})
        finally:
            mega.close()
    write_evidence({"poisoned_buffers": results})
