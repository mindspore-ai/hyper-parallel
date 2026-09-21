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
"""Device regression for local-capacity buffers with outstanding forwards."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore import profiler as multicore_profiler
from hyper_parallel.core.multicore.torch import ops
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def _route(shape: baseline.MoeShape, pattern: str) -> tuple:
    """Produce exact counts with odd tails or a completely empty destination."""
    positions = torch.arange(shape.local_num_tokens * shape.top_k).reshape(-1, shape.top_k)
    ids = (positions + baseline.RANK * shape.top_k).remainder(shape.num_experts)
    if pattern == "tail" and baseline.RANK == 0:
        ids[0, 0] = shape.local_experts
    elif pattern.startswith("destination"):
        destination = int(pattern[-1])
        ids = positions.remainder(shape.local_experts) + destination * shape.local_experts
    counts = torch.bincount(ids.reshape(-1), minlength=shape.num_experts).to(torch.int32)
    weights = torch.full(ids.shape, 1.0 / shape.top_k, dtype=torch.float32, device=baseline.DEVICE)
    return ids.to(device=baseline.DEVICE, dtype=torch.int32), weights, counts.to(baseline.DEVICE)


def _deferred_pair(layer: torch.nn.Module, shape: baseline.MoeShape, patterns: tuple) -> list:
    """Run two routes before reverse backward without changing the weights."""
    layer.zero_grad(set_to_none=True)
    source, upstream = baseline.make_data(shape)
    hidden = [(source * (1.0 + index / 8.0)).detach().requires_grad_(True) for index in range(2)]
    routes = [_route(shape, pattern) for pattern in patterns]
    route_weights = [weights.requires_grad_(True) for _, weights, _ in routes]
    outputs = [
        baseline.forward_layer(layer, tensor, ids, weights, tokens_per_expert=counts)
        for tensor, (ids, weights, counts) in zip(hidden, routes)
    ]
    for output in reversed(outputs):
        output.backward(upstream, retain_graph=True)
        output.backward(upstream)
    torch.npu.synchronize()
    compared = outputs + [tensor.grad for tensor in hidden + route_weights]
    compared.extend(baseline.expert_weight_gradients(layer))
    for index, tensor in enumerate(compared):
        assert tensor is not None, f"rank={baseline.RANK}: missing memory regression tensor {index}."
        baseline.assert_finite(f"memory regression tensor {index}", tensor)
    return [tensor.detach().cpu() for tensor in compared]


def test_mega_moe_local_capacity_lifetime() -> None:
    """Feature: Transport storage lifetime and conservative receive-buffer reuse.

    Description: Compare retained backward, hotspot/tail routes and profiling with the common backend.
    Expectation: All gradients match with reuse on/off, and pull fits a source-sized heap.
    """
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024)
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    results = []
    for factor in (None, 1.5):
        start_shmem_lifetime()
        mega, common = baseline.new_layers(shape, expert_capacity_factor=factor)
        resources = mega._get_execution_resources(baseline.make_data(shape)[0])  # pylint: disable=protected-access
        assert resources.plan.reuse_backward_dispatch
        original = ops.mega_moe_grad_with_profile_buffer
        pairs = [("balanced", "tail"), ("tail", "balanced")]
        if factor is None:
            pairs.extend([("destination0", "destination1"), ("destination1", "tail")])
        try:
            for reuse in (False, True):
                resources.plan = replace(resources.plan, reuse_backward_dispatch=reuse)

                def _backward(*args):
                    """Check receive-buffer ownership at the native boundary."""
                    assert (args[0].data_ptr() == args[12].data_ptr()) == reuse
                    assert args[17].data_ptr() != args[0].data_ptr()
                    return original(*args)

                with patch.object(ops, "mega_moe_grad_with_profile_buffer", side_effect=_backward):
                    for patterns in pairs:
                        expected = _deferred_pair(common, shape, patterns)
                        with multicore_profiler.mega_kernel_profile(
                            schedule=multicore_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                        ) as profiler:
                            actual = _deferred_pair(mega, shape, patterns)
                            profiler.step()
                        trace = profiler.export_chrome_trace(Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"]).with_name(
                            f"{mode}_rank{baseline.RANK}_trace.json"))
                        assert trace["megaKernelCycleTrace"]["invocationCount"] == 6
                        assert trace["megaKernelCycleTrace"]["droppedRecordCount"] == 0
                        for index, (reference, observed) in enumerate(zip(expected, actual)):
                            baseline.assert_close(f"capacity patterns={patterns} tensor={index}", observed, reference)
                        results.append({"factor": factor, "reuse": reuse, "patterns": patterns, "match": True})
            if mode == "pull":
                data_bytes = shape.local_num_tokens * shape.top_k * shape.hidden_size * 2
                assert 2 * data_bytes < int(os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"]) < 3 * data_bytes
                assert resources.workspace.expert_buffer is None
        finally:
            mega.close()
    write_evidence({"mode": mode, "local_capacity_lifetime": results})
