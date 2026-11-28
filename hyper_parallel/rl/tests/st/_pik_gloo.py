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
"""CPU/Gloo acceptance of the distributed PIK forward and backward."""
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.parallelism_invariant.row_parallel_linear import PikContext, PikPlan, collectives, linear
from hyper_parallel.components.parallelism_invariant.row_parallel_linear import reduction


def _cpu_tree(partials):
    """Independent adjacent-pair FP32 reference for the hardware-only tree kernel."""
    values = list(partials.float().unbind(0))
    while len(values) > 1:
        values = [left + right for left, right in zip(values[::2], values[1::2])]
    return values[0]


def _check_collectives(group):
    """Check actual Gloo transports against independently constructed rank roots."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    roots = torch.arange(world * 8 * 3, dtype=torch.float32).reshape(world, 8, 3)
    roots[0] += 1e8
    roots[-1] -= 1e8
    expected = _cpu_tree(roots)
    for transport in ("all_gather", "all_to_all"):
        for scatter in (False, True):
            actual = collectives.combine_partials(roots[rank], group, world, rank,
                                                 scatter=scatter, transport=transport)
            wanted = expected.chunk(world)[rank] if scatter else expected
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    gradient = torch.arange(30).reshape(2, 3, 5)[:, ::2] + rank * 100
    gathered = collectives.gather_sequence_gradient(gradient, group, world, 1)
    expected_gradient = torch.cat([torch.arange(30).reshape(2, 3, 5)[:, ::2] + peer * 100
                                   for peer in range(world)], dim=1)
    torch.testing.assert_close(gathered, expected_gradient, rtol=0, atol=0)


def _check_distributed_linear(group):
    """Exercise real BF16 leaf GEMMs and first-order gradients over TP2/TP4."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    full_x = (torch.arange(128).reshape(2, 4, 16) % 11 - 5).bfloat16() / 16
    full_weight = (torch.arange(48).reshape(3, 16) % 7 - 3).bfloat16() / 16
    full_bias = torch.tensor([.125, -.25, .375], dtype=torch.bfloat16)
    upstream = (torch.arange(24).reshape(2, 4, 3) % 5 - 2).bfloat16()
    dense_x, dense_weight, dense_bias = [value.clone().requires_grad_()
                                       for value in (full_x, full_weight, full_bias)]
    torch.nn.functional.linear(dense_x, dense_weight, dense_bias).backward(upstream)
    leaves = torch.stack([torch.nn.functional.linear(part, weight)
                          for part, weight in zip(full_x.chunk(4, -1), full_weight.chunk(4, -1))])
    expected = _cpu_tree(leaves).bfloat16() + full_bias
    for transport in ("all_gather", "all_to_all"):
        for sp in (False, True):
            context = PikContext(PikPlan(4), 16, 3, group, world, rank, 0, sp, 1, transport)
            x, weight, bias = [value.clone().requires_grad_() for value in (
                full_x.chunk(world, -1)[rank], full_weight.chunk(world, -1)[rank], full_bias)]
            # Gradient-enabled inference must retain the shared training backward.
            output = linear.pik_linear_inference(x, weight, bias, context=context)
            wanted = expected.chunk(world, 1)[rank] if sp else expected
            torch.testing.assert_close(output, wanted, rtol=0, atol=0)
            output.backward(upstream.chunk(world, 1)[rank].contiguous() if sp else upstream)
            for actual, reference in ((x.grad, dense_x.grad.chunk(world, -1)[rank]),
                                      (weight.grad, dense_weight.grad.chunk(world, -1)[rank]),
                                      (bias.grad, dense_bias.grad)):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if world == 2:
        for sp in (False, True):
            context = PikContext(PikPlan(2), 16, 3, group, world, rank, 0, sp, 1)
            x, weight = full_x.chunk(world, -1)[rank], full_weight.chunk(world, -1)[rank]
            with torch.no_grad():
                actual = linear.pik_linear_inference(x, weight, full_bias, context=context)
            leaves = torch.stack([torch.nn.functional.linear(part, shard)
                                  for part, shard in zip(full_x.chunk(2, -1), full_weight.chunk(2, -1))])
            expected_inference = _cpu_tree(leaves).bfloat16() + full_bias
            wanted = expected_inference.chunk(world, 1)[rank] if sp else expected_inference
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_pik_gloo() -> None:
    """Check TP4 and both disjoint TP2 subgroups under torchrun."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    try:
        first = dist.new_group([0, 1])
        second = dist.new_group([2, 3])
        # Only the NPU kernel and device guard are replaced. GEMMs, autograd,
        # tensor layouts and every communication operation execute unchanged.
        with patch.object(reduction, "_run_tree", side_effect=_cpu_tree), \
                patch.object(collectives, "fixed_tree", side_effect=_cpu_tree), \
                patch.object(linear, "_validate_input_tensors"):
            for group in (dist.group.WORLD, first if rank < 2 else second):
                _check_collectives(group)
                _check_distributed_linear(group)
    finally:
        dist.destroy_process_group()
