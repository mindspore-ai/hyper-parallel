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
"""Real collective MoonEP parity; CPU doubles only replace unavailable NPU mathematics."""

from contextlib import ExitStack
from itertools import product
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.ep.moonep_grouped_experts import MoonEPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.expert_parallel.moonep_context import MoonEPContext
from tests.common.grouped_experts_family import (
    cpu_grouped_matmul, cpu_swiglu, cpu_token_permute, cpu_token_unpermute,
)


def _case(device: torch.device, mode: str, replay: bool, skew: bool) -> None:
    """Compare ordinary EP and MoonEP on the same home weights, source tokens and gradients."""
    rank, size = dist.get_rank(), dist.get_world_size()
    hidden_size, intermediate, experts = 64, 96, size * 2
    dtype = torch.bfloat16 if device.type == "npu" else torch.float64
    generator = torch.Generator().manual_seed(58)
    gate = torch.randn(experts, 2 * intermediate, hidden_size, generator=generator) * .02
    down = torch.randn(experts, hidden_size, intermediate, generator=generator) * .02
    config = GroupedExpertsConfig(experts, hidden_size, intermediate, num_layers=2, routing_weight_mode=mode)
    ordinary = EPGroupedExperts(config=config).to(device=device, dtype=dtype)
    moon = MoonEPGroupedExperts(config=config).to(device=device, dtype=dtype)
    for member in (ordinary, moon):
        member.gate_up_proj = nn.Parameter(gate[rank * 2:rank * 2 + 2].to(device=device, dtype=dtype))
        member.down_proj = nn.Parameter(down[rank * 2:rank * 2 + 2].to(device=device, dtype=dtype))
    mesh = SimpleNamespace(ndim=1, size=lambda: size, get_local_rank=lambda: rank, get_group=lambda: dist.group.WORLD)
    ordinary.ep_context = EPContext(mesh, num_experts=experts)
    moon.ep_context = MoonEPContext(mesh, num_experts=experts)
    inputs = []
    for microbatch in range(2):
        generator.manual_seed(115 + rank + microbatch * 11)
        hidden = torch.randn(8, hidden_size, generator=generator).to(device=device, dtype=dtype).requires_grad_()
        weights = torch.randn(8, 2, generator=generator).softmax(-1).to(device=device).requires_grad_()
        ids = torch.tensor([[0, 1]] * 8, device=device) if skew else (
            torch.arange(16, device=device).reshape(8, 2) + rank) % experts
        inputs.append((hidden, weights, ids))
    reference_inputs = [(h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_(), ids)
                        for h, w, ids in inputs]
    tolerance = {"rtol": .02, "atol": .002} if device.type == "npu" else {"rtol": 1e-11, "atol": 1e-13}
    for (hidden, weights, ids), (ref_hidden, ref_weights, _) in zip(inputs, reference_inputs):
        actual = checkpoint(moon, hidden, ids, weights, use_reentrant=False) if replay else moon(hidden, ids, weights)
        expected = ordinary(ref_hidden, ids, ref_weights)
        torch.testing.assert_close(actual, expected, **tolerance)
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        torch.testing.assert_close(hidden.grad, ref_hidden.grad, **tolerance)
        torch.testing.assert_close(weights.grad, ref_weights.grad, **tolerance)
    for actual, expected in zip(moon.parameters(), ordinary.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    dist.barrier()


def test_collective_moonep_family() -> None:
    """Exercise both probability modes, skew, balanced loads and two-microbatch replay on real collectives."""
    device_type = os.environ.get("HP_MOONEP_TEST_DEVICE", "npu")
    DefaultDeviceType.set_device_type(device_type)
    device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if device_type == "npu" else torch.device("cpu")
    if device_type == "npu":
        torch.npu.set_device(device)
    dist.init_process_group("hccl" if device_type == "npu" else "gloo")
    try:
        with ExitStack() as stack:
            if device_type == "cpu":
                target = "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts."
                for name, replacement in (("moe_token_permute", cpu_token_permute),
                                          ("moe_token_unpermute", cpu_token_unpermute),
                                          ("grouped_matmul", cpu_grouped_matmul), ("npu_swiglu", cpu_swiglu)):
                    stack.enter_context(patch(target + name, replacement))
            for mode, replay, skew in product(("output", "before_fc2"), (False, True), (False, True)):
                _case(device, mode, replay, skew)
    finally:
        dist.destroy_process_group()
