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
"""CPU contracts for native MoE routing and differentiable MoonEP transport."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional
import torch.distributed as dist
from torch import multiprocessing
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.expert_parallel.moonep import native
from hyper_parallel.core.expert_parallel.moonep.placement import ExpertPlacement
from hyper_parallel.distributed.expert_parallel import experts as execution


class _Experts(nn.Module):
    """Use the real bound native experts body with small stacked projections."""

    def __init__(self, rank: int) -> None:
        """Initialize deterministic expert-major source weights."""
        super().__init__()
        self.num_experts = 4
        self.hidden_size = 3
        self.intermediate_size = 2
        generator = torch.Generator().manual_seed(902)
        gate = torch.randn(4, 4, 3, generator=generator) * 0.2
        down = torch.randn(4, 3, 2, generator=generator) * 0.2
        self.gate_up_proj = nn.Parameter(gate[rank * 2:(rank + 1) * 2].clone())
        self.down_proj = nn.Parameter(down[rank * 2:(rank + 1) * 2].clone())


def _forced_plan(states, ids, counts, capacity, rank):
    """Force borrowing in correctness tests; never use this in performance tests."""
    del states, capacity
    assert counts.dtype == torch.int32
    counts = counts.to(torch.int64)
    virtual = torch.zeros(2, 8, dtype=torch.int64)
    selected = ids.clone()
    for source in range(2):
        rows = int(counts[source, 0])
        virtual[source, 0] = (rows + 1) // 2
        virtual[source, 6] = rows // 2
        virtual[source, 1] = counts[source, 1]
        virtual[source, 4:6] = counts[source, 2:4]
    selected[ids >= 2] += 2
    hot = (ids.reshape(-1) == 0).nonzero().reshape(-1)
    selected.reshape(-1)[hot[1::2]] = 6
    assert selected.shape[0] == int(virtual[rank].sum())
    copies = torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32)
    return ExpertPlacement(selected, virtual, copies, ((-1, -1), (0, -1)))


def _cpu_selection(experts, states, indices, global_experts, group):
    """Reuse production count/split construction without querying NPU hardware."""
    return native.prepare_native_placement(
        states, indices, execution._expert_token_counts(indices, global_experts),
        ep_group=group, hidden_size=experts.hidden_size, intermediate_size=experts.intermediate_size,
        cube_cores=24, weight_bytes=experts.gate_up_proj.numel() * 4 + experts.down_proj.numel() * 4)


def _distributed_checks(rank: int, rendezvous: str) -> None:
    """Run real Gloo exchanges, reverse gradients and independent replay plans."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    group = dist.group.WORLD
    module = nn.Module()
    module.experts = _Experts(rank)
    execution.bind_local_expert_forward(module, 2)
    identities = tuple(id(parameter) for parameter in module.parameters())
    originals = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    references = [_Experts(owner) for owner in range(2)]

    def run(length: int, mode: str, enabled: bool) -> None:
        """Compare output, input, routing and globally owned parameter gradients."""
        module.zero_grad(set_to_none=True)
        for reference in references:
            reference.zero_grad(set_to_none=True)
        inputs = torch.randn(1, length, 3, generator=torch.Generator().manual_seed(71 + rank), requires_grad=True)
        oracle_inputs = inputs.detach().clone().requires_grad_()
        probabilities = torch.full((length, 1), 0.75, requires_grad=True)
        oracle_probabilities = probabilities.detach().clone().requires_grad_()
        ids = torch.zeros(length, 1, dtype=torch.int64)

        def compute(states: torch.Tensor) -> torch.Tensor:
            """Execute one fresh native dispatch, optionally replayed by checkpoint."""
            return execution.ep_routed_forward(
                module, states, router_fn=lambda *_: (ids, probabilities), ep_group=group,
                enable_moonep=enabled)

        if mode == "nonreentrant":
            output = checkpoint(compute, inputs, use_reentrant=False)
        elif mode == "reentrant":
            output = checkpoint(compute, inputs, use_reentrant=True)
        else:
            output = compute(inputs)
        # This rank's source routes to global expert 0 regardless of placement.
        reference = references[0]
        gate, up = functional.linear(oracle_inputs, reference.gate_up_proj[0]).chunk(2, -1)
        oracle = functional.linear(functional.silu(gate) * up, reference.down_proj[0])
        oracle = oracle * oracle_probabilities.reshape(1, length, 1)
        torch.testing.assert_close(output, oracle)
        output.sum().backward()
        oracle.sum().backward()
        torch.testing.assert_close(inputs.grad, oracle_inputs.grad)
        torch.testing.assert_close(probabilities.grad, oracle_probabilities.grad)
        for name, weight in module.experts.named_parameters():
            expected = torch.zeros(4, *weight.shape[1:])
            reference_gradient = getattr(reference, name).grad
            if reference_gradient is not None:
                expected[:2] = reference_gradient
            dist.all_reduce(expected, group=group)
            torch.testing.assert_close(weight.grad, expected[rank * 2:(rank + 1) * 2])
        assert tuple(id(parameter) for parameter in module.parameters()) == identities
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)

    try:
        with patch.object(execution, "_select_moonep_route", side_effect=_cpu_selection), \
             patch.object(native, "placement_is_profitable", return_value=True), \
             patch.object(native, "build_placement", side_effect=_forced_plan):
            for mode in ("plain", "nonreentrant", "reentrant"):
                run(4 if rank == 0 else 0, mode, True)
                run(3 + rank, mode, True)
            run(3 + rank, "plain", False)
    finally:
        dist.destroy_process_group()


class TestNativeMoonEPTransport(unittest.TestCase):
    """Run real distributed CPU transport outside the unit-test tree."""

    def test_real_alltoall_output_gradient_and_checkpoint(self) -> None:
        """
        Feature: Native MoE MoonEP ownership.
        Description: Exercise real variable-split A2A with empty sources and checkpoint replay.
        Expectation: Outputs and all gradients equal the home-expert oracle without replacing Parameters.
        """
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = 'file://' + str(Path(directory) / 'rendezvous')
            multiprocessing.spawn(_distributed_checks, args=(rendezvous,), nprocs=2, join=True)
