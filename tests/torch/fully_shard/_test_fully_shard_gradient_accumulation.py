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
"""Real Gloo regression for nested FSDP gradients across no-sync microbatches."""
import copy

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import HSDPModule, _extend_module_with_hsdp_interface
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from tests.torch.utils import init_dist_gloo


class _Block(nn.Module):
    """Small linear block with deterministic, exactly representable weights."""

    def __init__(self, offset: int) -> None:
        """Create deterministic weights with a distinct offset per unit."""
        super().__init__()
        self.weight = nn.Parameter((torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset) / 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic linear weight."""
        return inputs @ self.weight


class _ConditionalModel(nn.Module):
    """Keep one nested unit optional while the root participates in both microbatches."""

    def __init__(self) -> None:
        """Construct one always-active unit and one conditional unit."""
        super().__init__()
        self.common = _Block(1)
        self.early = _Block(3)

    def forward(self, inputs: torch.Tensor, use_early: bool) -> torch.Tensor:
        """Include the optional unit only in the selected microbatches."""
        result = self.common(inputs)
        return result + self.early(inputs) if use_early else result


def _inputs(rank: int, micro: int) -> torch.Tensor:
    return torch.tensor([[rank + 1.0, micro + 1.0, rank + micro + 1.0, 2.0]],
                        dtype=torch.float32, requires_grad=True)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return (to_local() if callable(to_local) else tensor).detach()


def _wrap_cpu_units(model: nn.Module, mesh, comm_fusion: bool) -> None:
    # Match the CPU scheduler seam in _test_fully_shard_hook_state_machine.py.
    # Public fully_shard rejects CPU; no production device check is overridden.
    policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
        apply_grad_on_fp32_main_grad=True,
    )
    for unit in (model.common, model.early, model):
        _extend_module_with_hsdp_interface(unit)
        unit.hsdp_scheduler = HSDPSchedulerV2(
            unit, mesh, True, None, policy, OffloadPolicy(), set(), set(), torch.device("cpu"), comm_fusion,
        )
    model.set_reduce_op_type("sum")


def _run_case(mesh, rank: int, comm_fusion: bool, branch_both: bool, sync_each: bool) -> None:
    reference = _ConditionalModel()
    model = copy.deepcopy(reference)
    _wrap_cpu_units(model, mesh, comm_fusion)
    reference_parameters = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        expected = reference_parameters[name].detach().chunk(2, dim=0)[rank]
        torch.testing.assert_close(_local(param), expected, rtol=0, atol=0)

    # Ordinary PyTorch graph, independent of HP's scheduler and communication.
    # SUM is explicit so the CPU test does not depend on Gloo AVG support.
    for source_rank in range(2):
        for micro in range(2):
            reference(_inputs(source_rank, micro), branch_both or micro == 0).sum().backward()

    with SkipDTensorDispatch():
        for micro in range(2):
            model.set_requires_gradient_sync(sync_each or micro == 1)
            model(_inputs(rank, micro), branch_both or micro == 0).sum().backward()

    for name, param in model.named_parameters():
        actual = getattr(param, "main_grad", None)
        assert actual is not None, (
            f"Missing main_grad for {name}: fusion={comm_fusion}, branch_both={branch_both}, sync_each={sync_each}"
        )
        expected = reference_parameters[name].grad.chunk(2, dim=0)[rank]
        torch.testing.assert_close(_local(actual), expected, rtol=0, atol=0)
    for unit in model.modules():
        if not isinstance(unit, HSDPModule):
            continue
        for param in unit.hsdp_scheduler.hsdp_state.hsdp_params:
            assert param.unsharded_accumulated_grad is None, f"Unconsumed full gradient for {param}"
            assert param.reduce_partial_output is None, f"Unconsumed reduction partial for {param}"


def test_nested_no_sync_flushes_inactive_microbatch() -> None:
    """Compare nested conditional gradients with an independent graph and check both reduction paths."""
    init_dist_gloo()
    try:
        assert dist.get_world_size() == 2, f"Expected two Gloo ranks, got {dist.get_world_size()}"
        rank = dist.get_rank()
        mesh = init_device_mesh(device_type="cpu", mesh_shape=(2,), mesh_dim_names=("fsdp",))
        for comm_fusion in (False, True):
            for branch_both, sync_each in ((True, True), (True, False), (False, True), (False, False)):
                _run_case(mesh, rank, comm_fusion, branch_both, sync_each)
                dist.barrier()
    finally:
        dist.destroy_process_group()
