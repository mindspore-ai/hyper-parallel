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
"""Exercise module-tree backward boundaries using real CPU/Gloo autograd."""
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from tests.torch.fully_shard._test_fully_shard_gradient_accumulation import _ConditionalModel, _inputs, _local
from tests.torch.utils import init_dist_gloo


class _CountingScheduler(HSDPSchedulerV2):
    """Count final callbacks without replacing any production hook or collective."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the scheduler and its completed-callback counter."""
        self.final_callbacks = 0
        super().__init__(*args, **kwargs)

    def _root_backward_hook(self):
        self.final_callbacks += 1
        super()._root_backward_hook()


class _CheckpointModel(_ConditionalModel):
    """Checkpoint each nested unit while the root owns the outer autograd graph."""

    def __init__(self, checkpoint_mode: str) -> None:
        """Select how each child unit is recomputed during backward."""
        super().__init__()
        self.checkpoint_mode = checkpoint_mode

    def _run_block(self, block: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_mode == "none":
            return block(inputs)
        return checkpoint(block, inputs, use_reentrant=self.checkpoint_mode == "reentrant")

    def forward(self, inputs: torch.Tensor, use_early: bool) -> torch.Tensor:
        """Run the selected units with the requested checkpoint implementation."""
        result = self._run_block(self.common, inputs)
        return result + self._run_block(self.early, inputs) if use_early else result


def _wrap(model: nn.Module, mesh, fusion: bool, wrapping: str = "nested") -> None:
    policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32,
        cast_forward_inputs=False, apply_grad_on_fp32_main_grad=True,
    )
    if wrapping == "nested":
        units = (model.common, model.early, model)
    elif wrapping == "grouped":
        units = ((model.common, model.early), model)
    else:
        units = (model,)
    for unit in units:
        modules = unit if isinstance(unit, tuple) else (unit,)
        for module in modules:
            _extend_module_with_hsdp_interface(module)
        scheduler = _CountingScheduler(
            unit, mesh, True, None, policy, OffloadPolicy(), set(), set(), torch.device("cpu"), fusion,
        )
        for module in modules:
            module.hsdp_scheduler = scheduler
    model.set_reduce_op_type("sum")


def _check_boundary(model: nn.Module, expected_callbacks: int) -> None:
    root = model.hsdp_scheduler
    assert root.final_callbacks == expected_callbacks, (
        f"Expected {expected_callbacks} tree callbacks, got {root.final_callbacks}"
    )
    for scheduler in root.scheduler_ctx.all_hsdp_schedulers:
        assert scheduler.scheduler_state is None, (
            f"Backward state leaked for {scheduler.hsdp_state}: {scheduler.scheduler_state}"
        )
    assert not root.scheduler_ctx.root_bp_state, (
        f"Expected an inactive backward context, got root_bp_state={root.scheduler_ctx.root_bp_state}"
    )


def _check_gradients(model: nn.Module, reference: nn.Module) -> None:
    expected_params = dict(reference.named_parameters())
    rank = dist.get_rank()
    for name, param in model.named_parameters():
        actual = getattr(param, "main_grad", None)
        expected = expected_params[name].grad
        if expected is None:
            assert actual is None, f"Unused parameter {name} unexpectedly has a gradient"
        else:
            assert actual is not None, f"Missing main_grad for {name}"
            torch.testing.assert_close(_local(actual), expected.chunk(2, dim=0)[rank], rtol=0, atol=0)
    for scheduler in model.hsdp_scheduler.scheduler_ctx.all_hsdp_schedulers:
        for param in scheduler.hsdp_state.hsdp_params:
            assert param.unsharded_accumulated_grad is None, f"Unconsumed full gradient for {param}"
            assert param.reduce_partial_output is None, f"Unconsumed partial for {param}"


def _accumulation_case(mesh, fusion: bool, wrapping: str, checkpoint_mode: str, input_grad: bool) -> None:
    reference = _ConditionalModel()
    model = _CheckpointModel(checkpoint_mode)
    _wrap(model, mesh, fusion, wrapping)
    rank = dist.get_rank()
    # Two accumulation windows on the same objects, without reset_iter_state.
    # Retaining final grads between windows also checks exactly-once accumulation.
    for window in range(2):
        for source_rank in range(2):
            for micro in range(3):
                reference(_inputs(source_rank, micro + window), micro == 0).sum().backward()
        with SkipDTensorDispatch():
            for micro in range(3):
                model.set_requires_gradient_sync(micro == 2)
                inputs = _inputs(rank, micro + window).requires_grad_(input_grad)
                model(inputs, micro == 0).sum().backward()
                _check_boundary(model, window * 3 + micro + 1)
                if micro < 2:
                    params = [p for s in model.hsdp_scheduler.scheduler_ctx.all_hsdp_schedulers
                              for p in s.hsdp_state.hsdp_params]
                    assert any(p.unsharded_accumulated_grad is not None for p in params), (
                        f"Expected pending no-sync gradients at window={window}, micro={micro}, got no buffers"
                    )
        _check_gradients(model, reference)


def test_backward_boundaries_preserve_accumulation() -> None:
    """Verify nested/single/grouped ownership, input hooks, and both checkpoint modes."""
    init_dist_gloo()
    try:
        assert dist.get_world_size() == 2, f"Expected two Gloo ranks, got {dist.get_world_size()}"
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        for fusion in (False, True):
            for wrapping in ("single", "nested", "grouped"):
                for input_grad in (False, True):
                    _accumulation_case(mesh, fusion, wrapping, "none", input_grad)
            for mode in ("non_reentrant", "reentrant"):
                _accumulation_case(mesh, fusion, "nested", mode, True)
    finally:
        dist.destroy_process_group()


def test_child_backward_finalizes_the_shared_tree() -> None:
    """Child-only backward must queue one root callback even with no input gradient hook."""
    init_dist_gloo()
    try:
        assert dist.get_world_size() == 2, f"Expected two Gloo ranks, got {dist.get_world_size()}"
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        rank = dist.get_rank()
        for fusion in (False, True):
            for input_grad in (False, True):
                reference = _ConditionalModel()
                model = copy.deepcopy(reference)
                _wrap(model, mesh, fusion)
                with SkipDTensorDispatch():
                    # Initialize ownership without putting root outputs in the backward graph.
                    with torch.no_grad():
                        model(_inputs(rank, 0), False)
                    for step in range(2):
                        for source_rank in range(2):
                            reference.early(_inputs(source_rank, step)).sum().backward()
                        inputs = _inputs(rank, step).requires_grad_(input_grad)
                        model.early(inputs).sum().backward()
                        _check_gradients(model, reference)
                        _check_boundary(model, step + 1)
    finally:
        dist.destroy_process_group()
