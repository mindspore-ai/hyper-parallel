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
"""Verify Torch fully_shard hook state transitions with activation checkpointing."""
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
os.environ.setdefault("HP_LOG_CONFIG", "FSDP:DEBUG")

# pylint: disable=C0413
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.platform.torch.fully_shard.param_group import get_comm_ctx
from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2
from tests.torch.utils import init_dist_gloo


class _Block(nn.Module):
    """Small deterministic module managed by one fully_shard scheduler."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.float32).view(4, 4) / 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs @ self.weight)


class _TracingTorchHSDPScheduler(TorchHSDPSchedulerV2):
    """Record scheduler transitions while retaining the production hook implementation."""

    def __init__(self, scenario: str, *args, **kwargs) -> None:
        self.scenario = scenario
        self.trace = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _state_name(state) -> str:
        return "NONE" if state is None else state.name

    def _record(self, event: str, before) -> None:
        before_name = self._state_name(before)
        after_name = self._state_name(self.scheduler_state)
        self.trace.append((event, before_name, after_name))
        print(
            f"FSDP_STATE_TRACE rank={dist.get_rank()} scenario={self.scenario} "
            f"event={event} before={before_name} after={after_name} "
            f"root_bp_state={HSDPSchedulerV2.root_bp_state}",
            flush=True,
        )

    def _forward_pre_hook(self, cell, args, kwargs):
        before = self.scheduler_state
        result = super()._forward_pre_hook(cell, args, kwargs)
        self._record("forward_pre", before)
        return result

    def _forward_hook(self, cell, inputs, outputs):
        before = self.scheduler_state
        result = super()._forward_hook(cell, inputs, outputs)
        self._record("forward", before)
        return result

    def _backward_pre_hook(self, grad):
        before = self.scheduler_state
        result = super()._backward_pre_hook(grad)
        self._record("backward_pre", before)
        return result

    def _backward_hook(self):
        before = self.scheduler_state
        super()._backward_hook()
        self._record("backward", before)

    def _root_backward_hook(self):
        before = self.scheduler_state
        self._record("root_backward_enter", before)
        super()._root_backward_hook()
        self._record("root_backward_exit", before)


def _reset_global_state() -> None:
    HSDPSchedulerV2.root_bp_state = False
    HSDPState.pre_reduce_scatter_params.clear()
    HSDPState.pre_all_reduce_params.clear()
    TorchHSDPStateV2.pre_direct_all_reduce_grads.clear()
    TorchHSDPStateV2.pre_all_reduce_groups.clear()
    TorchHSDPStateV2.all_reduce_work_groups.clear()
    comm_ctx = get_comm_ctx()
    comm_ctx.comm_handle = None
    comm_ctx.all_reduce_handle = None
    comm_ctx.pre_param_group = None
    comm_ctx.all_reduce_param_group = None


def _build_model(scenario: str, mesh) -> tuple[nn.Module, _TracingTorchHSDPScheduler]:
    model = _Block()
    _extend_module_with_hsdp_interface(model)
    scheduler = _TracingTorchHSDPScheduler(
        scenario,
        model,
        mesh,
        True,
        None,
        MixedPrecisionPolicy(),
        OffloadPolicy(),
        set(),
        set(),
        torch.device("cpu"),
        False,
    )
    model.hsdp_scheduler = scheduler
    return model, scheduler


def _run_scenario(scenario: str, mesh) -> list[tuple[str, str, str]]:
    _reset_global_state()
    model, scheduler = _build_model(scenario, mesh)
    inputs = torch.arange(8, dtype=torch.float32).view(2, 4) / 8

    if scenario == "normal":
        outputs = model(inputs)
    elif scenario == "non_reentrant":
        outputs = checkpoint(model, inputs, use_reentrant=False)
    elif scenario == "reentrant":
        dummy = torch.ones((), requires_grad=True)

        def run_with_dummy(tensor, unused_dummy):
            del unused_dummy
            return model(tensor)

        outputs = checkpoint(run_with_dummy, inputs, dummy, use_reentrant=True)
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    outputs.sum().backward()
    assert model.weight.grad is not None, (
        f"Expected a reduced gradient after {scenario} backward, got model.weight.grad={model.weight.grad}"
    )
    return scheduler.trace


def test_torch_fully_shard_hook_state_machine() -> None:
    """Verify normal, non-reentrant, and reentrant hook state transitions with real autograd."""
    init_dist_gloo()
    mesh = init_device_mesh(
        device_type="cpu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("fsdp",),
    )
    expected_traces = {
        "normal": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "BACKWARD"),
        ],
        "non_reentrant": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("forward_pre", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "BACKWARD"),
        ],
        "reentrant": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("forward_pre", "FORWARD", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "BACKWARD"),
        ],
    }

    for scenario, expected_trace in expected_traces.items():
        actual_trace = _run_scenario(scenario, mesh)
        assert actual_trace == expected_trace, (
            f"Unexpected hook state trace for scenario={scenario}: "
            f"expected={expected_trace}, got={actual_trace}"
        )
