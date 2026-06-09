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
"""Distributed test: set_gradient_scaling_factor under HSDP + grad accumulation
+ replicate_params (pure all-reduce path).

Validates that the scaling factor is applied exactly once to every managed
parameter's gradient -- including replicate_params whose grads only go through
all-reduce (no reduce-scatter) -- and that under gradient accumulation the
result is ``sum_i(g_i * factor)`` rather than scaling the accumulated grad.

Methodology: run the same fixed input twice, once with factor=None (baseline)
and once with factor=k (scaled), and assert ``scaled_grad ~= baseline_grad * k``
for both sharded and replicate parameters. This is reduce-op agnostic (works for
sum/avg) since it only checks the linear ratio.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch import nn

from tests.torch.utils import init_dist
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel import init_device_mesh, DeviceMesh, SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor


class C(nn.Module):
    """Leaf module whose params are kept replicated (pure all-reduce path)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_c = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_c(x)


class B(nn.Module):
    """Middle module mixing sharded params and a replicate subtree (module_c)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_b = nn.Linear(dim, dim)
        self.module_c = C(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_b(self.module_c(x))


class A(nn.Module):
    """Root module."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.module_b = B(dim)
        self.addend = nn.Parameter(torch.randn(dim, dim).npu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.module_b(x) + self.addend).sum()


_GRAD_KEYS = (
    "addend",               # sharded
    "lin_b.weight",         # sharded
    "lin_b.bias",           # sharded
    "lin_c.weight",         # replicate (pure all-reduce)
    "lin_c.bias",           # replicate (pure all-reduce)
)


def _collect_grads(model):
    """Return cloned local-shard grad numpy for each tracked param."""
    raw = {
        "addend": model.addend.grad,
        "lin_b.weight": model.module_b.lin_b.weight.grad,
        "lin_b.bias": model.module_b.lin_b.bias.grad,
        "lin_c.weight": model.module_b.module_c.lin_c.weight.grad,
        "lin_c.bias": model.module_b.module_c.lin_c.bias.grad,
    }
    out = {}
    for key, grad in raw.items():
        assert grad is not None, f"grad for {key} is None"
        local = grad.to_local() if isinstance(grad, DTensor) else grad
        out[key] = local.detach().clone().cpu().numpy()
    return out


def _discover_replicate_params(model):
    """Params under module_b.module_c are kept replicated."""
    return set(p for _, p in model.module_b.module_c.named_parameters())


def _get_gradient_scaling_factor(model):
    """Read the factor back from whichever layer the setter wrote it to.

    Mirrors HSDPState.set_gradient_scaling_factor: comm_fusion stores it on the
    param_group, otherwise per managed param. Kept local to the test since the
    factor is write-only on the public API.
    """
    state = model.hsdp_scheduler.hsdp_state
    param_group = getattr(state, "param_group", None)
    if param_group is not None:
        return param_group.gradient_scaling_factor
    for hsdp_param in state._iter_managed_params():  # pylint: disable=protected-access
        return hsdp_param.gradient_scaling_factor
    return None


def _build_model(mesh: DeviceMesh, factor):
    """Build an HSDP fully_shard model with module_c replicated; set factor."""
    dim = 8
    torch.manual_seed(42)
    with SkipDTensorDispatch():
        model = A(dim).npu()
    replicate_set = _discover_replicate_params(model)
    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(),
        replicate_params=replicate_set,
    )
    # sum reduce so every rank holds an identical (un-averaged) grad, keeping the
    # baseline/scaled ratio exact regardless of replica/shard layout.
    model.set_reduce_op_type("sum")
    model.set_gradient_scaling_factor(factor)
    assert _get_gradient_scaling_factor(model) == (factor if factor is not None else None)
    return model


def _run_accumulated_grads(mesh: DeviceMesh, factor, micro_steps: int):
    """Accumulate grads over several micro-batches (no optimizer step) and return them."""
    dim = 8
    torch.manual_seed(84)
    inputs = [torch.randn(dim, dim).npu() for _ in range(micro_steps)]

    model = _build_model(mesh, factor)
    with SkipDTensorDispatch():
        for i in range(micro_steps):
            # Only the last micro-batch syncs grads; earlier ones just accumulate
            # the unsharded grad locally (standard grad-accumulation pattern).
            model.set_requires_gradient_sync(i == micro_steps - 1)
            loss = model(inputs[i])
            loss.backward()
    return _collect_grads(model)


def _assert_ratio(baseline, scaled, factor):
    """Assert scaled[k] ~= baseline[k] * factor for every tracked param."""
    for key in _GRAD_KEYS:
        expected = baseline[key] * float(factor)
        assert np.allclose(scaled[key], expected, rtol=1e-3, atol=1e-3), (
            f"param {key}: scaling_factor={factor} not applied as sum_i(g_i*factor). "
            f"baseline={baseline[key]}, scaled={scaled[key]}, expected={expected}"
        )


def _run_check(mesh: DeviceMesh):
    micro_steps = 3
    baseline = _run_accumulated_grads(mesh, None, micro_steps)
    for factor in (2.0, 0.5, 3):
        scaled = _run_accumulated_grads(mesh, factor, micro_steps)
        _assert_ratio(baseline, scaled, factor)
    # Tensor factor on device.
    factor_tensor = torch.tensor(0.25, device="npu", dtype=torch.float32)
    scaled_t = _run_accumulated_grads(mesh, factor_tensor, micro_steps)
    _assert_ratio(baseline, scaled_t, 0.25)


def _run_multi_reduce_cycle_grads(mesh: DeviceMesh, factor, num_cycles: int):
    """Run several full forward/backward/reduce cycles WITHOUT zeroing grads.

    Every cycle syncs (``set_requires_gradient_sync(True)``), so each one runs a
    full reduce and ``apply_reduced_grad`` takes its accumulate (``+=``) branch
    from the second cycle on. This is the high-frequency "accumulate sharded
    grads across reduce cycles" pattern -- distinct from the single-reduce
    micro-batch accumulation in ``_run_accumulated_grads`` -- and is exactly
    the case the ``apply_reduced_grad`` note guards: the factor must scale each
    reduce input independently (giving ``factor * sum_cycles``) and must never
    re-scale the already-accumulated sharded grad (which would give a
    ``factor**2`` term on the earlier cycles).
    """
    dim = 8
    torch.manual_seed(126)
    inputs = [torch.randn(dim, dim).npu() for _ in range(num_cycles)]

    model = _build_model(mesh, factor)
    with SkipDTensorDispatch():
        for i in range(num_cycles):
            model.set_requires_gradient_sync(True)
            loss = model(inputs[i])
            loss.backward()
    return _collect_grads(model)


def _run_multi_cycle_check(mesh: DeviceMesh):
    num_cycles = 2
    baseline = _run_multi_reduce_cycle_grads(mesh, None, num_cycles)
    for factor in (2.0, 0.5, 3):
        scaled = _run_multi_reduce_cycle_grads(mesh, factor, num_cycles)
        _assert_ratio(baseline, scaled, factor)
    factor_tensor = torch.tensor(0.25, device="npu", dtype=torch.float32)
    scaled_t = _run_multi_reduce_cycle_grads(mesh, factor_tensor, num_cycles)
    _assert_ratio(baseline, scaled_t, 0.25)


def test_gradient_scaling_factor_hsdp_grad_accumulation():
    """
    Feature: HSDPModule.set_gradient_scaling_factor under HSDP + grad accumulation
        + replicate_params.
    Description: 4-card HSDP mesh (2x2). module_c params are replicate_params
        (pure all-reduce path), others are reduce-scatter+all-reduce. Covers two
        accumulation patterns: (1) single-reduce micro-batch accumulation (sync
        only on the last micro-batch), and (2) multi-cycle accumulation where
        every cycle reduces and the sharded grads accumulate across reduce
        cycles (apply_reduced_grad's += branch). Compares scaled vs baseline.
    Expectation: scaled grad == baseline grad * factor for every parameter,
        including the replicate (pure all-reduce) ones, under both patterns;
        i.e. the factor is applied exactly once per reduce input and the
        already-accumulated sharded grad is never re-scaled.
    """
    init_dist()
    hsdp_mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "op")
    )
    _run_check(hsdp_mesh)
    _run_multi_cycle_check(hsdp_mesh)
