# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Test fully_shard replicate_params: verify replicate params are not sharded and grads match reference.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.torch.utils import init_dist
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule
from hyper_parallel import init_device_mesh, DeviceMesh
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor


# -------- Nested model: A -> module_b (B) -> lin_b, module_c (C) -> lin_c; B has subtrahend --------


class C(nn.Module):
    """Model C with replicate params"""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_c = nn.Linear(dim, dim)
        self.output_scale_weight = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_c(x) + self.output_scale_weight.mean()


class B(nn.Module):
    """Model B."""
    def __init__(self, dim: int, subtrahend: torch.Tensor) -> None:
        super().__init__()
        self.lin_b = nn.Linear(dim, dim)
        self.module_c = C(dim)
        self.subtrahend = nn.Parameter(subtrahend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_result = self.module_c(x)
        return self.lin_b(c_result) - self.subtrahend


class A(nn.Module):
    """Model A."""
    def __init__(self, dim, addend, subtrahend) -> None:
        super().__init__()
        self.module_b = B(dim, subtrahend)
        self.addend = nn.Parameter(addend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.module_b(x) + self.addend
        return result.sum()


def _append_prefix(prefix: str, name: str) -> str:
    if prefix and name:
        return prefix + "." + name
    return prefix + name


def _discover_params_under_path(module: nn.Module, target_path: str, path: str) -> set:
    """Collect all parameters under the subtree at target_path."""
    if path == target_path:
        return set(p for _, p in module.named_parameters(path))
    out = set()
    for name, child in module.named_children():
        child_path = _append_prefix(path, name)
        out |= _discover_params_under_path(child, target_path, child_path)
    return out


def _get_tensor_view(param, *, gather_dtensor: bool = False):
    """Tensor view for comparison."""
    if isinstance(param, DTensor):
        return param.full_tensor() if gather_dtensor else param.to_local()
    return param


def _compare_params(ref_param, test_param, rtol=1e-3, atol=1e-3):
    """Compare two parameters (support DTensor via full_tensor())."""
    ref_t = _get_tensor_view(ref_param, gather_dtensor=True)
    test_t = _get_tensor_view(test_param, gather_dtensor=True)
    ref_np = ref_t.detach().cpu().numpy()
    test_np = test_t.detach().cpu().numpy()
    assert np.allclose(ref_np, test_np, rtol=rtol, atol=atol), \
        f"param mismatch: ref shape {ref_np.shape} vs test shape {test_np.shape}"


def get_standalone_result(step, acc_grad=False):
    """Standalone run for nested model A; returns loss and key grads."""
    dim = 8
    torch.manual_seed(42)
    addend = torch.randn(dim, dim).npu()
    torch.manual_seed(70)
    subtrahend = torch.randn(dim, dim).npu()
    model = A(dim, addend, subtrahend).npu()

    torch.manual_seed(84)
    inp = torch.randn(dim, dim).npu()

    opt = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(step):
        loss = model(inp)
        loss.backward()
        grads = {
            "addend": model.addend.grad.clone(),
            "lin_b.weight": model.module_b.lin_b.weight.grad.clone(),
            "lin_b.bias": model.module_b.lin_b.bias.grad.clone(),
            "subtrahend": model.module_b.subtrahend.grad.clone(),
            "lin_c.weight": model.module_b.module_c.lin_c.weight.grad.clone(),
            "lin_c.bias": model.module_b.module_c.lin_c.bias.grad.clone(),
            "output_scale_weight": model.module_b.module_c.output_scale_weight.grad.clone(),
        }
        if not acc_grad:
            opt.step()
            opt.zero_grad()
    if acc_grad:
        opt.step()
        opt.zero_grad()

    return loss, grads


def get_fully_shard_result(step, acc_grad=False, **fsdp_kwargs):
    """
    Run nested model with fully_shard(..., replicate_params=...) and return
    loss + key grads. Backward uses 1/|mesh| scaling to match standalone.
    """
    dim = 8
    torch.manual_seed(42)
    addend = torch.randn(dim, dim).npu()
    torch.manual_seed(70)
    subtrahend = torch.randn(dim, dim).npu()
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    with SkipDTensorDispatch():
        model = A(dim, addend, subtrahend).npu()

    torch.manual_seed(84)
    inp = torch.randn(dim, dim).npu()

    replicate_set = _discover_params_under_path(model, "module_b.module_c", "")
    fsdp_kwargs['replicate_params'] = replicate_set
    dist_model = fully_shard(model, **fsdp_kwargs)
    dist_model.set_reduce_op_type("sum")
    output_scale_weight = dist_model.module_b.module_c.output_scale_weight
    output_scale_hsdp_param = next(
        hsdp_param for hsdp_param in dist_model.hsdp_scheduler.hsdp_state.hsdp_params
        if hsdp_param.sharded_param is output_scale_weight
    )
    assert output_scale_weight.shape[0] % mesh.mesh_shape[-1] != 0
    assert output_scale_hsdp_param.is_replicate_param and output_scale_hsdp_param.shard_world_size == 1

    opt = optim.SGD(dist_model.parameters(), lr=0.01)
    with SkipDTensorDispatch():
        for _ in range(step):
            loss = dist_model(inp)
            loss.backward(torch.tensor(1.0 / len(mesh.rank_list)))
            grads = {
                "addend": dist_model.addend.grad,
                "lin_b.weight": dist_model.module_b.lin_b.weight.grad,
                "lin_b.bias": dist_model.module_b.lin_b.bias.grad,
                "subtrahend": dist_model.module_b.subtrahend.grad,
                "lin_c.weight": dist_model.module_b.module_c.lin_c.weight.grad,
                "lin_c.bias": dist_model.module_b.module_c.lin_c.bias.grad,
                "output_scale_weight": dist_model.module_b.module_c.output_scale_weight.grad,
            }
            grads = {k: v.clone() if v is not None else None for k, v in grads.items()}
            if not acc_grad:
                opt.step()
                opt.zero_grad()
        if acc_grad:
            opt.step()
            opt.zero_grad()

    return loss, grads


def shard_param_data_parallel(acc_grad=False, **fsdp_kwargs):
    """shard param data parallel"""
    rank, _ = init_dist()
    step = 20
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    shard_size = mesh.mesh_shape[-1]
    standalone_loss, standalone_grad = get_standalone_result(step, acc_grad=acc_grad)
    dist_loss, dist_grad = get_fully_shard_result(step, acc_grad=acc_grad, **fsdp_kwargs)

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.cpu().detach().numpy(),
                       0.001, 0.001)
    dp_stride = 8 // shard_size
    dp_offset = rank % shard_size * dp_stride
    for (_, ref_param), (_, test_param) in zip(standalone_grad.items(), dist_grad.items()):
        compare_param = _get_tensor_view(test_param)
        if ref_param.shape != compare_param.shape:
            # fully shard
            _compare_params(ref_param[dp_offset: dp_offset + dp_stride], compare_param, rtol=0.001, atol=0.001)
        else:
            # replicate_params
            _compare_params(ref_param, compare_param, rtol=0.001, atol=0.001)


def _get_standard_fully_shard_kwargs(mp_policy, offload_policy=None):
    """get standard fully shard kwargs"""
    default_world_size = dist.get_world_size() if dist.is_initialized() else 8
    default_mesh = init_device_mesh(device_type="npu", mesh_shape=(default_world_size,), mesh_dim_names=("dp",))
    fsdp_kwargs = {
        'mesh': default_mesh,
        'reshard_after_forward': True,
        'shard_placement_fn': None,
        'mp_policy': mp_policy,
        'offload_policy': offload_policy,
        'replicate_params': None,
    }
    return fsdp_kwargs


def test_zero3_fully_shard_replicate_params():
    """test zero3 fully shard parallel with replicate params"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    for acc_grad in [False, True]:
        shard_param_data_parallel(acc_grad=acc_grad, **fsdp_kwargs)


def test_zero3_fully_shard_replicate_params_with_offload():
    """test zero3 fully shard parallel with offload"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy, offload_policy)
    for acc_grad in [False, True]:
        shard_param_data_parallel(acc_grad=acc_grad, **fsdp_kwargs)


def test_zero3_partial_shard_replicate_params():
    """test zero3 partial shard parallel"""
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(4, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    for acc_grad in [False, True]:
        shard_param_data_parallel(acc_grad=acc_grad, **fsdp_kwargs)
