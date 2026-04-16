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
"""test data parallel"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import torch
import torch_npu
from torch import optim
from hyper_parallel import DTensor, init_device_mesh, DeviceMesh, SkipDTensorDispatch
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper, CheckpointPolicy, SwapManager
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from tests.torch.utils import init_dist
from tests.torch.common_net import SimpleModel, DenseNet


torch.manual_seed(0)
standalone_x = torch.rand(8, 8)


class SimpleRecomputeModel(torch.nn.Module):
    """Small nested model used to verify fully_shard prefetch with activation recompute."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8, 8).npu())
        self.layers = torch.nn.ModuleList([DenseNet(8, 8, has_bias=False) for _ in range(3)])

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return torch.sum(x)

def _get_standard_fully_shard_kwargs(mp_policy, offload_policy=None):
    """get standard fully shard kwargs"""
    default_mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    fsdp_kwargs = {
        'mesh': default_mesh,
        'reshard_after_forward': True,
        'shard_placement_fn': None,
        'mp_policy': mp_policy,
        'offload_policy': offload_policy,
        'ignored_params': None
    }
    return fsdp_kwargs


def get_standalone_result(step, acc_grad=False):  # pylint: disable=unused-argument
    """
    Get results from standalone (non-distributed) model training for comparison.

    Args:
        step (int): Number of training steps to execute
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False

    Returns:
        tuple: (loss tensor, gradient tensor) from standalone training
    """
    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    acc_epoch = 2
    acc_step = 4
    for _ in range(acc_epoch):
        for _ in range(acc_step):
            standalone_loss = standalone_model(standalone_x.npu())
            standalone_loss.backward()
            standalone_grad = standalone_model.weight.grad.data.clone()  # using for validation
            if not acc_grad:
                standalone_optimizer.step()
                standalone_optimizer.zero_grad()
        if acc_grad:
            standalone_optimizer.step()
            standalone_optimizer.zero_grad()
    return standalone_loss, standalone_grad


def get_fully_shard_result(step, acc_grad=False, **fsdp_kwargs):  # pylint: disable=unused-argument
    """
    Get results from HSDP (Hybrid Sharded Data Parallel) distributed training.

    Args:
        step (int): Number of training steps to execute
        shard_size (int): Size of parameter sharding, defaults to 1
        optimizer_level (str): Optimization level ("level1", "level2", "level3"), defaults to "level1"
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False

    Returns:
        tuple: (loss tensor, gradient tensor) from distributed HSDP training
    """
    dist_model = SimpleModel().npu()
    dist_x = standalone_x.npu()
    dist_model = fully_shard(dist_model, **fsdp_kwargs)
    # when loss is sum and DTensor not used, set grad comm type to sum for single-card precision compare
    dist_model.set_reduce_op_type("sum")
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    acc_epoch = 2
    acc_step = 4
    with SkipDTensorDispatch():
        dist_grad = None
        for _ in range(acc_epoch):
            for _ in range(acc_step):
                # if i == acc_step - 1:
                #     dist_model.set_requires_grad_sync(True)
                # else:
                #     dist_model.set_requires_grad_sync(False)
                dist_loss = dist_model(dist_x)
                # handle backward input
                repeat_num = len(mesh.rank_list)
                backward_input = torch.tensor(1.0 / repeat_num)
                dist_loss.backward(backward_input)
                if dist_model.weight.grad is not None:
                    assert isinstance(dist_model.weight.grad, DTensor),\
                        f"Expected dist_model.weight.grad to be a DTensor, but got {type(dist_model.weight.grad)}"
                    dist_grad = dist_model.weight.grad.data.clone()
                if not acc_grad:
                    dist_optimizer.step()
                    dist_optimizer.zero_grad()
            if acc_grad:
                dist_optimizer.step()
                dist_optimizer.zero_grad()
    return dist_loss, dist_grad


def _build_prefetch_recompute_model(enable_recompute=False):
    """Create the nested test model and optionally wrap child layers with activation recompute."""
    model = SimpleRecomputeModel().npu()
    def swap_policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
        return CheckpointPolicy.MUST_SWAP
    def recomp_policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
        return CheckpointPolicy.MUST_RECOMPUTE
    if enable_recompute:
        model.layers[0] = checkpoint_wrapper(model.layers[0], policy_fn=swap_policy_fn)
        model.layers[1] = checkpoint_wrapper(model.layers[1], policy_fn=recomp_policy_fn)
        model.layers[2] = swap_wrapper(model.layers[2])
        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    return model


def _setup_prefetch_for_layers(layers):
    """Configure one-hop forward/backward prefetch among fully_shard child layers."""
    for idx in range(len(layers) - 1):
        layers[idx].set_modules_to_forward_prefetch([layers[idx + 1]])
    for idx in range(len(layers) - 1, 0, -1):
        layers[idx].set_modules_to_backward_prefetch([layers[idx - 1]])


def get_standalone_prefetch_recompute_result(step, acc_grad=False):  # pylint: disable=unused-argument
    """Reference eager training result for the nested prefetch+recompute model."""
    standalone_model = _build_prefetch_recompute_model(enable_recompute=False)
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    acc_epoch = 2
    acc_step = 4
    for _ in range(acc_epoch):
        for _ in range(acc_step):
            standalone_loss = standalone_model(standalone_x.npu())
            standalone_loss.backward()
            standalone_grad = standalone_model.weight.grad.data.clone()
            if not acc_grad:
                standalone_optimizer.step()
                standalone_optimizer.zero_grad()
        if acc_grad:
            standalone_optimizer.step()
            standalone_optimizer.zero_grad()
    return standalone_loss, standalone_grad


def get_fully_shard_prefetch_recompute_result(step, acc_grad=False, **fsdp_kwargs):  # pylint: disable=unused-argument
    """Distributed result for nested fully_shard child layers with prefetch and recompute enabled."""
    dist_x = standalone_x.npu()

    dist_model = _build_prefetch_recompute_model(enable_recompute=True)
    for layer in dist_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    _setup_prefetch_for_layers(dist_model.layers)
    dist_model = fully_shard(dist_model, **fsdp_kwargs)

    dist_model.set_reduce_op_type("sum")
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    acc_epoch = 2
    acc_step = 4
    with SkipDTensorDispatch():
        dist_grad = None
        for _ in range(acc_epoch):
            for _ in range(acc_step):
                dist_loss = dist_model(dist_x)
                repeat_num = len(mesh.rank_list)
                backward_input = torch.tensor(1.0 / repeat_num)
                dist_loss.backward(backward_input)
                if dist_model.weight.grad is not None:
                    assert isinstance(dist_model.weight.grad, DTensor), \
                        f"Expected dist_model.weight.grad to be a DTensor, but got {type(dist_model.weight.grad)}"
                    dist_grad = dist_model.weight.grad.data.clone()
                if not acc_grad:
                    dist_optimizer.step()
                    dist_optimizer.zero_grad()
            if acc_grad:
                dist_optimizer.step()
                dist_optimizer.zero_grad()
    return dist_loss, dist_grad


def shard_param_data_parallel_prefetch_recompute(acc_grad=False, **fsdp_kwargs):
    """Compare nested fully_shard prefetch+recompute training against eager baseline."""
    rank, _ = init_dist()
    step = 4
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    shard_size = mesh.mesh_shape[-1]
    standalone_loss, standalone_grad = get_standalone_prefetch_recompute_result(step, acc_grad=acc_grad)
    dist_loss, dist_grad = get_fully_shard_prefetch_recompute_result(step, acc_grad=acc_grad, **fsdp_kwargs)

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.cpu().detach().numpy(),
                       0.001, 0.001)
    dp_stride = 8 // shard_size
    dp_offset = rank % shard_size * dp_stride
    assert np.allclose(standalone_grad.cpu().detach().numpy()[dp_offset: dp_offset + dp_stride, :],
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)


def shard_param_data_parallel(acc_grad=False, **fsdp_kwargs):
    """shard param data parallel"""
    rank, _ = init_dist()
    step = 4
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    shard_size = mesh.mesh_shape[-1]
    standalone_loss, standalone_grad = get_standalone_result(step, acc_grad=acc_grad)
    dist_loss, dist_grad = get_fully_shard_result(step, acc_grad=acc_grad, **fsdp_kwargs)

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.cpu().detach().numpy(),
                       0.001, 0.001)
    dp_stride = 8 // shard_size
    dp_offset = rank % shard_size * dp_stride
    assert np.allclose(standalone_grad.cpu().detach().numpy()[dp_offset: dp_offset + dp_stride, :],
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)


def test_zero3_fully_shard():
    """test zero3 fully shard parallel"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)


def test_zero3_fully_shard_with_mp():
    """test zero3 fully shard parallel with mixed precision"""
    init_dist()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32,
                                     output_dtype=torch.float32, cast_forward_inputs=True)
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)


def test_zero3_fully_shard_with_offload():
    """test zero3 fully shard parallel with offload"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy, offload_policy)
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)


def test_zero3_partial_shard():
    """test zero3 partial shard parallel"""
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)


def test_zero3_fully_shard_prefetch_recompute():
    """test zero3 fully shard parallel with child-module prefetch and activation recompute"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    shard_param_data_parallel_prefetch_recompute(**fsdp_kwargs)


def test_zero3_partial_shard_prefetch_recompute():
    """test zero3 partial shard parallel with child-module prefetch and activation recompute"""
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    shard_param_data_parallel_prefetch_recompute(**fsdp_kwargs)


def test_zero3_fully_shard_prefetch_recompute_grad_accum():
    """test zero3 fully shard parallel with child-module prefetch, activation recompute and grad accumulation"""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    shard_param_data_parallel_prefetch_recompute(acc_grad=True, **fsdp_kwargs)


def test_zero3_fully_shard_comm_fusion():
    """
    Feature: Test_zero3_fully_shard_comm_fusion.
    Description: Test_zero3_fully_shard with comm fusion.
    Expectation: case run successfully.
    """
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    fsdp_kwargs["comm_fusion"] = True
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)


def test_zero3_partial_shard_comm_fusion():
    """
    Feature: Test_zero3_partial_shard_comm_fusion.
    Description: Test_zero3_partial_shard with comm fusion.
    Expectation: case run successfully.
    """
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    fsdp_kwargs["comm_fusion"] = True
    shard_param_data_parallel(acc_grad=False, **fsdp_kwargs)
