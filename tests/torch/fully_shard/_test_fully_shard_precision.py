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
import pytest
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


class UnevenShardModel(torch.nn.Module):
    """Two-layer model whose parameters are uneven on dim 0 for four-way FSDP."""

    def __init__(self):
        super().__init__()
        self.input_projection_weight = torch.nn.Parameter(torch.full((5, 8), 0.01).npu())
        self.output_projection_weight = torch.nn.Parameter(torch.full((7, 5), 0.01).npu())

    def forward(self, x):
        x = torch.matmul(x, self.input_projection_weight.t())
        x = torch.relu(x)
        x = torch.matmul(x, self.output_projection_weight.t())
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


def get_standalone_result(step, acc_grad=False, model_factory=SimpleModel):  # pylint: disable=unused-argument
    """
    Get results from standalone (non-distributed) model training for comparison.

    Args:
        step (int): Number of training steps to execute
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False
        model_factory (Callable): Builds the model used by this precision case.

    Returns:
        tuple: Loss tensor and a mapping of parameter names to full gradients.
    """
    standalone_model = model_factory().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    acc_epoch = 2
    acc_step = 4
    for _ in range(acc_epoch):
        for _ in range(acc_step):
            standalone_loss = standalone_model(standalone_x.npu())
            standalone_loss.backward()
            standalone_grads = {
                param_name: param.grad.data.clone()
                for param_name, param in standalone_model.named_parameters()
            }
            if not acc_grad:
                standalone_optimizer.step()
                standalone_optimizer.zero_grad()
        if acc_grad:
            standalone_optimizer.step()
            standalone_optimizer.zero_grad()
    return standalone_loss, standalone_grads


def _tensor_storage_info(tensor):
    """Return the raw TensorImpl storage pointer and byte size."""
    with torch._C.DisableTorchFunctionSubclass():  # pylint: disable=protected-access
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.nbytes()


def _assert_comm_fusion_flat_buffer_memory(model, optimizer):
    """Verify comm_fusion keeps managed parameter storage in its AllGather buckets."""
    state = model.hsdp_scheduler.hsdp_state
    param_group = state.param_group
    if param_group is None:
        raise AssertionError("comm_fusion should create an HSDPParamGroup.")
    flat_buffers = [
        bucket.flat_param_buffer
        for bucket in param_group.all_gather_buckets
        if bucket.flat_param_buffer is not None
    ]
    if not flat_buffers:
        raise AssertionError("comm_fusion zero-copy should create AllGather flat parameter buffers.")

    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    flat_storage_nbytes = dict(_tensor_storage_info(flat_buffer) for flat_buffer in flat_buffers)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    unique_storage_nbytes = dict(flat_storage_nbytes)
    non_flat_storages = []

    for hsdp_param in state.hsdp_params:
        param_fqn = hsdp_param._param_fqn or "<unknown>"  # pylint: disable=protected-access
        if id(hsdp_param.sharded_param) not in optimizer_param_ids:
            raise AssertionError(f"Optimizer does not hold managed parameter {param_fqn}.")

        checked_tensors = (
            ("_sharded_param_data", hsdp_param._sharded_param_data),  # pylint: disable=protected-access
            ("sharded_param._local_tensor", hsdp_param.sharded_param._local_tensor),  # pylint: disable=protected-access
            ("sharded_param", hsdp_param.sharded_param),
        )
        for label, tensor in checked_tensors:
            storage_ptr, storage_nbytes = _tensor_storage_info(tensor)
            unique_storage_nbytes[storage_ptr] = storage_nbytes
            if storage_ptr not in flat_storage_nbytes:
                non_flat_storages.append((param_fqn, label, storage_ptr, storage_nbytes))

    param_storage_nbytes = sum(unique_storage_nbytes.values())
    flat_storage_total_nbytes = sum(flat_storage_nbytes.values())
    if non_flat_storages or param_storage_nbytes != flat_storage_total_nbytes:
        raise AssertionError(
            "comm_fusion zero-copy should leave managed sharded parameter storage backed only by "
            f"flat_param_buffer after fully_shard. memory_allocated={allocated}, "
            f"flat_param_buffer_nbytes={flat_storage_total_nbytes}, "
            f"managed_param_storage_nbytes={param_storage_nbytes}, "
            f"unique_storage_nbytes={unique_storage_nbytes}, non_flat_storages={non_flat_storages}."
        )


def get_fully_shard_result(
        step,
        acc_grad=False,
        check_comm_fusion_memory=False,
        model_factory=SimpleModel,
        **fsdp_kwargs):  # pylint: disable=unused-argument
    """
    Get results from HSDP (Hybrid Sharded Data Parallel) distributed training.

    Args:
        step (int): Number of training steps to execute
        shard_size (int): Size of parameter sharding, defaults to 1
        optimizer_level (str): Optimization level ("level1", "level2", "level3"), defaults to "level1"
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False
        model_factory (Callable): Builds the model used by this precision case.

    Returns:
        tuple: Loss tensor and a mapping of parameter names to actual local gradients.
    """
    dist_model = model_factory().npu()
    dist_x = standalone_x.npu()
    dist_model = fully_shard(dist_model, **fsdp_kwargs)
    # when loss is sum and DTensor not used, set grad comm type to sum for single-card precision compare
    dist_model.set_reduce_op_type("sum")
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    acc_epoch = 2
    acc_step = 4
    comm_fusion_memory_checked = False
    with SkipDTensorDispatch():
        dist_grads = {}
        for _ in range(acc_epoch):
            for _ in range(acc_step):
                # if i == acc_step - 1:
                #     dist_model.set_requires_grad_sync(True)
                # else:
                #     dist_model.set_requires_grad_sync(False)
                dist_loss = dist_model(dist_x)
                if check_comm_fusion_memory and not comm_fusion_memory_checked:
                    _assert_comm_fusion_flat_buffer_memory(dist_model, dist_optimizer)
                    comm_fusion_memory_checked = True
                # handle backward input
                repeat_num = len(mesh.rank_list)
                backward_input = torch.tensor(1.0 / repeat_num)
                dist_loss.backward(backward_input)
                for param_name, param in dist_model.named_parameters():
                    if param.grad is None:
                        continue
                    assert isinstance(param.grad, DTensor), \
                        f"Expected {param_name}.grad to be a DTensor, but got {type(param.grad)}"
                    dist_grads[param_name] = param.grad.data.clone()
                if not acc_grad:
                    dist_optimizer.step()
                    dist_optimizer.zero_grad()
            if acc_grad:
                dist_optimizer.step()
                dist_optimizer.zero_grad()
    dist_model.reset_iter_state()
    return dist_loss, dist_grads


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


def shard_param_data_parallel(
        acc_grad=False,
        check_comm_fusion_memory=False,
        model_factory=SimpleModel,
        **fsdp_kwargs):
    """shard param data parallel"""
    rank, _ = init_dist()
    step = 4
    mesh: DeviceMesh = fsdp_kwargs['mesh']
    shard_size = mesh.mesh_shape[-1]
    standalone_loss, standalone_grads = get_standalone_result(
        step,
        acc_grad=acc_grad,
        model_factory=model_factory,
    )
    dist_loss, dist_grads = get_fully_shard_result(
        step,
        acc_grad=acc_grad,
        check_comm_fusion_memory=check_comm_fusion_memory,
        model_factory=model_factory,
        **fsdp_kwargs,
    )

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.cpu().detach().numpy(),
                       0.001, 0.001)
    shard_rank = rank % shard_size
    assert standalone_grads.keys() == dist_grads.keys()
    for param_name, standalone_grad in standalone_grads.items():
        dim0_shard_size = (standalone_grad.size(0) + shard_size - 1) // shard_size
        shard_offset = min(shard_rank * dim0_shard_size, standalone_grad.size(0))
        actual_shard_length = min(dim0_shard_size, standalone_grad.size(0) - shard_offset)
        expected_grad = standalone_grad.narrow(0, shard_offset, actual_shard_length)
        dist_grad = dist_grads[param_name]
        assert expected_grad.shape == dist_grad.shape, (
            f"Gradient shape mismatch for {param_name}: "
            f"expected={expected_grad.shape}, actual={dist_grad.shape}"
        )
        assert np.allclose(expected_grad.cpu().detach().numpy(),
                           dist_grad.cpu().detach().numpy(),
                           0.001, 0.001), (
            f"Gradient values mismatch for {param_name}: "
            f"expected={expected_grad}, actual={dist_grad}"
        )


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_zero3_fully_shard(comm_fusion):
    """Test zero3 fully shard through the per-parameter and ParamGroup paths."""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    fsdp_kwargs["comm_fusion"] = comm_fusion
    shard_param_data_parallel(
        acc_grad=False,
        check_comm_fusion_memory=comm_fusion,
        model_factory=UnevenShardModel,
        **fsdp_kwargs,
    )


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


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_zero3_partial_shard(comm_fusion):
    """Test zero3 partial shard through the per-parameter and ParamGroup paths."""
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    fsdp_kwargs["comm_fusion"] = comm_fusion
    shard_param_data_parallel(
        acc_grad=False,
        check_comm_fusion_memory=comm_fusion,
        **fsdp_kwargs,
    )


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_zero3_fully_shard_prefetch_recompute(comm_fusion):
    """Test zero3 fully shard with child-module prefetch and activation recompute on both comm paths."""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    fsdp_kwargs["comm_fusion"] = comm_fusion
    shard_param_data_parallel_prefetch_recompute(**fsdp_kwargs)


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_zero3_partial_shard_prefetch_recompute(comm_fusion):
    """Test zero3 partial shard with child-module prefetch and activation recompute on both comm paths."""
    init_dist()
    op_size = 2
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    hsdp_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, op_size), mesh_dim_names=("dp", "op"))
    fsdp_kwargs['mesh'] = hsdp_mesh
    fsdp_kwargs["comm_fusion"] = comm_fusion
    shard_param_data_parallel_prefetch_recompute(**fsdp_kwargs)


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_zero3_fully_shard_prefetch_recompute_grad_accum(comm_fusion):
    """Test zero3 fully shard with prefetch, activation recompute and grad accumulation on both comm paths."""
    init_dist()
    mp_policy = MixedPrecisionPolicy()
    fsdp_kwargs = _get_standard_fully_shard_kwargs(mp_policy)
    fsdp_kwargs["comm_fusion"] = comm_fusion
    shard_param_data_parallel_prefetch_recompute(acc_grad=True, **fsdp_kwargs)
