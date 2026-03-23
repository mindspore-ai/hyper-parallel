# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test TorchHSDPParamV2 implementation"""
# pylint: disable=W0611
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch.distributed as dist
import torch_npu
import torch
from tests.torch.utils import init_dist
from tests.torch.common_net import DenseNet
from hyper_parallel.platform import get_platform
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    ShardedState,
)
from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    FSDPMeshInfo,
    HSDPMeshInfo,
    DDPMeshInfo,
)
from hyper_parallel.platform.torch.fully_shard.param import (
    TorchHSDPParamV2,
    ParamModuleInfo
)
from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard, Replicate
from hyper_parallel import DTensor, init_device_mesh


platform = get_platform()


def _current_device():
    device_handle = platform.get_device_handle()
    return torch.device(device_handle.current_device())


def _build_hsdp_param(**kwargs):
    """Construct TorchHSDPParamV2 with param_mode defaults derived from test inputs."""
    if "param_mode" not in kwargs:
        param = kwargs.get("param")
        if isinstance(param, DTensor):
            kwargs["param_mode"] = FullyShardParamMode.DTENSOR_UNIFIED
        else:
            kwargs["param_mode"] = FullyShardParamMode.LOCAL_PARAM
    return TorchHSDPParamV2(**kwargs)

def test_hsdp_param_v2_fsdp_1d_mesh():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with 1D FSDP mesh
    Expectation: sharded param shape is correct and state is SHARDED
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP (shard only)
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model and get parameter
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Verify sharded state
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    # Verify sharded param is Parameter
    assert isinstance(hsdp_param.sharded_param, torch.nn.Parameter)

    # Verify sharded size (should be original_size / world_size along dim 0)
    expected_sharded_dim0 = in_channels // world_size
    assert hsdp_param.sharded_size[0] == expected_sharded_dim0
    assert hsdp_param.sharded_size[1] == hidden_size

    # Verify original size is preserved
    assert hsdp_param._orig_size == torch.Size([in_channels, hidden_size])

    print(f"[Rank {rank}] FSDP 1D mesh test passed")
    print(f"  Original size: {hsdp_param._orig_size}")
    print(f"  Sharded size: {hsdp_param.sharded_size}")
    print(f"  Sharded state: {hsdp_param.sharded_state}")


def test_hsdp_param_v2_hsdp_2d_mesh():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with 2D HSDP mesh (replicate + shard)
    Expectation: sharded param shape is correct with 2D mesh
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 2D mesh for HSDP: (replicate_dim, shard_dim)
    # e.g., (2, 4) means 2 replicate groups, each with 4 shards
    replicate_size = 2
    shard_size = world_size // replicate_size
    device_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard")
    )
    mesh_info = HSDPMeshInfo(
        mesh=device_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0
    )

    # Create model and get parameter
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Verify sharded state
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    # Verify sharded size (should be original_size / shard_size along dim 0)
    expected_sharded_dim0 = in_channels // shard_size
    assert hsdp_param.sharded_size[0] == expected_sharded_dim0
    assert hsdp_param.sharded_size[1] == hidden_size

    # Verify HSDP placements
    assert len(hsdp_param._spmd_placements) == 2

    print(f"[Rank {rank}] HSDP 2D mesh test passed")
    print(f"  Mesh shape: ({replicate_size}, {shard_size})")
    print(f"  Original size: {hsdp_param._orig_size}")
    print(f"  Sharded size: {hsdp_param.sharded_size}")


def test_hsdp_param_v2_sharded_state_transitions():
    """
    Feature: TorchHSDPParamV2.
    Description: Test state transitions (sharded -> unsharded -> sharded)
    Expectation: state transitions work correctly
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model and parameter
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Initial state should be SHARDED
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    # Simulate all-gather by initializing all_gather_outputs
    sharded_numel = hsdp_param._sharded_param_data.numel()
    device = hsdp_param._sharded_param_data.device
    dtype = hsdp_param._sharded_param_data.dtype

    hsdp_param.init_all_gather_outputs(
        all_gather_input_numels=[sharded_numel],
        all_gather_input_dtypes=[dtype],
        world_size=world_size,
        device=device,
    )

    # Fill all_gather_outputs with gathered data (simulating all-gather)
    # In real scenario, this would be done by collective communication
    all_gather_output = hsdp_param.all_gather_outputs[0]
    all_gather_output.fill_(1.0)  # Fill with test data

    # Initialize unsharded param
    hsdp_param.init_unsharded_param()

    # Transition to unsharded state
    hsdp_param.to_unsharded()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert hsdp_param.unsharded_param.shape == torch.Size([in_channels, hidden_size])

    # Transition back to sharded state
    hsdp_param.to_sharded()
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    print(f"[Rank {rank}] State transitions test passed")


def test_hsdp_param_v2_custom_shard_placement():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with custom shard placement function
    Expectation: custom shard placement is applied correctly
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model with 2D weight
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")

    # Custom shard placement function - shard along dim 1 instead of default dim 0
    def custom_shard_fn(param):
        return Shard(1)
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2 with custom placement
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        shard_placement_fn=custom_shard_fn,
        device=torch.device(device_handle.current_device()),
    )

    # Verify shard placement
    assert hsdp_param.hsdp_placement.dim == 1

    # Verify sharded size (should shard along dim 1)
    expected_sharded_dim1 = hidden_size // world_size
    assert hsdp_param.sharded_size[0] == in_channels
    assert hsdp_param.sharded_size[1] == expected_sharded_dim1

    print(f"[Rank {rank}] Custom shard placement test passed")
    print(f"  Shard dim: {hsdp_param.hsdp_placement.dim}")
    print(f"  Original size: {hsdp_param._orig_size}")
    print(f"  Sharded size: {hsdp_param.sharded_size}")


def test_hsdp_param_v2_mixed_precision():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with mixed precision policy
    Expectation: dtype attributes are set correctly
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create mesh
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")

    # Create mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
    )
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2 with mixed precision
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        mp_policy=mp_policy,
        device=torch.device(device_handle.current_device()),
    )

    # Initialize dtype attributes
    hsdp_param.init_dtype_attrs(mp_policy)

    # Verify dtype attributes
    assert hsdp_param.param_dtype == torch.float16
    assert hsdp_param.reduce_dtype == torch.float32

    print(f"[Rank {rank}] Mixed precision test passed")
    print(f"  param_dtype: {hsdp_param.param_dtype}")
    print(f"  reduce_dtype: {hsdp_param.reduce_dtype}")


def test_hsdp_param_v2_all_gather_comm():
    """
    Feature: TorchHSDPParamV2.
    Description: Test param-level all-gather communication
    Expectation: all-gather correctly reconstructs full parameter
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model with known values
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    # Initialize weight with rank-based values for verification
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Verify initial state
    assert hsdp_param.sharded_state == ShardedState.SHARDED
    assert hsdp_param.shard_world_size == world_size

    # Execute all-gather
    unsharded_data, handle = hsdp_param._get_unsharded_param_data(async_op=False)

    # Verify output shape
    expected_numel = hsdp_param._sharded_param_data.numel() * world_size
    assert unsharded_data.numel() == expected_numel

    print(f"[Rank {rank}] All-gather comm test passed")
    print(f"  Sharded numel: {hsdp_param._sharded_param_data.numel()}")
    print(f"  Unsharded numel: {unsharded_data.numel()}")


def test_hsdp_param_v2_prefetch_unshard():
    """
    Feature: TorchHSDPParamV2.
    Description: Test async prefetch and unshard workflow
    Expectation: prefetch correctly prepares unsharded parameter
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Test prefetch workflow
    assert hsdp_param.sharded_state == ShardedState.SHARDED
    assert hsdp_param.prefetch_handle is None

    hsdp_param.unshard(async_op=True)
    hsdp_param.wait_for_unshard()

    # Verify state transition
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert hsdp_param.unsharded_param.shape == torch.Size([in_channels, hidden_size])
    assert hsdp_param.prefetch_handle is None  # Should be cleared

    print(f"[Rank {rank}] Prefetch unshard test passed")


def test_hsdp_param_v2_unshard_shard_cycle():
    """
    Feature: TorchHSDPParamV2.
    Description: Test complete unshard -> shard cycle with communication
    Expectation: state transitions and communication work correctly
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Initial state
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    # Unshard
    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert hsdp_param.unsharded_param.shape == torch.Size([in_channels, hidden_size])

    # Shard
    hsdp_param.shard()
    assert hsdp_param.sharded_state == ShardedState.SHARDED

    print(f"[Rank {rank}] Unshard-shard cycle test passed")


def test_hsdp_param_v2_reduce_scatter_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test reduce-scatter gradient communication with deterministic data
    Expectation: gradient is correctly reduced and scattered
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Unshard first
    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED

    # Create deterministic gradient: each rank has gradient = rank
    hsdp_param._unsharded_param.grad = torch.full(
        (in_channels, hidden_size),
        float(rank),
        dtype=hsdp_param._unsharded_param.dtype,
        device=hsdp_param._unsharded_param.device
    )

    # Execute reduce-scatter
    hsdp_param.reduce_scatter_grad(async_op=False, reduce_op=dist.ReduceOp.SUM)
    sharded_grad = hsdp_param.reduce_scatter_output()
    hsdp_param.clear_reduce_scatter_output()
    # Verify output size
    expected_numel = in_channels * hidden_size // world_size
    assert sharded_grad.numel() == expected_numel, \
        f"Expected numel {expected_numel}, got {sharded_grad.numel()}"

    # Calculate expected value after reduce-scatter:
    expected_sum = sum(i for i in range(world_size))
    expected_value = float(expected_sum)

    # Each element in sharded_grad should equal expected_sum
    assert torch.allclose(sharded_grad, torch.full_like(sharded_grad, expected_value)), \
        f"Expected all values to be {expected_value}, got {sharded_grad.flatten()[:5].tolist()}..."


def test_hsdp_param_v2_all_reduce_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test all-reduce gradient communication in HSDP mode
    Expectation: gradient is correctly all-reduced across replicate dimension
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 2D mesh for HSDP: (replicate_dim, shard_dim)
    replicate_size = 2
    shard_size = world_size // replicate_size
    device_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard")
    )
    mesh_info = HSDPMeshInfo(
        mesh=device_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0
    )

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Unshard first
    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED

    # Create a gradient with rank-specific values
    # Each rank's gradient = rank
    hsdp_param._unsharded_param.grad = torch.full(
        (in_channels, hidden_size),
        float(rank),
        dtype=hsdp_param._unsharded_param.dtype,
        device=hsdp_param._unsharded_param.device
    )

    # Execute all-reduce on gradient
    grad = hsdp_param._unsharded_param.grad.clone()
    hsdp_param.all_reduce_grad(grad=grad, async_op=False, reduce_op=dist.ReduceOp.SUM)
    reduced_grad = hsdp_param.all_reduce_output()
    hsdp_param.clear_all_reduce_output()
    # Calculate sum of ranks in the same replicate group
    # Ranks in same replicate group share the same shard_idx
    expected_sum = 0.0
    shard_idx = rank % shard_size
    for rep_idx in range(replicate_size):
        group_rank = rep_idx * shard_size + shard_idx
        expected_sum += float(group_rank)  # gradient value = rank

    assert torch.allclose(reduced_grad, torch.full_like(reduced_grad, expected_sum)), \
        f"Rank {rank}: Expected all values to be {expected_sum}, got {reduced_grad.flatten()[:5].tolist()}..."


def test_hsdp_param_v2_accumulate_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test gradient accumulation workflow with deterministic data
    Expectation: gradients are correctly accumulated across iterations
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    # Create 1D mesh for FSDP
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=device_mesh, shard_mesh_dim=0)

    # Create model
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")
    device_handle = platform.get_device_handle()
    # Create TorchHSDPParamV2
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=torch.device(device_handle.current_device()),
    )

    # Unshard
    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    # Iteration 1: Create gradient with rank-specific value
    # Gradient 1: all elements = rank
    hsdp_param._unsharded_param.grad = torch.full(
        (in_channels, hidden_size),
        float(rank),
        dtype=hsdp_param._unsharded_param.dtype,
        device=hsdp_param._unsharded_param.device
    )
    hsdp_param.unsharded_accumulated_grad = hsdp_param._unsharded_param.grad.clone()
    hsdp_param._unsharded_param.grad = None

    # Iteration 2: Create another gradient and accumulate
    # Gradient 2: all elements = rank + 1
    hsdp_param._unsharded_param.grad = torch.full(
        (in_channels, hidden_size),
        float(rank + 1),
        dtype=hsdp_param._unsharded_param.dtype,
        device=hsdp_param._unsharded_param.device
    )
    hsdp_param.accumulate_unsharded_grad_if_needed()

    # Verify accumulated grad has sum of both iterations
    expected_accumulated = float(rank) + float(rank + 1)
    assert torch.allclose(
        hsdp_param.unsharded_accumulated_grad,
        torch.full_like(hsdp_param.unsharded_accumulated_grad, expected_accumulated)
    ), f"Rank {rank}: Expected accumulated {expected_accumulated}, got {hsdp_param.unsharded_accumulated_grad.flatten()[0].item()}"

    assert hsdp_param._unsharded_param.grad is None, "Gradient should be cleared after accumulation"

    hsdp_param.reduce_scatter_grad(async_op=False, reduce_op=dist.ReduceOp.SUM)
    sharded_grad = hsdp_param.reduce_scatter_output()
    hsdp_param.clear_reduce_scatter_output()
    # Calculate expected reduced value:
    # Each rank's accumulated grad = rank + (rank + 1) = 2 * rank + 1
    expected_reduced_value = sum(2 * i + 1 for i in range(world_size))

    assert torch.allclose(sharded_grad, torch.full_like(sharded_grad, expected_reduced_value)), \
        f"Rank {rank}: Expected reduced {expected_reduced_value}, got {sharded_grad.flatten()[0].item()}"


def test_hsdp_param_v2_dtensor_dp_tp_preserve_tp_layout():
    """
    Feature: TorchHSDPParamV2.
    Description: Test DTensor param on 2D dp x tp mesh with fully_shard enabled.
    Expectation: fully_shard only adds DP sharding and preserves TP placement after unshard.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    tp_size = 2
    assert world_size % tp_size == 0, "world_size should be divisible by tp_size"
    dp_size = world_size // tp_size

    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = root_mesh["dp"]
    tp_mesh = root_mesh["tp"]
    mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels, hidden_size // tp_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_mesh, (Shard(1),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert hsdp_param.sharded_state == ShardedState.SHARDED
    assert hsdp_param.sharded_size == torch.Size((in_channels // dp_size, hidden_size // tp_size))
    assert hsdp_param._spmd_placements[0] == Shard(0)
    assert hsdp_param._spmd_placements[1] == Shard(1)

    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert isinstance(hsdp_param.unsharded_param, DTensor)
    assert tuple(hsdp_param.unsharded_param.placements) == (Shard(1),)
    assert hsdp_param.unsharded_param.to_local().shape == torch.Size((in_channels, hidden_size // tp_size))

    print(f"[Rank {rank}] 2D dp x tp DTensor preserve TP layout test passed")


def test_hsdp_param_v2_dtensor_dp_tp_same_dim_uses_strided_shard():
    """
    Feature: TorchHSDPParamV2.
    Description: Test DTensor param on 2D dp x tp mesh when TP and fully_shard split the same tensor dim.
    Expectation: fully_shard uses StridedShard and layout tensor_map preserves the split order.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    tp_size = 2
    assert world_size % tp_size == 0, "world_size should be divisible by tp_size"
    dp_size = world_size // tp_size

    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = root_mesh["dp"]
    tp_mesh = root_mesh["tp"]
    mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels // tp_size, hidden_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_mesh, (Shard(0),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert hsdp_param.sharded_state == ShardedState.SHARDED
    assert hsdp_param.sharded_size == torch.Size((in_channels // (dp_size * tp_size), hidden_size))
    assert tuple(hsdp_param._spmd_placements) == (StridedShard(0, tp_size), Shard(0))
    assert hsdp_param._sharding_spec.tensor_map == ((0, 1), -1)

    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert isinstance(hsdp_param.unsharded_param, DTensor)
    assert tuple(hsdp_param.unsharded_param.placements) == (Shard(0),)
    assert hsdp_param.unsharded_param.to_local().shape == torch.Size((in_channels // tp_size, hidden_size))

    print(f"[Rank {rank}] 2D dp x tp same-dim StridedShard test passed")


def test_hsdp_param_v2_dtensor_dp_tp_ep_unshard_only_fsdp_dim():
    """
    Feature: TorchHSDPParamV2.
    Description: Test DTensor param on 3D dp x tp x ep mesh.
    Expectation: unshard only restores the DP/FSDP dim and keeps TP/EP placements unchanged.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    dp_size = 2
    tp_size = 2
    assert world_size % (dp_size * tp_size) == 0, "world_size should be divisible by dp_size * tp_size"
    ep_size = world_size // (dp_size * tp_size)

    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size, ep_size),
        mesh_dim_names=("dp", "tp", "ep"),
    )
    dp_mesh = root_mesh["dp"]
    tp_ep_mesh = root_mesh[("tp", "ep")]
    mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels, hidden_size // tp_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_ep_mesh, (Shard(1), Replicate()))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert hsdp_param.sharded_state == ShardedState.SHARDED
    assert hsdp_param.sharded_size == torch.Size((in_channels // dp_size, hidden_size // tp_size))
    assert tuple(hsdp_param._spmd_placements) == (Shard(0), Shard(1), Replicate())

    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert isinstance(hsdp_param.unsharded_param, DTensor)
    assert tuple(hsdp_param.unsharded_param.placements) == (Shard(1), Replicate())
    assert hsdp_param.unsharded_param.to_local().shape == torch.Size((in_channels, hidden_size // tp_size))

    print(f"[Rank {rank}] 3D dp x tp x ep DTensor unshard test passed")


def test_hsdp_param_v2_pure_tp_no_param_shard_all_reduce():
    """
    Feature: TorchHSDPParamV2.
    Description: Test the DTensor compatibility mode without extra fully_shard parameter sharding.
    Expectation: TP-replicated parameters are synchronized by all-reduce only.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("tp",),
    )
    mesh_info = DDPMeshInfo(mesh=mesh, replicate_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels, hidden_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, mesh, (Replicate(),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_COMPAT,
    )

    assert hsdp_param.is_sharded is False
    assert hsdp_param.shard_size == 1
    assert hsdp_param.dp_size == world_size
    assert hsdp_param.unsharded_group_info.rank_size == world_size

    grad = torch.full_like(local_weight, float(rank))
    hsdp_param.all_reduce_grad(grad=grad, async_op=False, reduce_op=dist.ReduceOp.SUM)
    reduced_grad = hsdp_param.all_reduce_output()
    hsdp_param.clear_all_reduce_output()

    expected_value = float(sum(range(world_size)))
    assert torch.allclose(
        reduced_grad,
        torch.full_like(reduced_grad, expected_value),
    ), f"Rank {rank}: expected {expected_value}, got {reduced_grad.flatten()[0].item()}"

    print(f"[Rank {rank}] Pure TP no-param-shard all-reduce test passed")


def test_hsdp_param_v2_pure_tp_sharded_param_skips_all_reduce():
    """
    Feature: TorchHSDPParamV2.
    Description: Test DTensor compatibility mode for already sharded distributed parameters.
    Expectation: TP-sharded parameters do not create an all-reduce group in the compatibility path.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("tp",),
    )
    mesh_info = DDPMeshInfo(mesh=mesh, replicate_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels, hidden_size // world_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, mesh, (Shard(1),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_COMPAT,
    )

    assert hsdp_param.is_sharded is False
    assert hsdp_param.shard_size == 1
    assert hsdp_param.dp_size == 1
    assert hsdp_param.unsharded_group_info.rank_size == 1

    grad = torch.full_like(local_weight, float(rank))
    reduced_grad, handle = hsdp_param.all_reduce_grad(grad=grad, async_op=False, reduce_op=dist.ReduceOp.SUM)

    assert handle is None
    assert torch.allclose(reduced_grad, grad)
    print(f"[Rank {rank}] Pure TP sharded parameter skip all-reduce test passed")


def test_hsdp_param_v2_explicit_dp_mesh_prefixes_unified_layout():
    """
    Feature: TorchHSDPParamV2.
    Description: Test unified mesh uses explicit DP mesh prefix followed by TP mesh.
    Expectation: DP replicate group size stays correct and StridedShard is applied on the FSDP axis.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    if world_size % 4 != 0:
        print(f"[Rank {rank}] Skip explicit DP mesh prefix test because world_size={world_size} is not divisible by 4")
        return

    tp_size = world_size // 4
    dp_size = 2
    fsdp_size = 2
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, fsdp_size, tp_size),
        mesh_dim_names=("dp", "fsdp", "tp"),
    )
    dp_fsdp_mesh = root_mesh[("dp", "fsdp")]
    tp_mesh = root_mesh["tp"]
    mesh_info = HSDPMeshInfo(mesh=dp_fsdp_mesh, shard_mesh_dim=1, replicate_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels // max(tp_size, 1), hidden_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_mesh, (Shard(0),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")
    hsdp_param = TorchHSDPParamV2(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert hsdp_param._spmd_mesh.mesh_dim_names == ("dp", "fsdp", "tp")
    assert hsdp_param._spmd_replicate_mesh_dim == 0
    assert hsdp_param._spmd_shard_mesh_dim == 1
    assert hsdp_param.unsharded_group_info.rank_size == dp_size
    assert hsdp_param.dp_size == dp_size

    if tp_size > 1:
        assert hsdp_param._spmd_placements[0] == Replicate()
        assert hsdp_param._spmd_placements[1] == StridedShard(0, tp_size)
        assert hsdp_param._spmd_placements[2] == Shard(0)
    print(f"[Rank {rank}] Explicit DP-prefix unified mesh/group-info test passed")


def test_hsdp_param_v2_reordered_mesh_remaps_dp_dims_for_dtensor():
    """
    Feature: TorchHSDPParamV2.
    Description: Test DTensor unified layout keeps the explicit DP/FSDP dims on the unified mesh.
    Expectation: shard/replicate mesh dims and unsharded group construction stay correct.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    if world_size % 4 != 0:
        print(f"[Rank {rank}] Skip reordered mesh remap test because world_size={world_size} is not divisible by 4")
        return

    tp_size = world_size // 4
    dp_size = 2
    fsdp_size = 2
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, fsdp_size, tp_size),
        mesh_dim_names=("dp", "fsdp", "tp"),
    )
    dp_fsdp_mesh = root_mesh[("dp", "fsdp")]
    tp_mesh = root_mesh["tp"]
    mesh_info = HSDPMeshInfo(mesh=dp_fsdp_mesh, shard_mesh_dim=1, replicate_mesh_dim=0)

    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels // max(tp_size, 1), hidden_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_mesh, (Shard(0),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")
    hsdp_param = TorchHSDPParamV2(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert hsdp_param._spmd_mesh.mesh_dim_names == ("dp", "fsdp", "tp")
    assert hsdp_param._spmd_replicate_mesh_dim == 0
    assert hsdp_param._spmd_shard_mesh_dim == 1
    assert hsdp_param.sharded_group_info.rank_size == fsdp_size
    assert hsdp_param.unsharded_group_info.rank_size == dp_size
    assert hsdp_param.dp_size == dp_size
    assert hsdp_param.shard_size == fsdp_size

    if tp_size > 1:
        assert hsdp_param._spmd_placements[0] == Replicate()
        assert hsdp_param._spmd_placements[1] == StridedShard(0, tp_size)
        assert hsdp_param._spmd_placements[2] == Shard(0)

    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    assert hsdp_param.sharded_state == ShardedState.UNSHARDED
    assert isinstance(hsdp_param.unsharded_param, DTensor)
    assert tuple(hsdp_param.unsharded_param.placements) == (Shard(0),)

    print(f"[Rank {rank}] Reordered mesh remap DTensor test passed")


def test_hsdp_param_v2_reduce_scatter_guard_non_dim0():
    """
    Feature: TorchHSDPParamV2.
    Description: Test reduce_scatter_grad rejects non-dim0 fully_shard placement.
    Expectation: reduce_scatter_grad raises NotImplementedError.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    if world_size <= 1:
        print(f"[Rank {rank}] Skip non-dim0 RS guard test because world_size={world_size}")
        return

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mesh_info = FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
    in_channels, hidden_size = 16, world_size * 8
    net = DenseNet(in_channels, hidden_size)
    module_info = ParamModuleInfo(module=net, param_name="weight")

    def custom_shard_fn(param):  # pylint: disable=unused-argument
        return Shard(1)

    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        shard_placement_fn=custom_shard_fn,
        device=_current_device(),
        param_mode=FullyShardParamMode.LOCAL_PARAM,
    )

    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    hsdp_param.unsharded_param.grad = torch.ones(
        (in_channels, hidden_size), dtype=hsdp_param.unsharded_param.dtype, device=_current_device()
    )
    try:
        hsdp_param.reduce_scatter_grad(async_op=False, reduce_op=dist.ReduceOp.SUM)
    except NotImplementedError as exc:
        assert "dim=0" in str(exc)
    else:
        raise AssertionError("Expected reduce_scatter_grad to reject non-dim0 fully_shard placement")
    print(f"[Rank {rank}] Non-dim0 RS guard test passed")


def test_hsdp_param_v2_reduce_scatter_guard_strided_shard():
    """
    Feature: TorchHSDPParamV2.
    Description: Test reduce_scatter_grad rejects StridedShard layout until placement-aware packing is implemented.
    Expectation: reduce_scatter_grad raises NotImplementedError.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        print(f"[Rank {rank}] Skip StridedShard RS guard test because world_size={world_size}")
        return

    tp_size = 2
    dp_size = world_size // tp_size
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = root_mesh["dp"]
    tp_mesh = root_mesh["tp"]
    mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)
    in_channels, hidden_size = 32, 64
    net = DenseNet(in_channels, hidden_size)
    local_weight = torch.full((in_channels // tp_size, hidden_size), float(rank), device="npu")
    net.weight = torch.nn.Parameter(
        DTensor.from_local(local_weight, tp_mesh, (Shard(0),))
    )
    module_info = ParamModuleInfo(module=net, param_name="weight")
    hsdp_param = _build_hsdp_param(
        param=net.weight,
        module_info=module_info,
        mesh_info=mesh_info,
        device=_current_device(),
        param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
    )

    assert any(isinstance(placement, StridedShard) for placement in hsdp_param._spmd_placements)
    hsdp_param.unshard()
    hsdp_param.wait_for_unshard()
    hsdp_param.unsharded_accumulated_grad = torch.ones_like(hsdp_param.unsharded_param.to_local())
    try:
        hsdp_param.reduce_scatter_grad(async_op=False, reduce_op=dist.ReduceOp.SUM)
    except NotImplementedError as exc:
        assert "StridedShard" in str(exc)
    else:
        raise AssertionError("Expected reduce_scatter_grad to reject StridedShard layout")
    print(f"[Rank {rank}] StridedShard RS guard test passed")
