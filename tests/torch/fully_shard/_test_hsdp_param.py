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
from hyper_parallel.platform.torch.fully_shard.utils import (
    FSDPMeshInfo,
    HSDPMeshInfo,
    MixedPrecisionPolicy,
)
from hyper_parallel.platform.torch.fully_shard.param import (
    TorchHSDPParamV2,
    ParamModuleInfo
)
from hyper_parallel.core.placement_types import Shard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState
from hyper_parallel import init_device_mesh

platform = get_platform()

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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
    hsdp_param = TorchHSDPParamV2(
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
