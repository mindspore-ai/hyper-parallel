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
"""DCP save and load integration tests."""
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# pylint: disable=W0611
from hyper_parallel import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.core.distributed_checkpoint import async_save, load, save
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.topology_mapper import TopologyMapper
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


def _run_dcp_save_load_test(
    mesh_shape: tuple[int, int],
    param_configs: list[dict[str, Any]],
    checkpoint_path: Path,
    seed: int = 1,
    scalar_values: Optional[dict[str, Any]] = None,
    tensor_values: Optional[dict[str, tuple[int, ...]]] = None,
    use_collectives: bool = True,
) -> None:
    """
    Common function to test checkpoint save and load API with DTensor state_dict, scalar and torch Tensor values.

    Args:
        mesh_shape (tuple[int, int]): Tuple of (dp_size, tp_size) for device mesh shape.
        param_configs (list[dict[str, Any]]): List of parameter configurations, each containing:
            - 'name': parameter name (str)
            - 'placements': list of Placement objects, e.g., [Replicate(), Shard(1)]
            - 'local_shape': tuple of local tensor shape
        checkpoint_path (Path): Path to checkpoint directory.
        seed (int): Random seed for reproducibility. Default 1.
        scalar_values (Optional[dict[str, Any]]): Optional dict of scalar values to save/load
            (e.g., {'epoch': 10, 'lr': 0.001}). Default None.
        tensor_values (Optional[dict[str, tuple[int, ...]]]): Optional dict of torch Tensor configs,
            key is name, value is shape tuple (e.g., {'buffer': (10, 8)}). Default None.
        use_collectives (bool): If True, use collective communication for save/load coordination. Default True.
    """
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    # Create DeviceMesh
    alias_name = ("dp", "tp")
    device_mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=mesh_shape, mesh_dim_names=alias_name)

    # Create DTensors based on configurations
    state_dict = {}
    original_local_tensors = {}
    original_scalars = {}

    for param_config in param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        local_tensor = to_device(torch.randn(*local_shape), _DEVICE_TYPE)
        dtensor = DTensor.from_local(local_tensor, device_mesh, placements)
        state_dict[param_name] = dtensor
        original_local_tensors[param_name] = dtensor.to_local().clone()

    # Add scalar values to state_dict
    if scalar_values:
        for scalar_name, scalar_value in scalar_values.items():
            state_dict[scalar_name] = scalar_value
            original_scalars[scalar_name] = scalar_value

    # Add torch Tensor values to state_dict
    original_tensors: dict[str, Any] = {}
    if tensor_values:
        for tensor_name, tensor_shape in tensor_values.items():
            tensor = to_device(torch.randn(*tensor_shape), _DEVICE_TYPE)
            state_dict[tensor_name] = tensor
            original_tensors[tensor_name] = tensor.clone()

    # Call save API
    metadata = save(state_dict, checkpoint_id=checkpoint_path, use_collectives=use_collectives)
    print("metadata: ", metadata)

    # Verify save results
    assert metadata is not None
    assert hasattr(metadata, 'state_dict_metadata')
    for param_config in param_configs:
        assert param_config['name'] in metadata.state_dict_metadata
    if scalar_values:
        for scalar_name in scalar_values.keys():
            assert scalar_name in metadata.state_dict_metadata
    if tensor_values:
        for tensor_name in tensor_values.keys():
            assert tensor_name in metadata.state_dict_metadata

    # Verify checkpoint directory exists
    assert checkpoint_path.exists()

    # Create new state_dict for load with same structure but empty/zero DTensors
    load_state_dict = {}
    for param_config in param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        load_local_tensor = to_device(torch.zeros(*local_shape), _DEVICE_TYPE)
        load_dtensor = DTensor.from_local(load_local_tensor, device_mesh, placements)
        load_state_dict[param_name] = load_dtensor

    # Initialize scalar values in load_state_dict (will be overwritten by load)
    if scalar_values:
        for scalar_name in scalar_values.keys():
            # Initialize with different values to verify they get loaded correctly
            if isinstance(scalar_values[scalar_name], int):
                load_state_dict[scalar_name] = 0
            elif isinstance(scalar_values[scalar_name], float):
                load_state_dict[scalar_name] = 0.0
            else:
                load_state_dict[scalar_name] = None

    # Initialize torch Tensor values in load_state_dict (will be overwritten by load)
    if tensor_values:
        for tensor_name, tensor_shape in tensor_values.items():
            load_state_dict[tensor_name] = to_device(torch.zeros(*tensor_shape), _DEVICE_TYPE)

    # Call load API
    load(load_state_dict, checkpoint_id=checkpoint_path, use_collectives=use_collectives)

    # Verify load results - compare loaded tensors with original tensors
    for param_config in param_configs:
        param_name = param_config['name']
        loaded_local_tensor = load_state_dict[param_name].to_local()
        original_local_tensor = original_local_tensors[param_name]

        assert np.allclose(
            original_local_tensor.cpu().detach().numpy(),
            loaded_local_tensor.cpu().detach().numpy(),
            rtol=1e-5, atol=1e-5
        ), f"{param_name} values do not match after load"

    # Verify load results - compare loaded scalars with original scalars
    if scalar_values:
        for scalar_name, original_value in original_scalars.items():
            loaded_value = load_state_dict[scalar_name]
            assert loaded_value == original_value, \
                f"{scalar_name} scalar value mismatch: expected {original_value}, got {loaded_value}"
            assert type(loaded_value) == type(original_value), \
                f"{scalar_name} scalar type mismatch: expected {type(original_value)}, got {type(loaded_value)}"

    # Verify load results - compare loaded torch Tensors with original tensors
    if tensor_values:
        for tensor_name, original_tensor in original_tensors.items():
            loaded_tensor = load_state_dict[tensor_name]
            assert np.allclose(
                original_tensor.cpu().detach().numpy(),
                loaded_tensor.cpu().detach().numpy(),
                rtol=1e-5, atol=1e-5
            ), f"{tensor_name} torch Tensor values do not match after load"


def _run_dcp_async_save_load_test(
    mesh_shape: tuple[int, int],
    param_configs: list[dict[str, Any]],
    checkpoint_path: Path,
    seed: int = 1,
    scalar_values: Optional[dict[str, Any]] = None,
    tensor_values: Optional[dict[str, tuple[int, ...]]] = None,
    use_collectives: bool = True,
) -> None:
    """
    Same as :func:`_run_dcp_save_load_test` but persists via :func:`async_save` and ``persist_completion``.
    """
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    alias_name = ("dp", "tp")
    device_mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=mesh_shape, mesh_dim_names=alias_name)

    state_dict = {}
    original_local_tensors = {}
    original_scalars = {}

    for param_config in param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        local_tensor = to_device(torch.randn(*local_shape), _DEVICE_TYPE)
        dtensor = DTensor.from_local(local_tensor, device_mesh, placements)
        state_dict[param_name] = dtensor
        original_local_tensors[param_name] = dtensor.to_local().clone()

    if scalar_values:
        for scalar_name, scalar_value in scalar_values.items():
            state_dict[scalar_name] = scalar_value
            original_scalars[scalar_name] = scalar_value

    original_tensors: dict[str, Any] = {}
    if tensor_values:
        for tensor_name, tensor_shape in tensor_values.items():
            tensor = to_device(torch.randn(*tensor_shape), _DEVICE_TYPE)
            state_dict[tensor_name] = tensor
            original_tensors[tensor_name] = tensor.clone()

    async_resp = async_save(state_dict, checkpoint_id=checkpoint_path, use_collectives=False)
    metadata = async_resp.persist_completion.result()
    assert isinstance(metadata, Metadata)
    print("metadata: ", metadata)

    assert metadata is not None
    assert hasattr(metadata, 'state_dict_metadata')
    for param_config in param_configs:
        assert param_config['name'] in metadata.state_dict_metadata
    if scalar_values:
        for scalar_name in scalar_values.keys():
            assert scalar_name in metadata.state_dict_metadata
    if tensor_values:
        for tensor_name in tensor_values.keys():
            assert tensor_name in metadata.state_dict_metadata

    assert checkpoint_path.exists()

    load_state_dict = {}
    for param_config in param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        load_local_tensor = to_device(torch.zeros(*local_shape), _DEVICE_TYPE)
        load_dtensor = DTensor.from_local(load_local_tensor, device_mesh, placements)
        load_state_dict[param_name] = load_dtensor

    if scalar_values:
        for scalar_name in scalar_values.keys():
            if isinstance(scalar_values[scalar_name], int):
                load_state_dict[scalar_name] = 0
            elif isinstance(scalar_values[scalar_name], float):
                load_state_dict[scalar_name] = 0.0
            else:
                load_state_dict[scalar_name] = None

    if tensor_values:
        for tensor_name, tensor_shape in tensor_values.items():
            load_state_dict[tensor_name] = to_device(torch.zeros(*tensor_shape), _DEVICE_TYPE)

    load(load_state_dict, checkpoint_id=checkpoint_path, use_collectives=use_collectives)

    for param_config in param_configs:
        param_name = param_config['name']
        loaded_local_tensor = load_state_dict[param_name].to_local()
        original_local_tensor = original_local_tensors[param_name]

        assert np.allclose(
            original_local_tensor.cpu().detach().numpy(),
            loaded_local_tensor.cpu().detach().numpy(),
            rtol=1e-5, atol=1e-5
        ), f"{param_name} values do not match after load"

    if scalar_values:
        for scalar_name, original_value in original_scalars.items():
            loaded_value = load_state_dict[scalar_name]
            assert loaded_value == original_value, (
                f"{scalar_name} scalar value mismatch: expected {original_value}, got {loaded_value}"
            )
            assert type(loaded_value) == type(original_value), (
                f"{scalar_name} scalar type mismatch: expected {type(original_value)}, got {type(loaded_value)}"
            )

    if tensor_values:
        for tensor_name, original_tensor in original_tensors.items():
            loaded_tensor = load_state_dict[tensor_name]
            assert np.allclose(
                original_tensor.cpu().detach().numpy(),
                loaded_tensor.cpu().detach().numpy(),
                rtol=1e-5, atol=1e-5
            ), f"{tensor_name} torch Tensor values do not match after load"


def test_dcp_async_save_and_load_with_dtensor_and_tensor_and_scalar() -> None:
    """
    Feature: ``async_save`` + ``load`` with DTensor, scalars, and dense tensors (same layout as group1 sync test).

    Description: Same mesh and state_dict shapes as ``test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar``
        (collectives path only); persistence uses :func:`async_save`.
    Expectation: Run success; loaded values match originals.
    """
    checkpoint_path = Path("./test_dcp_save_and_load_async")
    mesh_shape = (2, 2)
    param_configs = [
        {'name': 'param1', 'placements': [Shard(0), Replicate()], 'local_shape': (8, 8)},
        {'name': 'param2', 'placements': [Shard(0), Shard(1)], 'local_shape': (6, 4)},
        {'name': 'param3', 'placements': [Replicate(), Replicate()], 'local_shape': (6, 6)},
        {'name': 'param4', 'placements': [Replicate(), Shard(1)], 'local_shape': (8, 4)},
        {'name': 'param5', 'placements': [Shard(0), Replicate()], 'local_shape': (10, 10)},
        {'name': 'param6', 'placements': [Shard(0), Shard(1)], 'local_shape': (8, 6)},
        {'name': 'param7', 'placements': [Replicate(), Replicate()], 'local_shape': (10, 10)},
        {'name': 'param8', 'placements': [Replicate(), Shard(1)], 'local_shape': (14, 3)},
        {'name': 'param9', 'placements': [Shard(0), Replicate()], 'local_shape': (12, 8)},
        {'name': 'param10', 'placements': [Shard(0), Shard(1)], 'local_shape': (4, 5)},
    ]
    scalar_values = {
        'epoch': 25,
        'learning_rate': 0.0005,
        'step': 5000,
        'best_loss': 0.05678,
        'warmup_steps': 100,
    }
    tensor_values = {
        'buffer': (16, 8),
        'position_ids': (1, 32),
    }
    _run_dcp_async_save_load_test(
        mesh_shape=mesh_shape,
        param_configs=param_configs,
        checkpoint_path=checkpoint_path,
        seed=2,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
    )

    platform_obj = get_platform()
    platform_obj.barrier()
    if platform_obj.get_rank() == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar() -> None:
    """
    Feature: Test checkpoint save and load API with DTensor state_dict using different mesh_shape and layouts.
    Description: Test save and load function with state_dict containing DTensors on 4-card setup with mesh_shape (2, 2).
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values.
    """
    checkpoint_path = Path("./test_dcp_save_and_load")
    mesh_shape = (2, 2)  # Different mesh_shape: 2 data parallel, 2 tensor parallel

    # Parameter configurations:
    # Note: placements order is [dp_mesh_dim, tp_mesh_dim]
    # param1: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (16, 8), Local shape: (8, 8) per rank (dp dimension has 2 devices)
    # param2: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (12, 8), Local shape: (6, 4) per rank (dp has 2 devices, tp has 2 devices)
    # param3: replicated (all dimensions replicated)
    #   Global shape: (6, 6), Local shape: (6, 6) per rank
    # param4: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (8, 8), Local shape: (8, 4) per rank (tp dimension has 2 devices)
    # param5: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (20, 10), Local shape: (10, 10) per rank (dp dimension has 2 devices)
    # param6: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (16, 12), Local shape: (8, 6) per rank (dp has 2 devices, tp has 2 devices)
    # param7: replicated (all dimensions replicated)
    #   Global shape: (10, 10), Local shape: (10, 10) per rank
    # param8: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (14, 6), Local shape: (14, 3) per rank (tp dimension has 2 devices)
    # param9: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (24, 8), Local shape: (12, 8) per rank (dp dimension has 2 devices)
    # param10: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (8, 10), Local shape: (4, 5) per rank (dp has 2 devices, tp has 2 devices)
    param_configs = [
        {'name': 'param1', 'placements': [Shard(0), Replicate()], 'local_shape': (8, 8)},
        {'name': 'param2', 'placements': [Shard(0), Shard(1)], 'local_shape': (6, 4)},
        {'name': 'param3', 'placements': [Replicate(), Replicate()], 'local_shape': (6, 6)},
        {'name': 'param4', 'placements': [Replicate(), Shard(1)], 'local_shape': (8, 4)},
        {'name': 'param5', 'placements': [Shard(0), Replicate()], 'local_shape': (10, 10)},
        {'name': 'param6', 'placements': [Shard(0), Shard(1)], 'local_shape': (8, 6)},
        {'name': 'param7', 'placements': [Replicate(), Replicate()], 'local_shape': (10, 10)},
        {'name': 'param8', 'placements': [Replicate(), Shard(1)], 'local_shape': (14, 3)},
        {'name': 'param9', 'placements': [Shard(0), Replicate()], 'local_shape': (12, 8)},
        {'name': 'param10', 'placements': [Shard(0), Shard(1)], 'local_shape': (4, 5)},
    ]

    # Add scalar values (int and float) to test mixed DTensor and scalar save/load
    scalar_values = {
        'epoch': 25,
        'learning_rate': 0.0005,
        'step': 5000,
        'best_loss': 0.05678,
        'warmup_steps': 100
    }

    # Add torch Tensor values to test mixed DTensor, scalar and torch Tensor save/load
    tensor_values = {
        'buffer': (16, 8),
        'position_ids': (1, 32),
    }

    _run_dcp_save_load_test(
        mesh_shape=mesh_shape,
        param_configs=param_configs,
        checkpoint_path=checkpoint_path,
        seed=2,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
    )

    # Scenario: save/load without collective communication
    checkpoint_path_no_coll = Path("./test_dcp_save_and_load_no_collectives")
    _run_dcp_save_load_test(
        mesh_shape=mesh_shape,
        param_configs=param_configs,
        checkpoint_path=checkpoint_path_no_coll,
        seed=3,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
        use_collectives=False,
    )

    platform_obj = get_platform()
    platform_obj.barrier()
    if platform_obj.get_rank() == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        shutil.rmtree(checkpoint_path_no_coll, ignore_errors=True)


def test_dcp_save_and_load_with_full_tensor() -> None:
    """
    Feature: Test checkpoint save and load API with state_dict containing only torch Tensors.
    Description: Test save and load function with state_dict containing purely torch Tensors
                 (no DTensor, no scalars) on 8-card setup.
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values.
    """
    checkpoint_path = Path("./test_dcp_full_tensor")
    mesh_shape = (1, 2)

    # Only torch Tensors - no DTensor, no scalars
    tensor_values = {
        'tensor1': (8, 8),
        'tensor2': (16, 4),
        'tensor3': (1, 128),
        'buffer': (32, 64),
    }

    _run_dcp_save_load_test(
        mesh_shape=mesh_shape,
        param_configs=[],
        checkpoint_path=checkpoint_path,
        seed=42,
        scalar_values=None,
        tensor_values=tensor_values,
        use_collectives=False,
    )

    platform_obj = get_platform()
    platform_obj.barrier()
    if platform_obj.get_rank() == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def _run_dcp_save_load_with_different_mesh_test(
    save_mesh_shape: tuple[int, int],
    load_mesh_shape: tuple[int, int],
    save_param_configs: list[dict[str, Any]],
    load_param_configs: list[dict[str, Any]],
    checkpoint_path: Path,
    seed: int = 1,
    scalar_values: Optional[dict[str, Any]] = None,
) -> None:
    """
    Test checkpoint save with one mesh_shape and load with different mesh_shape.

    Args:
        save_mesh_shape (tuple[int, int]): Tuple of (dp_size, tp_size) for device mesh shape during save.
        load_mesh_shape (tuple[int, int]): Tuple of (dp_size, tp_size) for device mesh shape during load.
        save_param_configs (list[dict[str, Any]]): List of parameter configurations for save, each containing:
            - 'name': parameter name (str)
            - 'placements': list of Placement objects, e.g., [Replicate(), Shard(1)]
            - 'local_shape': tuple of local tensor shape
        load_param_configs (list[dict[str, Any]]): List of parameter configurations for load, each containing:
            - 'name': parameter name (str)
            - 'placements': list of Placement objects, e.g., [Replicate(), Shard(1)]
            - 'local_shape': tuple of local tensor shape
        checkpoint_path (Path): Path to checkpoint directory.
        seed (int): Random seed for reproducibility. Default 1.
        scalar_values (Optional[dict[str, Any]]): Optional dict of scalar values to save/load
            (e.g., {'epoch': 10, 'lr': 0.001}). Default None.
    """
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    # ========== SAVE PHASE: Use save_mesh_shape ==========
    # Create DeviceMesh for save
    alias_name = ("dp", "tp")
    save_device_mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=save_mesh_shape, mesh_dim_names=alias_name)

    # Create DTensors based on save configurations
    save_state_dict = {}
    original_global_tensors = {}
    original_scalars = {}

    for param_config in save_param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        local_tensor = to_device(torch.randn(*local_shape), _DEVICE_TYPE)
        dtensor = DTensor.from_local(local_tensor, save_device_mesh, placements)
        save_state_dict[param_name] = dtensor
        # Store global tensor for verification (gather from all ranks)
        # Use full_tensor() to get the complete global tensor for comparison
        original_global_tensors[param_name] = dtensor.full_tensor().clone()

    # Add scalar values to state_dict
    if scalar_values:
        for scalar_name, scalar_value in scalar_values.items():
            save_state_dict[scalar_name] = scalar_value
            original_scalars[scalar_name] = scalar_value

    # Call save API
    metadata = save(save_state_dict, checkpoint_id=checkpoint_path)

    # Verify save results
    assert metadata is not None
    assert hasattr(metadata, 'state_dict_metadata')
    for param_config in save_param_configs:
        assert param_config['name'] in metadata.state_dict_metadata
    if scalar_values:
        for scalar_name in scalar_values.keys():
            assert scalar_name in metadata.state_dict_metadata

    # Verify checkpoint directory exists
    assert checkpoint_path.exists()

    # ========== LOAD PHASE: Use load_mesh_shape ==========
    # Get current rank and world_size
    platform = get_platform()
    current_rank = platform.get_rank()


    # Calculate load mesh size
    load_mesh_size = load_mesh_shape[0] * load_mesh_shape[1]

    # Only participate in load if current rank is within the load mesh
    # For 8-card save -> 4-card load, only ranks 0-3 participate in load
    if current_rank < load_mesh_size:
        # Create DeviceMesh for load using only the first load_mesh_size ranks
        load_rank_list = tuple(range(load_mesh_size))
        load_device_mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=load_mesh_shape,
                                            mesh_dim_names=alias_name, rank_list=load_rank_list)
    else:
        # Ranks beyond load_mesh_size don't participate in load
        load_device_mesh = None

    # Create new state_dict for load with load configurations
    # Only ranks within load_mesh_size participate in load
    load_state_dict = {}
    if load_device_mesh is not None:
        for param_config in load_param_configs:
            param_name = param_config['name']
            placements = param_config['placements']
            local_shape = param_config['local_shape']

            load_local_tensor = to_device(torch.zeros(*local_shape), _DEVICE_TYPE)
            load_dtensor = DTensor.from_local(load_local_tensor, load_device_mesh, placements)
            load_state_dict[param_name] = load_dtensor

        # Initialize scalar values in load_state_dict (will be overwritten by load)
        if scalar_values:
            for scalar_name in scalar_values.keys():
                # Initialize with different values to verify they get loaded correctly
                if isinstance(scalar_values[scalar_name], int):
                    load_state_dict[scalar_name] = 0
                elif isinstance(scalar_values[scalar_name], float):
                    load_state_dict[scalar_name] = 0.0
                else:
                    load_state_dict[scalar_name] = None

        # Call load API (only ranks within load_mesh_size participate)
        load(load_state_dict, checkpoint_id=checkpoint_path)

    # Verify load results - compare loaded tensors with original tensors
    # Only ranks within load_mesh_size verify results
    if load_device_mesh is not None:
        # Note: Due to different mesh shapes, we compare global tensors
        # The checkpoint system should handle resharding automatically
        for param_config in save_param_configs:
            param_name = param_config['name']
            if param_name in load_state_dict:
                # Get full global tensor from loaded DTensor for comparison
                loaded_full_tensor = load_state_dict[param_name].full_tensor()
                original_full_tensor = original_global_tensors[param_name]

                # Compare global tensors (they should match if resharding is correct)
                assert np.allclose(
                    original_full_tensor.cpu().detach().numpy(),
                    loaded_full_tensor.cpu().detach().numpy(),
                    rtol=1e-5, atol=1e-5
                ), f"{param_name} values do not match after load with different mesh shape"

        # Verify load results - compare loaded scalars with original scalars
        if scalar_values:
            for scalar_name, original_value in original_scalars.items():
                loaded_value = load_state_dict[scalar_name]
                assert loaded_value == original_value, \
                    f"{scalar_name} scalar value mismatch: expected {original_value}, got {loaded_value}"
                assert type(loaded_value) == type(original_value), \
                    f"{scalar_name} scalar type mismatch: expected {type(original_value)}, got {type(loaded_value)}"


def test_dcp_save_and_load_save_8card_load_4card() -> None:
    """
    Feature: Test checkpoint save with 8-card cluster and load with 4-card cluster.
    Description: Test save function with state_dict containing DTensors and scalars on 8-card setup,
                 then load with 4-card setup. This tests resharding capability.
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values
                 after resharding from 8-card to 4-card layout.
    """
    checkpoint_path = Path("./test_dcp_8card_reshard_to_4card")

    # Save phase: 4-card cluster (1 data parallel, 4 tensor parallel)
    save_mesh_shape = (1, 4)

    # Load phase: 2-card cluster (1 data parallel, 2 tensor parallel)
    load_mesh_shape = (1, 2)

    # Parameter configurations for save (8-card):
    # param1: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (8, 8), Local shape: (8, 2) per rank (tp dimension has 4 devices)
    # param2: replicated (all dimensions replicated)
    #   Global shape: (4, 4), Local shape: (4, 4) per rank
    # param3: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (16, 8), Local shape: (16, 2) per rank (tp dimension has 4 devices)
    # Note: placements order is [dp_mesh_dim, tp_mesh_dim]
    save_param_configs = [
        {'name': 'param1', 'placements': [Replicate(), Shard(1)], 'local_shape': (8, 2)},
        {'name': 'param2', 'placements': [Replicate(), Replicate()], 'local_shape': (4, 4)},
        {'name': 'param3', 'placements': [Replicate(), Shard(1)], 'local_shape': (16, 2)},
    ]

    # Parameter configurations for load (4-card):
    # param1: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (8, 8), Local shape: (8, 4) per rank (tp dimension has 2 devices)
    # param2: replicated (all dimensions replicated)
    #   Global shape: (4, 4), Local shape: (4, 4) per rank
    # param3: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (16, 8), Local shape: (16, 4) per rank (tp dimension has 2 devices)
    # Note: placements order is [dp_mesh_dim, tp_mesh_dim]
    load_param_configs = [
        {'name': 'param1', 'placements': [Replicate(), Shard(1)], 'local_shape': (8, 4)},
        {'name': 'param2', 'placements': [Replicate(), Replicate()], 'local_shape': (4, 4)},
        {'name': 'param3', 'placements': [Replicate(), Shard(1)], 'local_shape': (16, 4)},
    ]

    # Add scalar values (int and float) to test mixed DTensor and scalar save/load
    scalar_values = {
        'epoch': 15,
        'learning_rate': 0.0008,
        'step': 2500,
        'loss': 0.23456,
        'best_accuracy': 0.95
    }

    _run_dcp_save_load_with_different_mesh_test(
        save_mesh_shape=save_mesh_shape,
        load_mesh_shape=load_mesh_shape,
        save_param_configs=save_param_configs,
        load_param_configs=load_param_configs,
        checkpoint_path=checkpoint_path,
        seed=3,
        scalar_values=scalar_values
    )

    platform_obj = get_platform()
    platform_obj.barrier()
    if platform_obj.get_rank() == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_save_and_load_with_static_dp_tp_pp() -> None:
    """
    Feature: DCP save and load with static DP + TP + PP topology.
    Description: Save a checkpoint on a 3-D mesh (pp=2, dp=2, tp=2) and reload on the
        same topology. Each PP stage wraps its parameters under a ``pp_stage_{pp_rank}``
        namespace so that different stages do not collide in the global plan.
    Expectation:
        - Each rank loads its local shard and the values match the originals.
        - Metadata contains FQNs for all PP stages.
        - Each TP-sharded parameter has exactly ``tp_size`` chunks in metadata (DP
          replicas are deduplicated).
        - Different stages with the same local parameter name do not overwrite each other.
    """
    checkpoint_path = Path("./test_dcp_static_dp_tp_pp")
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(42)
    np.random.seed(41)

    pp_size, dp_size, tp_size = 2, 2, 2
    root_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(pp_size, dp_size, tp_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )

    platform_obj = get_platform()
    rank = platform_obj.get_rank()

    pp_rank = rank // (dp_size * tp_size)
    stage_mesh = root_mesh["dp", "tp"]

    dp_tp_placements = [Replicate(), Shard(1)]
    param_local_shape = (8, 4)

    local_tensor = to_device(
        torch.randn(*param_local_shape) + pp_rank * 100.0 + (rank % tp_size) * 10.0,
        _DEVICE_TYPE,
    )
    dtensor = DTensor.from_local(local_tensor, stage_mesh, dp_tp_placements)

    stage_key = f"pp_stage_{pp_rank}"
    save_state_dict = {
        "model": {
            stage_key: {
                "layers.0.weight": dtensor,
            },
        },
    }
    original_local = dtensor.to_local().clone()

    metadata = save(save_state_dict, checkpoint_id=checkpoint_path, use_collectives=True)
    assert metadata is not None

    fqn0 = "model.pp_stage_0.layers.0.weight"
    fqn1 = "model.pp_stage_1.layers.0.weight"
    assert fqn0 in metadata.state_dict_metadata, f"Missing {fqn0} in metadata"
    assert fqn1 in metadata.state_dict_metadata, f"Missing {fqn1} in metadata"

    md0 = metadata.state_dict_metadata[fqn0]
    md1 = metadata.state_dict_metadata[fqn1]
    assert len(md0.chunks) == tp_size, (
        f"Stage 0 should have {tp_size} TP chunks, got {len(md0.chunks)}"
    )
    assert len(md1.chunks) == tp_size, (
        f"Stage 1 should have {tp_size} TP chunks, got {len(md1.chunks)}"
    )

    load_local_tensor = to_device(torch.zeros(*param_local_shape), _DEVICE_TYPE)
    load_dtensor = DTensor.from_local(load_local_tensor, stage_mesh, dp_tp_placements)
    load_state_dict = {
        "model": {
            stage_key: {
                "layers.0.weight": load_dtensor,
            },
        },
    }

    load(load_state_dict, checkpoint_id=checkpoint_path, use_collectives=True)

    loaded_local = load_state_dict["model"][stage_key]["layers.0.weight"].to_local()
    assert np.allclose(
        original_local.cpu().detach().numpy(),
        loaded_local.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    ), "Loaded local shard does not match original"

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_save_and_load_dynamic_tp_pp() -> None:
    """
    Feature: DCP save and load with dynamic TP+PP topology change.
    Description: Save a checkpoint on PP2×TP4 (4 logical layers, 2 per PP stage,
        each stage sharded across 4 TP ranks) and reload on PP4×TP2 (4 logical
        layers, 1 per PP stage, each stage sharded across 2 TP ranks).
        Each weight uses a deterministic value derived from "logical layer index +
        TP shard index" so that correctness can be verified after topology change.
        The TopologyMapper provides the explicit mapping from target FQNs (under the
        new PP stages) to checkpoint FQNs (under the old PP stages).
    Expectation:
        - Each rank loads its TP2 local shard and the values match the expected
          slice of the global weight for the corresponding logical layer.
        - Parameters that migrated to a different PP stage are loaded correctly
          via the FQN mapping.
        - A target TP2 shard may read from multiple source TP4 chunks.
    """
    checkpoint_path = Path("./test_dcp_dynamic_tp_pp")
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(100)
    np.random.seed(99)

    platform_obj = get_platform()
    rank = platform_obj.get_rank()
    world_size = platform_obj.get_world_size()
    assert world_size == 8, f"This test requires 8 ranks, got {world_size}"

    num_layers = 4
    global_shape = (8, 8)

    # ========== SAVE PHASE: PP2×TP4 ==========
    save_pp_size, save_dp_size, save_tp_size = 2, 1, 4
    save_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(save_pp_size, save_dp_size, save_tp_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )

    save_pp_rank = rank // (save_dp_size * save_tp_size)
    save_tp_rank = rank % save_tp_size
    save_stage_mesh = save_mesh["dp", "tp"]

    save_placements = [Replicate(), Shard(1)]
    local_cols = global_shape[1] // save_tp_size
    param_local_shape = (global_shape[0], local_cols)

    layers_per_stage = num_layers // save_pp_size
    stage_offset = save_pp_rank * layers_per_stage

    save_state_dict: dict[str, Any] = {"model": {}}
    original_global_weights: dict[int, torch.Tensor] = {}

    for local_idx in range(layers_per_stage):
        global_layer_idx = stage_offset + local_idx
        local_tensor = to_device(
            torch.full(param_local_shape, float(global_layer_idx * 100 + save_tp_rank * 10 + 1)),
            _DEVICE_TYPE,
        )
        dtensor = DTensor.from_local(local_tensor, save_stage_mesh, save_placements)
        stage_key = f"pp_stage_{save_pp_rank}"
        save_state_dict["model"].setdefault(stage_key, {})[
            f"layers.{local_idx}.weight"
        ] = dtensor
        original_global_weights[global_layer_idx] = dtensor.full_tensor().clone()

    metadata = save(save_state_dict, checkpoint_id=checkpoint_path, use_collectives=True)
    assert metadata is not None

    platform_obj.barrier()

    # ========== LOAD PHASE: PP4×TP2 ==========
    load_pp_size, load_dp_size, load_tp_size = 4, 1, 2
    load_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(load_pp_size, load_dp_size, load_tp_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )

    load_pp_rank = rank // (load_dp_size * load_tp_size)
    load_tp_rank = rank % load_tp_size
    load_stage_mesh = load_mesh["dp", "tp"]

    load_placements = [Replicate(), Shard(1)]
    load_local_cols = global_shape[1] // load_tp_size
    load_local_shape = (global_shape[0], load_local_cols)

    load_layers_per_stage = num_layers // load_pp_size
    assert load_layers_per_stage == 1

    load_state_dict: dict[str, Any] = {"model": {}}
    load_stage_key = f"pp_stage_{load_pp_rank}"
    load_local_tensor = to_device(torch.zeros(*load_local_shape), _DEVICE_TYPE)
    load_dtensor = DTensor.from_local(load_local_tensor, load_stage_mesh, load_placements)
    load_state_dict["model"][load_stage_key] = {
        "layers.0.weight": load_dtensor,
    }

    target_fqn = f"model.{load_stage_key}.layers.0.weight"
    global_layer_idx_for_load = load_pp_rank

    old_pp_stage = global_layer_idx_for_load // layers_per_stage
    old_local_idx = global_layer_idx_for_load % layers_per_stage
    checkpoint_fqn = f"model.pp_stage_{old_pp_stage}.layers.{old_local_idx}.weight"

    fqn_mapping = {target_fqn: checkpoint_fqn}
    mapper = TopologyMapper(target_to_checkpoint_fqn=fqn_mapping)
    planner = StandardLoadPlanner(topology_mapper=mapper)

    load(
        load_state_dict,
        checkpoint_id=checkpoint_path,
        planner=planner,
        use_collectives=True,
    )

    loaded_local = load_state_dict["model"][load_stage_key]["layers.0.weight"].to_local()
    expected_global = original_global_weights[global_layer_idx_for_load]
    expected_local = expected_global[:, load_tp_rank * load_local_cols:(load_tp_rank + 1) * load_local_cols]

    assert np.allclose(
        expected_local.cpu().detach().numpy(),
        loaded_local.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"Rank {rank} (load pp={load_pp_rank}, tp={load_tp_rank}): "
        f"local shard mismatch for layer {global_layer_idx_for_load}"
    )



    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_save_and_load_hsdp_ep_moe() -> None:
    """
    Feature: DCP save and load with HSDP + EP (MoE) topology, including EP resize
        and PP FQN mapping.
    Description: Save a MoE-like state_dict on a 3-D mesh
        (hsdp_rep=2, hsdp_shard=2, ep=2) where expert weights w1/w2/w3 are
        sharded along dim-0 by both hsdp_shard and ep, the router gate weight
        is replicated, and expert_bias / tokens_per_expert are regular tensors.
        All parameters are wrapped under ``model.pp_stage_0``.

        Reload on a 2-D mesh (ep=4, tp=2) where expert weights are sharded
        by ep on dim-0 and by tp on dim-1 (simulating ExpertTensorParallel).
        Parameters are wrapped under ``model.pp_stage_1`` with explicit
        TopologyMapper FQN mapping.

        Deterministic values encode the expert index so that correctness can
        be verified after the topology change.
    Expectation:
        - HSDP replicate ranks produce identical chunks; after dedup the
          expert weight metadata has exactly ``hsdp_shard * ep = 4`` unique
          chunks instead of 8.
        - Router gate weight has exactly 1 unique chunk (replicated on all
          8 ranks).
        - After loading with EP4×TP2, each rank's local expert shard matches
          the expected slice of the global weight.
        - Router, expert_bias, and tokens_per_expert are restored correctly.
    """
    checkpoint_path = Path("./test_dcp_hsdp_ep_moe")
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(200)
    np.random.seed(199)

    platform_obj = get_platform()
    rank = platform_obj.get_rank()
    world_size = platform_obj.get_world_size()
    assert world_size == 8, f"This test requires 8 ranks, got {world_size}"

    num_experts = 8
    dim = 16
    hidden_dim = 32
    expert_global_shape = (num_experts, hidden_dim, dim)

    # ========== SAVE PHASE: HSDP(rep=2, shard=2) × EP=2 ==========
    save_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("hsdp_rep", "hsdp_shard", "ep"),
    )

    save_shard_rank = (rank % 4) // 2
    save_ep_rank = rank % 2

    save_stage_mesh = save_mesh["hsdp_shard", "ep"]

    expert_placements = [Shard(0), Shard(0)]
    local_experts = num_experts // (2 * 2)
    expert_local_shape = (local_experts, hidden_dim, dim)

    local_tensor = to_device(
        torch.arange(
            save_shard_rank * 4 + save_ep_rank * local_experts,
            save_shard_rank * 4 + save_ep_rank * local_experts + local_experts,
            dtype=torch.float32,
        ).unsqueeze(-1).unsqueeze(-1).expand(expert_local_shape) + 0.1,
        _DEVICE_TYPE,
    )
    expert_dtensor = DTensor.from_local(
        local_tensor, save_stage_mesh, expert_placements,
    )

    router_placements = [Replicate(), Replicate()]
    router_local_shape = expert_global_shape[:1] + expert_global_shape[2:]
    router_local_tensor = to_device(
        torch.full(router_local_shape, 42.0),
        _DEVICE_TYPE,
    )
    router_dtensor = DTensor.from_local(
        router_local_tensor, save_stage_mesh, router_placements,
    )

    original_expert_global = expert_dtensor.full_tensor().clone()
    original_router_global = router_dtensor.full_tensor().clone()

    expert_bias_val = to_device(
        torch.full((num_experts,), 7.0), _DEVICE_TYPE,
    )
    tokens_per_expert_val = to_device(
        torch.zeros((num_experts,)), _DEVICE_TYPE,
    )

    save_state_dict: dict[str, Any] = {
        "model": {
            "pp_stage_0": {
                "experts.w1": expert_dtensor,
                "experts.w2": expert_dtensor.clone(),
                "experts.w3": expert_dtensor.clone(),
                "router.gate.weight": router_dtensor,
                "expert_bias": expert_bias_val,
                "tokens_per_expert": tokens_per_expert_val,
            },
        },
    }

    metadata = save(
        save_state_dict,
        checkpoint_id=checkpoint_path,
        use_collectives=True,
    )
    assert metadata is not None

    fqn_prefix = "model.pp_stage_0"
    for weight_name in ("experts.w1", "experts.w2", "experts.w3"):
        fqn = f"{fqn_prefix}.{weight_name}"
        assert fqn in metadata.state_dict_metadata, f"Missing {fqn}"
        md = metadata.state_dict_metadata[fqn]
        assert len(md.chunks) == 4, (
            f"{fqn} should have 4 unique chunks after HSDP dedup, got {len(md.chunks)}"
        )

    router_fqn = f"{fqn_prefix}.router.gate.weight"
    assert router_fqn in metadata.state_dict_metadata
    router_md = metadata.state_dict_metadata[router_fqn]
    assert len(router_md.chunks) == 1, (
        f"Router should have 1 unique chunk after dedup, got {len(router_md.chunks)}"
    )

    platform_obj.barrier()

    # ========== LOAD PHASE: EP=4 × TP=2, PP stage 0 → stage 1 ==========
    load_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(4, 2),
        mesh_dim_names=("ep", "tp"),
    )

    load_ep_rank = rank // 2
    load_tp_rank = rank % 2

    load_expert_placements = [Shard(0), Shard(2)]
    load_local_experts = num_experts // 4
    load_expert_local_shape = (load_local_experts, hidden_dim, dim // 2)

    load_expert_tensor = to_device(
        torch.zeros(*load_expert_local_shape), _DEVICE_TYPE,
    )
    load_expert_dtensor = DTensor.from_local(
        load_expert_tensor, load_mesh, load_expert_placements,
    )

    load_router_placements = [Replicate(), Replicate()]
    load_router_tensor = to_device(
        torch.zeros(*router_local_shape), _DEVICE_TYPE,
    )
    load_router_dtensor = DTensor.from_local(
        load_router_tensor, load_mesh, load_router_placements,
    )

    load_expert_bias = to_device(
        torch.full((num_experts,), -1.0), _DEVICE_TYPE,
    )
    load_tokens_per_expert = to_device(
        torch.full((num_experts,), -1.0), _DEVICE_TYPE,
    )

    load_state_dict: dict[str, Any] = {
        "model": {
            "pp_stage_1": {
                "experts.w1": load_expert_dtensor,
                "experts.w2": load_expert_dtensor.clone(),
                "experts.w3": load_expert_dtensor.clone(),
                "router.gate.weight": load_router_dtensor,
                "expert_bias": load_expert_bias,
                "tokens_per_expert": load_tokens_per_expert,
            },
        },
    }

    fqn_mapping: dict[str, str] = {}
    for name in (
        "experts.w1", "experts.w2", "experts.w3",
        "router.gate.weight", "expert_bias", "tokens_per_expert",
    ):
        target_fqn = f"model.pp_stage_1.{name}"
        ckpt_fqn = f"model.pp_stage_0.{name}"
        fqn_mapping[target_fqn] = ckpt_fqn

    mapper = TopologyMapper(target_to_checkpoint_fqn=fqn_mapping)
    planner = StandardLoadPlanner(topology_mapper=mapper)

    load(
        load_state_dict,
        checkpoint_id=checkpoint_path,
        planner=planner,
        use_collectives=True,
    )

    for weight_name in ("experts.w1", "experts.w2", "experts.w3"):
        loaded_local = load_state_dict["model"]["pp_stage_1"][weight_name].to_local()
        expected_global = original_expert_global
        expert_start = load_ep_rank * load_local_experts
        expert_end = expert_start + load_local_experts
        tp_start = load_tp_rank * (dim // 2)
        tp_end = tp_start + (dim // 2)
        expected_local = expected_global[expert_start:expert_end, :, tp_start:tp_end]

        assert np.allclose(
            expected_local.cpu().detach().numpy(),
            loaded_local.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"Rank {rank} (ep={load_ep_rank}, tp={load_tp_rank}): "
            f"{weight_name} local shard mismatch"
        )

    loaded_router_local = load_state_dict["model"]["pp_stage_1"][
        "router.gate.weight"
    ].to_local()
    assert np.allclose(
        original_router_global.cpu().detach().numpy(),
        loaded_router_local.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: router gate weight mismatch after load"

    loaded_bias = load_state_dict["model"]["pp_stage_1"]["expert_bias"]
    assert np.allclose(
        torch.full((num_experts,), 7.0).numpy(),
        loaded_bias.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: expert_bias mismatch after load"

    loaded_tpe = load_state_dict["model"]["pp_stage_1"]["tokens_per_expert"]
    assert np.allclose(
        torch.zeros((num_experts,)).numpy(),
        loaded_tpe.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: tokens_per_expert mismatch after load"

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)
