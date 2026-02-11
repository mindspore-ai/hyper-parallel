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
"""base dcp API"""
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# pylint: disable=W0611
from hyper_parallel import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.core.checkpoint import save, load
from hyper_parallel.core.device_mesh import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist


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
    init_dist()
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    # Create DeviceMesh
    alias_name = ("dp", "tp")
    device_mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=alias_name)

    # Create DTensors based on configurations
    state_dict = {}
    original_local_tensors = {}
    original_scalars = {}

    for param_config in param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        local_tensor = torch.randn(*local_shape).npu()
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
            tensor = torch.randn(*tensor_shape).npu()
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

        load_local_tensor = torch.zeros(*local_shape).npu()
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
            load_state_dict[tensor_name] = torch.zeros(*tensor_shape).npu()

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



def test_dcp_api_with_dtensor_and_tensor_and_scalar() -> None:
    """
    Feature: Test checkpoint save and load API with DTensor state_dict using different mesh_shape and layouts.
    Description: Test save and load function with state_dict containing DTensors on 8-card setup with mesh_shape (4, 2).
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values.
    """
    checkpoint_path = Path("./test_dcp_api")
    mesh_shape = (4, 2)  # Different mesh_shape: 4 data parallel, 2 tensor parallel

    # Parameter configurations:
    # Note: placements order is [dp_mesh_dim, tp_mesh_dim]
    # param1: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (16, 8), Local shape: (4, 8) per rank (dp dimension has 4 devices)
    # param2: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (12, 8), Local shape: (3, 4) per rank (dp has 4 devices, tp has 2 devices)
    # param3: replicated (all dimensions replicated)
    #   Global shape: (6, 6), Local shape: (6, 6) per rank
    # param4: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (8, 8), Local shape: (8, 4) per rank (tp dimension has 2 devices)
    # param5: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (20, 10), Local shape: (5, 10) per rank (dp dimension has 4 devices)
    # param6: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (16, 12), Local shape: (4, 6) per rank (dp has 4 devices, tp has 2 devices)
    # param7: replicated (all dimensions replicated)
    #   Global shape: (10, 10), Local shape: (10, 10) per rank
    # param8: sharded along tp dimension (tensor dim 1 sharded on tp mesh dim)
    #   Global shape: (14, 6), Local shape: (14, 3) per rank (tp dimension has 2 devices)
    # param9: sharded along dp dimension (tensor dim 0 sharded on dp mesh dim)
    #   Global shape: (24, 8), Local shape: (6, 8) per rank (dp dimension has 4 devices)
    # param10: sharded along both dimensions (tensor dim 0 on dp, dim 1 on tp)
    #   Global shape: (8, 10), Local shape: (2, 5) per rank (dp has 4 devices, tp has 2 devices)
    param_configs = [
        {'name': 'param1', 'placements': [Shard(0), Replicate()], 'local_shape': (4, 8)},
        {'name': 'param2', 'placements': [Shard(0), Shard(1)], 'local_shape': (3, 4)},
        {'name': 'param3', 'placements': [Replicate(), Replicate()], 'local_shape': (6, 6)},
        {'name': 'param4', 'placements': [Replicate(), Shard(1)], 'local_shape': (8, 4)},
        {'name': 'param5', 'placements': [Shard(0), Replicate()], 'local_shape': (5, 10)},
        {'name': 'param6', 'placements': [Shard(0), Shard(1)], 'local_shape': (4, 6)},
        {'name': 'param7', 'placements': [Replicate(), Replicate()], 'local_shape': (10, 10)},
        {'name': 'param8', 'placements': [Replicate(), Shard(1)], 'local_shape': (14, 3)},
        {'name': 'param9', 'placements': [Shard(0), Replicate()], 'local_shape': (6, 8)},
        {'name': 'param10', 'placements': [Shard(0), Shard(1)], 'local_shape': (2, 5)},
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
    checkpoint_path_no_coll = Path("./test_dcp_api_no_collectives")
    _run_dcp_save_load_test(
        mesh_shape=mesh_shape,
        param_configs=param_configs,
        checkpoint_path=checkpoint_path_no_coll,
        seed=3,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
        use_collectives=False,
    )


def test_dcp_api_with_full_tensor() -> None:
    """
    Feature: Test checkpoint save and load API with state_dict containing only torch Tensors.
    Description: Test save and load function with state_dict containing purely torch Tensors
                 (no DTensor, no scalars) on 8-card setup.
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values.
    """
    checkpoint_path = Path("./test_dcp_full_tensor")
    mesh_shape = (2, 4)

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
    init_dist()
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    # ========== SAVE PHASE: Use save_mesh_shape ==========
    # Create DeviceMesh for save
    alias_name = ("dp", "tp")
    save_device_mesh = init_device_mesh(device_type="npu", mesh_shape=save_mesh_shape, mesh_dim_names=alias_name)

    # Create DTensors based on save configurations
    save_state_dict = {}
    original_global_tensors = {}
    original_scalars = {}

    for param_config in save_param_configs:
        param_name = param_config['name']
        placements = param_config['placements']
        local_shape = param_config['local_shape']

        local_tensor = torch.randn(*local_shape).npu()
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
        load_device_mesh = init_device_mesh(device_type="npu", mesh_shape=load_mesh_shape,
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

            load_local_tensor = torch.zeros(*local_shape).npu()
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


def test_dcp_api_save_8card_load_4card() -> None:
    """
    Feature: Test checkpoint save with 8-card cluster and load with 4-card cluster.
    Description: Test save function with state_dict containing DTensors and scalars on 8-card setup,
                 then load with 4-card setup. This tests resharding capability.
    Expectation: Run success, checkpoint saved correctly, and loaded values match original values
                 after resharding from 8-card to 4-card layout.
    """
    checkpoint_path = Path("./test_dcp_8card_reshard_to_4card")

    # Save phase: 8-card cluster (2 data parallel, 4 tensor parallel)
    save_mesh_shape = (2, 4)

    # Load phase: 4-card cluster (2 data parallel, 2 tensor parallel)
    load_mesh_shape = (2, 2)

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
