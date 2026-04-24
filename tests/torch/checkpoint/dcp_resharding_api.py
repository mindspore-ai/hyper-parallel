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
"""DCP safe_open tests with real load-time resharding."""
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel import DTensor
from hyper_parallel.core.distributed_checkpoint import load, save
from hyper_parallel.core.distributed_checkpoint import filesystem_storage
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_module, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform import get_platform
from tests.torch.common_net import FullyShardTestNet
from tests.torch.utils import init_dist

FULLY_SHARD_HIDDEN_SIZE = 32
FULLY_SHARD_LAYERS = 2
FULLY_SHARD_MP_POLICY = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)


def _make_shared_temp_checkpoint_dir(prefix: str) -> Path:
    """Create one temporary checkpoint directory shared by all distributed ranks."""
    platform = get_platform()
    path_holder = [None]
    if platform.get_rank() == 0:
        path_holder[0] = tempfile.mkdtemp(prefix=prefix)
    dist.broadcast_object_list(path_holder, src=0)
    return Path(path_holder[0])


def _run_safe_open_reshard_case(
        case_name: str,
        save_mesh_shape: tuple[int, int],
        load_mesh_shape: tuple[int, int],
        save_param_configs: list[dict[str, Any]],
        load_param_configs: list[dict[str, Any]],
) -> None:
    """
    Save DTensors with one layout and load them with another layout.

    This verifies that real DCP load-time resharding reads tensor data through
    filesystem_storage.safe_open and reconstructs the original global tensors.
    """
    platform = get_platform()
    current_rank = platform.get_rank()
    load_mesh_size = load_mesh_shape[0] * load_mesh_shape[1]
    mesh_dim_names = ("dp", "tp")
    checkpoint_path = _make_shared_temp_checkpoint_dir(f"test_dcp_safe_open_{case_name}_")

    try:
        save_mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=save_mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        save_state_dict = {}
        original_global_tensors = {}
        for param_config in save_param_configs:
            param_name = param_config["name"]
            local_shape = param_config["local_shape"]
            placements = param_config["placements"]

            local_tensor = torch.randn(*local_shape).npu()
            dtensor = DTensor.from_local(local_tensor, save_mesh, placements)
            save_state_dict[param_name] = dtensor
            original_global_tensors[param_name] = dtensor.full_tensor().clone()

        metadata = save(save_state_dict, checkpoint_id=checkpoint_path)
        dist.barrier()

        assert metadata is not None
        for param_config in save_param_configs:
            assert param_config["name"] in metadata.state_dict_metadata

        load_state_dict = {}
        load_mesh = None
        if current_rank < load_mesh_size:
            load_mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=load_mesh_shape,
                mesh_dim_names=mesh_dim_names,
                rank_list=tuple(range(load_mesh_size)),
            )
            for param_config in load_param_configs:
                param_name = param_config["name"]
                local_shape = param_config["local_shape"]
                placements = param_config["placements"]

                local_tensor = torch.zeros(*local_shape).npu()
                load_state_dict[param_name] = DTensor.from_local(local_tensor, load_mesh, placements)

        with mock.patch.object(
                filesystem_storage, "safe_open", wraps=filesystem_storage.safe_open
        ) as safe_open_mock:
            load(load_state_dict, checkpoint_id=checkpoint_path)

        if load_mesh is not None:
            assert safe_open_mock.called, f"{case_name} did not load tensors through safe_open"
            for param_config in load_param_configs:
                param_name = param_config["name"]
                loaded_full_tensor = load_state_dict[param_name].full_tensor()
                original_full_tensor = original_global_tensors[param_name]
                assert np.allclose(
                    original_full_tensor.cpu().detach().numpy(),
                    loaded_full_tensor.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-5,
                ), f"{case_name}.{param_name} mismatch after safe_open resharding load"
    finally:
        dist.barrier()
        if current_rank == 0:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        dist.barrier()


def _build_tp_dp_fully_shard_model(num_cards: int) -> FullyShardTestNet:
    """Build a small model whose state_dict carries TP + fully_shard DP layouts."""
    assert num_cards == 4, f"fully_shard resharding test requires 4 cards, but got {num_cards}"

    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(num_cards,),
        mesh_dim_names=("axis0",),
    )
    mesh_2d = DeviceMesh(
        device_type="npu",
        mesh=np.array(root_mesh.rank_list, dtype=np.int32).reshape(2, 2),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    model = FullyShardTestNet(
        FULLY_SHARD_HIDDEN_SIZE,
        FULLY_SHARD_LAYERS,
        has_bias=False,
    )
    def partition_fn(mod_name: str, module: Any, device_mesh: Any) -> None:
        """Apply TP placements before fully_shard adds the DP shard dimension."""
        if mod_name == "":
            param = module.w1
            module.register_parameter(
                "w1",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, (Replicate(),)),
                    requires_grad=param.requires_grad,
                ),
            )
            return
        if mod_name == "dense_layers.layers.layer0":
            param = module.weight
            module.register_parameter(
                "weight",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, (Shard(1),)),
                    requires_grad=param.requires_grad,
                ),
            )
            return
        if mod_name == "dense_layers.layers.layer1":
            param = module.weight
            module.register_parameter(
                "weight",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, (Shard(0),)),
                    requires_grad=param.requires_grad,
                ),
            )

    distribute_module(
        model,
        device_mesh=tp_mesh,
        partition_fn=partition_fn,
    )
    for dense_layer in model.dense_layers.layers:
        fully_shard(
            dense_layer,
            mesh=dp_mesh,
            reshard_after_forward=True,
            mp_policy=FULLY_SHARD_MP_POLICY,
        )
    fully_shard(
        model,
        mesh=dp_mesh,
        reshard_after_forward=True,
        mp_policy=FULLY_SHARD_MP_POLICY,
    )
    model.set_reduce_op_type("sum")
    return model


def test_dcp_safe_open_basic_resharding_load() -> None:
    """
    Feature: Test safe_open reads in a lightweight DCP load-time resharding smoke case.
    Description:
        1. Save a small pair of DTensors on a 4-card TP mesh.
        2. Load them into a 2-card TP mesh with different target layouts.
        3. Verify the loaded global tensors still match the saved global tensors.
    Expectation: The basic TP4-to-TP2 resharding smoke case runs successfully.
    """
    init_dist()
    torch.manual_seed(3)
    np.random.seed(2)

    _run_safe_open_reshard_case(
        case_name="basic_tp4_to_tp2",
        save_mesh_shape=(1, 4),
        load_mesh_shape=(1, 2),
        save_param_configs=[
            {"name": "replicate_to_tp", "placements": [Replicate(), Replicate()], "local_shape": (8, 16)},
            {"name": "tp4_to_tp2_same_dim", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
        ],
        load_param_configs=[
            {"name": "replicate_to_tp", "placements": [Replicate(), Shard(1)], "local_shape": (8, 8)},
            {"name": "tp4_to_tp2_same_dim", "placements": [Replicate(), Shard(1)], "local_shape": (8, 8)},
        ],
    )


def test_dcp_safe_open_with_real_resharding_load() -> None:
    """
    Feature: Test safe_open reads in real DCP load-time resharding.
    Description:
        1. Save DTensors with one mesh and placement layout.
        2. Load them with a different mesh or placement layout.
        3. Verify load enters filesystem_storage.safe_open and reconstructed tensors are correct.
    Expectation: safe_open is used and loaded global tensors match the saved global tensors.
    """
    init_dist()
    torch.manual_seed(3)
    np.random.seed(2)

    _run_safe_open_reshard_case(
        case_name="tp4_to_dp2tp2",
        save_mesh_shape=(1, 4),
        load_mesh_shape=(2, 2),
        save_param_configs=[
            {"name": "tp_to_dp", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
            {"name": "tp_to_replicate", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
            {"name": "replicate_to_dp", "placements": [Replicate(), Replicate()], "local_shape": (8, 8)},
        ],
        load_param_configs=[
            {"name": "tp_to_dp", "placements": [Shard(0), Replicate()], "local_shape": (4, 16)},
            {"name": "tp_to_replicate", "placements": [Replicate(), Replicate()], "local_shape": (8, 16)},
            {"name": "replicate_to_dp", "placements": [Shard(0), Replicate()], "local_shape": (4, 8)},
        ],
    )

    _run_safe_open_reshard_case(
        case_name="dp2tp2_to_tp2",
        save_mesh_shape=(2, 2),
        load_mesh_shape=(1, 2),
        save_param_configs=[
            {"name": "two_dim_to_tp", "placements": [Shard(0), Shard(1)], "local_shape": (4, 4)},
            {"name": "dp_to_replicate", "placements": [Shard(0), Replicate()], "local_shape": (4, 8)},
            {"name": "tp_col_to_tp_row", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
        ],
        load_param_configs=[
            {"name": "two_dim_to_tp", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
            {"name": "dp_to_replicate", "placements": [Replicate(), Replicate()], "local_shape": (8, 8)},
            {"name": "tp_col_to_tp_row", "placements": [Replicate(), Shard(0)], "local_shape": (4, 8)},
        ],
    )

    _run_safe_open_reshard_case(
        case_name="tp4_to_tp2",
        save_mesh_shape=(1, 4),
        load_mesh_shape=(1, 2),
        save_param_configs=[
            {"name": "replicate_to_tp", "placements": [Replicate(), Replicate()], "local_shape": (8, 16)},
            {"name": "tp4_to_tp2_same_dim", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
        ],
        load_param_configs=[
            {"name": "replicate_to_tp", "placements": [Replicate(), Shard(1)], "local_shape": (8, 8)},
            {"name": "tp4_to_tp2_same_dim", "placements": [Replicate(), Shard(1)], "local_shape": (8, 8)},
        ],
    )


def test_dcp_safe_open_with_fully_shard_tp_dp_resharding_load() -> None:
    """
    Feature: Test safe_open reads when loading fully_shard TP+DP checkpoint shards into TP targets.
    Description:
        1. Save a model state_dict produced by distribute_module(TP) + fully_shard(DP).
        2. Load selected parameters into a smaller TP-only mesh with different placements.
        3. Verify safe_open is used and reconstructed full tensors match the saved full tensors.
    Expectation: DCP load-time resharding rebuilds TP target tensors from fully_shard checkpoint shards.
    """
    init_dist()
    torch.manual_seed(7)
    np.random.seed(7)

    platform = get_platform()
    current_rank = platform.get_rank()
    world_size = platform.get_world_size()
    checkpoint_path = _make_shared_temp_checkpoint_dir("test_dcp_safe_open_fully_shard_tp_dp_to_tp2_")

    try:
        save_model = _build_tp_dp_fully_shard_model(world_size)
        save_model_state = save_model.state_dict()
        load_param_configs = [
            {
                "name": "w1",
                "placements": [Replicate(), Replicate()],
                "local_shape": (FULLY_SHARD_HIDDEN_SIZE, FULLY_SHARD_HIDDEN_SIZE),
            },
            {
                "name": "dense_layers.layers.layer0.weight",
                "placements": [Replicate(), Shard(1)],
                "local_shape": (FULLY_SHARD_HIDDEN_SIZE, FULLY_SHARD_HIDDEN_SIZE // 2),
            },
            {
                "name": "dense_layers.layers.layer1.weight",
                "placements": [Replicate(), Shard(0)],
                "local_shape": (FULLY_SHARD_HIDDEN_SIZE // 2, FULLY_SHARD_HIDDEN_SIZE),
            },
        ]
        original_global_tensors = {
            param_config["name"]: save_model_state[param_config["name"]].full_tensor().clone()
            for param_config in load_param_configs
        }

        metadata = save({"model": save_model_state}, checkpoint_id=checkpoint_path)
        dist.barrier()

        assert metadata is not None
        for param_config in load_param_configs:
            assert f"model.{param_config['name']}" in metadata.state_dict_metadata

        load_state_dict = {}
        load_mesh = None
        if current_rank < 2:
            load_mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=(1, 2),
                mesh_dim_names=("dp", "tp"),
                rank_list=(0, 1),
            )
            load_state_dict["model"] = {}
            for param_config in load_param_configs:
                local_tensor = torch.zeros(*param_config["local_shape"]).npu()
                load_state_dict["model"][param_config["name"]] = DTensor.from_local(
                    local_tensor,
                    load_mesh,
                    param_config["placements"],
                )

        with mock.patch.object(
                filesystem_storage, "safe_open", wraps=filesystem_storage.safe_open
        ) as safe_open_mock:
            load(load_state_dict, checkpoint_id=checkpoint_path)

        if load_mesh is not None:
            assert safe_open_mock.called, "fully_shard TP+DP resharding load did not use safe_open"
            for param_config in load_param_configs:
                param_name = param_config["name"]
                loaded_full_tensor = load_state_dict["model"][param_name].full_tensor()
                original_full_tensor = original_global_tensors[param_name]
                assert np.allclose(
                    original_full_tensor.cpu().detach().numpy(),
                    loaded_full_tensor.cpu().detach().numpy(),
                    rtol=1e-5,
                    atol=1e-5,
                ), f"fully_shard_tp_dp_to_tp2.{param_name} mismatch after safe_open resharding load"
    finally:
        dist.barrier()
        if current_rank == 0:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        dist.barrier()
