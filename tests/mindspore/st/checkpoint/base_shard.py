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
"""base shard"""

import os
import shutil
import time
from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from mindspore.communication import get_rank

from hyper_parallel import DTensor, shard_module, parallelize_value_and_grad, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.core.distributed_checkpoint import save, load
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.layout import get_current_layout, save_layout, load_layout, get_global_layout, \
    combine_layout
from hyper_parallel.core.distributed_checkpoint.loader import load_checkpoint
from hyper_parallel.core.distributed_checkpoint.metadata import TensorStorageMetadata
from hyper_parallel.core.distributed_checkpoint.saver import save_checkpoint
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.topology_mapper import TopologyMapper
from hyper_parallel.core.distributed_checkpoint.versioning import CURRENT_CHECKPOINT_VERSION
from hyper_parallel.platform import get_platform

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(initializer("ones", [input_size, output_size], ms.float32), name='weight')
        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        x = ms.mint.sum(x)
        return x


def run_model(x, model, parallel=False):
    """rum model"""

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    if parallel is False:
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    else:
        grad_fn = parallelize_value_and_grad(forward_fn, optimizer.parameters)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        with NoFallbackGuard():
            optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[standalone] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def test_base_layout():
    """
    Feature: test layout save and load.
    Description: Test base layout save.
    Expectation: Run success.
    """
    D.init()

    # standalone
    input_size = 32
    output_size = 2

    # Create DeviceMesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 1), mesh_dim_names=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: save layout
    layout_dict = get_current_layout(model)
    rank_id = get_rank()
    file_name = f"test_{rank_id}.layout"
    save_layout(layout_dict, file_name)
    assert os.path.isfile(file_name)
    layout_dict = load_layout(file_name)
    assert isinstance(layout_dict, dict)
    # Sync all ranks before rank 0 reads files written by other ranks
    ms.ops.AllReduce()(ms.Tensor([1], dtype=ms.float32))
    if rank_id == 0:
        combine_dict = combine_layout(".")
        assert isinstance(combine_dict, dict)
    ms.ops.AllReduce()(ms.Tensor([1], dtype=ms.float32))
    os.remove(file_name)


def test_get_global_layout():
    """
    Feature: Test get global layout on all ranks.
    Description: Test when a simple model sharded by dp and mp, gather global layout on all ranks.
    Expectation: Run success.
    """
    D.init()

    # standalone
    input_size = 32
    output_size = 4

    # Create DeviceMesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 1), mesh_dim_names=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: get global layout
    global_layout = get_global_layout(model)
    assert isinstance(global_layout, dict)


def test_saver_loader():
    """
    Feature: Test checkpoint saver and loader.
    Description: Test when a simple model sharded by dp and mp, use saver and loader to save checkpoint to safetensors
    file, and use loader to load checkpoint from this safetensors file.
    Expectation: Run success.
    """
    D.init()

    # standalone
    input_size = 32
    output_size = 2

    # Create DeviceMesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 1), mesh_dim_names=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: save checkpoint
    rank_id = get_rank()
    file_path = f"tmp_{rank_id}.safetensors"
    save_checkpoint(model, file_path)

    # step 4: load checkpoint
    param_dict = load_checkpoint(file_path)
    os.remove(file_path)
    assert isinstance(param_dict, dict)


def test_dcp_save_and_load_with_static_dp_tp_pp():
    """
    Feature: DCP save and load with static DP + TP + PP topology (MindSpore).
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
    D.init()

    pp_size, dp_size, tp_size = 2, 2, 2
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(pp_size, dp_size, tp_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )

    platform_obj = get_platform()
    rank = platform_obj.get_rank()

    pp_rank = rank // (dp_size * tp_size)
    stage_mesh = root_mesh["dp", "tp"]

    dp_tp_placements = [Replicate(), Shard(1)]
    param_local_shape = (8, 4)

    np.random.seed(42 + rank)
    local_data = np.random.randn(*param_local_shape).astype(np.float32)
    local_data = local_data + float(pp_rank * 100) + float(rank % tp_size) * 10.0
    local_tensor = ms.Tensor(local_data)

    dtensor = DTensor.from_local(local_tensor, stage_mesh, dp_tp_placements)

    stage_key = f"pp_stage_{pp_rank}"
    save_state_dict = {
        "model": {
            stage_key: {
                "layers.0.weight": dtensor,
            },
        },
    }
    original_local = dtensor.to_local().copy()

    checkpoint_path = "./test_dcp_static_dp_tp_pp_ms"
    metadata = save(save_state_dict, checkpoint_id=checkpoint_path, use_collectives=True)
    assert metadata is not None

    fqn0 = "model.pp_stage_0.layers.0.weight"
    fqn1 = "model.pp_stage_1.layers.0.weight"
    assert fqn0 in metadata.state_dict_metadata, f"Missing {fqn0} in metadata"
    assert fqn1 in metadata.state_dict_metadata, f"Missing {fqn1} in metadata"

    md0 = metadata.state_dict_metadata[fqn0]
    md1 = metadata.state_dict_metadata[fqn1]
    assert isinstance(md0, TensorStorageMetadata)
    assert isinstance(md1, TensorStorageMetadata)
    assert len(md0.chunks) == tp_size, (
        f"Stage 0 should have {tp_size} TP chunks, got {len(md0.chunks)}"
    )
    assert len(md1.chunks) == tp_size, (
        f"Stage 1 should have {tp_size} TP chunks, got {len(md1.chunks)}"
    )

    load_local_tensor = ms.Tensor(np.zeros(param_local_shape, dtype=np.float32))
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
        original_local.asnumpy(),
        loaded_local.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), "Loaded local shard does not match original"

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_save_and_load_dynamic_tp_pp():
    """
    Feature: DCP save and load with dynamic TP+PP topology change (MindSpore).
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
    D.init()

    platform_obj = get_platform()
    rank = platform_obj.get_rank()
    world_size = platform_obj.get_world_size()
    assert world_size == 8, f"This test requires 8 ranks, got {world_size}"

    num_layers = 4
    global_shape = (8, 8)

    # ========== SAVE PHASE: PP2×TP4 ==========
    save_pp_size, save_dp_size, save_tp_size = 2, 1, 4
    save_mesh = init_device_mesh(
        device_type="npu",
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

    save_state_dict: dict = {"model": {}}
    original_global_weights: dict = {}

    for local_idx in range(layers_per_stage):
        global_layer_idx = stage_offset + local_idx
        local_data = np.full(
            param_local_shape,
            float(global_layer_idx * 100 + save_tp_rank * 10 + 1),
            dtype=np.float32,
        )
        local_tensor = ms.Tensor(local_data)
        dtensor = DTensor.from_local(local_tensor, save_stage_mesh, save_placements)
        stage_key = f"pp_stage_{save_pp_rank}"
        save_state_dict["model"].setdefault(stage_key, {})[
            f"layers.{local_idx}.weight"
        ] = dtensor
        original_global_weights[global_layer_idx] = dtensor.full_tensor().copy()

    checkpoint_path = "./test_dcp_dynamic_tp_pp_ms"
    metadata = save(save_state_dict, checkpoint_id=checkpoint_path, use_collectives=True)
    assert metadata is not None

    platform_obj.barrier()

    # ========== LOAD PHASE: PP4×TP2 ==========
    load_pp_size, load_dp_size, load_tp_size = 4, 1, 2
    load_mesh = init_device_mesh(
        device_type="npu",
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

    load_state_dict: dict = {"model": {}}
    load_stage_key = f"pp_stage_{load_pp_rank}"
    load_local_tensor = ms.Tensor(np.zeros(load_local_shape, dtype=np.float32))
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
    col_start = load_tp_rank * load_local_cols
    col_end = col_start + load_local_cols
    expected_local = expected_global[:, col_start:col_end]

    assert np.allclose(
        expected_local.asnumpy(),
        loaded_local.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"Rank {rank} (load pp={load_pp_rank}, tp={load_tp_rank}): "
        f"local shard mismatch for layer {global_layer_idx_for_load}"
    )

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)



def test_dcp_save_and_load_hsdp_ep_moe():
    """
    Feature: DCP save and load with HSDP + EP (MoE-like) DTensor topology (MindSpore).
    Description: Save a MoE-like state_dict on a 3-D mesh
        (hsdp_rep=2, hsdp_shard=2, ep=2) where expert weights w1/w2/w3 are
        sharded along dim-0 by both hsdp_shard and ep, the router gate weight
        is replicated, and expert_bias / tokens_per_expert are regular tensors.
        All parameters are wrapped under ``model.pp_stage_0``.

        Reload on a 2-D mesh (ep=4, tp=2) where expert weights are sharded
        by ep on dim-0 and by tp on dim-1.
        Parameters are wrapped under ``model.pp_stage_1`` with explicit
        TopologyMapper FQN mapping.

        This test only validates the DTensor data plane (chunk metadata,
        dedup, overlap planning, and value correctness). It does NOT
        exercise the MindSpore MoE/EP token-dispatch runtime.
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
    D.init()

    platform_obj = get_platform()
    rank = platform_obj.get_rank()
    world_size = platform_obj.get_world_size()
    assert world_size == 8, f"This test requires 8 ranks, got {world_size}"

    num_experts = 8
    dim = 16
    hidden_dim = 32
    _expert_global_shape = (num_experts, hidden_dim, dim)

    # ========== SAVE PHASE: HSDP(rep=2, shard=2) × EP=2 ==========
    save_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("hsdp_rep", "hsdp_shard", "ep"),
    )

    save_shard_rank = (rank % 4) // 2
    save_ep_rank = rank % 2

    save_stage_mesh = save_mesh["hsdp_shard", "ep"]

    expert_placements = [Shard(0), Shard(0)]
    local_experts = num_experts // (2 * 2)
    expert_local_shape = (local_experts, hidden_dim, dim)

    local_data = np.zeros(expert_local_shape, dtype=np.float32)
    expert_offset = save_shard_rank * 4 + save_ep_rank * local_experts
    for e in range(local_experts):
        local_data[e, :, :] = float(expert_offset + e) + 0.1
    local_tensor = ms.Tensor(local_data)

    expert_dtensor = DTensor.from_local(
        local_tensor, save_stage_mesh, expert_placements,
    )

    router_placements = [Replicate(), Replicate()]
    router_local_shape = (num_experts, dim)
    router_local_data = np.full(router_local_shape, 42.0, dtype=np.float32)
    router_local_tensor = ms.Tensor(router_local_data)
    router_dtensor = DTensor.from_local(
        router_local_tensor, save_stage_mesh, router_placements,
    )

    original_expert_global = expert_dtensor.full_tensor().copy()
    original_router_global = router_dtensor.full_tensor().copy()

    expert_bias_val = ms.Tensor(np.full((num_experts,), 7.0, dtype=np.float32))
    tokens_per_expert_val = ms.Tensor(np.zeros((num_experts,), dtype=np.float32))

    save_state_dict: dict = {
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

    checkpoint_path = "./test_dcp_hsdp_ep_moe_ms"
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
        assert isinstance(md, TensorStorageMetadata)
        assert len(md.chunks) == 4, (
            f"{fqn} should have 4 unique chunks after HSDP dedup, got {len(md.chunks)}"
        )

    router_fqn = f"{fqn_prefix}.router.gate.weight"
    assert router_fqn in metadata.state_dict_metadata
    router_md = metadata.state_dict_metadata[router_fqn]
    assert isinstance(router_md, TensorStorageMetadata)
    assert len(router_md.chunks) == 1, (
        f"Router should have 1 unique chunk after dedup, got {len(router_md.chunks)}"
    )

    platform_obj.barrier()

    # ========== LOAD PHASE: EP=4 × TP=2, PP stage 0 → stage 1 ==========
    load_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("ep", "tp"),
    )

    load_ep_rank = rank // 2
    load_tp_rank = rank % 2

    load_expert_placements = [Shard(0), Shard(2)]
    load_local_experts = num_experts // 4
    load_expert_local_shape = (load_local_experts, hidden_dim, dim // 2)

    load_expert_tensor = ms.Tensor(
        np.zeros(load_expert_local_shape, dtype=np.float32)
    )
    load_expert_dtensor = DTensor.from_local(
        load_expert_tensor, load_mesh, load_expert_placements,
    )

    load_router_placements = [Replicate(), Replicate()]
    load_router_tensor = ms.Tensor(
        np.zeros(router_local_shape, dtype=np.float32)
    )
    load_router_dtensor = DTensor.from_local(
        load_router_tensor, load_mesh, load_router_placements,
    )

    load_expert_bias = ms.Tensor(np.full((num_experts,), -1.0, dtype=np.float32))
    load_tokens_per_expert = ms.Tensor(np.full((num_experts,), -1.0, dtype=np.float32))

    load_state_dict: dict = {
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
            expected_local.asnumpy(),
            loaded_local.asnumpy(),
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
        original_router_global.asnumpy(),
        loaded_router_local.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: router gate weight mismatch after load"

    loaded_bias = load_state_dict["model"]["pp_stage_1"]["expert_bias"]
    assert np.allclose(
        np.full((num_experts,), 7.0, dtype=np.float32),
        loaded_bias.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: expert_bias mismatch after load"

    loaded_tpe = load_state_dict["model"]["pp_stage_1"]["tokens_per_expert"]
    assert np.allclose(
        np.zeros((num_experts,), dtype=np.float32),
        loaded_tpe.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: tokens_per_expert mismatch after load"

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_incremental_save_and_load():
    """
    Feature: incremental checkpoint save and load (MindSpore).
    Description:
        1) Save a full model checkpoint (baseline) with a dense tensor, a
           DTensor sharded along dim-0, a DTensor replicated, and an
           ``io_payload`` bytes dict.
        2) Mutate only the dense tensor and ``io_payload`` while keeping
           the two DTensors unchanged.
        3) Save an incremental checkpoint referencing the baseline with
           ``changed_fqns={"dense", "io_payload"}``.
        4) Load the incremental checkpoint and verify all values are
           correct — both changed and unchanged items.
        5) Verify the incremental checkpoint metadata version is
           ``CURRENT_CHECKPOINT_VERSION`` (``"2.0"``).
    Expectation: Incremental save only writes changed items; load resolves
        relocated baseline paths transparently and yields correct data.
    """
    D.init()

    platform_obj = get_platform()
    rank = platform_obj.get_rank()
    world_size = platform_obj.get_world_size()

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )

    base_path = Path("./test_dcp_incremental_save_and_load_ms")
    baseline_ckpt = base_path / "baseline"
    incremental_ckpt = base_path / "incremental"

    if rank == 0:
        if base_path.exists():
            shutil.rmtree(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
    platform_obj.barrier()

    dense_shape = (64, 32)
    shard_local_shape = (32, 32)
    replica_local_shape = (16, 24)

    # Step 1: build and save baseline
    dense_val = ms.Tensor(np.full(dense_shape, 1.0, dtype=np.float32))
    shard_local = ms.Tensor(np.arange(0, 32 * 32, dtype=np.float32).reshape(shard_local_shape) + 1.0)
    replica_local = ms.Tensor(np.full(replica_local_shape, 1.0, dtype=np.float32))
    io_payload = {"step": 1, "tag": "model-1"}

    dt_sharded = DTensor.from_local(shard_local, mesh, [Shard(0)])
    dt_replicated = DTensor.from_local(replica_local, mesh, [Replicate()])

    save_state_dict = {
        "dense": dense_val,
        "dt_sharded": dt_sharded,
        "dt_replicated": dt_replicated,
        "io_payload": io_payload,
    }

    save(save_state_dict, checkpoint_id=str(baseline_ckpt), use_collectives=True)
    platform_obj.barrier()

    # Step 2: mutate only dense and io_payload
    new_dense_val = ms.Tensor(np.full(dense_shape, 5.0, dtype=np.float32))
    new_shard_local = ms.Tensor(np.arange(0, 32 * 32, dtype=np.float32).reshape(shard_local_shape) + 1.0)
    new_replica_local = ms.Tensor(np.full(replica_local_shape, 1.0, dtype=np.float32))
    new_io_payload = {"step": 5, "tag": "model-5"}

    new_dt_sharded = DTensor.from_local(new_shard_local, mesh, [Shard(0)])
    new_dt_replicated = DTensor.from_local(new_replica_local, mesh, [Replicate()])

    save_state_dict["dense"] = new_dense_val
    save_state_dict["dt_sharded"] = new_dt_sharded
    save_state_dict["dt_replicated"] = new_dt_replicated
    save_state_dict["io_payload"] = new_io_payload

    # Step 3: incremental save — only dense and io_payload changed
    save(
        save_state_dict,
        checkpoint_id=str(incremental_ckpt),
        incremental_from=str(baseline_ckpt),
        changed_fqns={"dense", "io_payload"},
        use_collectives=True,
    )
    platform_obj.barrier()

    # Step 4: load the incremental checkpoint
    load_dense = ms.Tensor(np.zeros(dense_shape, dtype=np.float32))
    load_shard_local = ms.Tensor(np.zeros(shard_local_shape, dtype=np.float32))
    load_replica_local = ms.Tensor(np.zeros(replica_local_shape, dtype=np.float32))

    load_dt_sharded = DTensor.from_local(load_shard_local, mesh, [Shard(0)])
    load_dt_replicated = DTensor.from_local(load_replica_local, mesh, [Replicate()])

    load_state_dict = {
        "dense": load_dense,
        "dt_sharded": load_dt_sharded,
        "dt_replicated": load_dt_replicated,
        "io_payload": {"step": 0, "tag": ""},
    }

    load(load_state_dict, checkpoint_id=str(incremental_ckpt), use_collectives=True)
    platform_obj.barrier()

    assert np.allclose(
        load_state_dict["dense"].asnumpy(),
        new_dense_val.asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: dense tensor mismatch after incremental load"

    assert np.allclose(
        load_state_dict["dt_sharded"].to_local().asnumpy(),
        new_dt_sharded.to_local().asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: dt_sharded mismatch after incremental load"

    assert np.allclose(
        load_state_dict["dt_replicated"].to_local().asnumpy(),
        new_dt_replicated.to_local().asnumpy(),
        rtol=1e-5,
        atol=1e-5,
    ), f"Rank {rank}: dt_replicated mismatch after incremental load"

    loaded_io = load_state_dict["io_payload"]
    assert isinstance(loaded_io, dict)
    assert loaded_io["step"] == 5, (
        f"Rank {rank}: io_payload step should be 5, got {loaded_io['step']}"
    )

    # Step 5: verify metadata version
    reader = FileSystemReader(str(incremental_ckpt))
    md = reader.load_metadata()
    assert md.version == CURRENT_CHECKPOINT_VERSION, (
        f"Rank {rank}: metadata version should be {CURRENT_CHECKPOINT_VERSION}, got {md.version}"
    )

    platform_obj.barrier()
    if rank == 0:
        shutil.rmtree(str(base_path), ignore_errors=True)
