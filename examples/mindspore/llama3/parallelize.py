# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensor-parallel placement for MindSpore ``Llama3Model`` (see ``model.py``).

Mirrors ``examples/torch/llama3/parallelize.py``: embedding row-wise + sequence axis shard,
sequence-parallel norms, Colwise/Rowwise ``Dense`` pairs for attention and SwiGLU.
"""
from __future__ import annotations

from mindspore import nn
from mindspore.communication import get_group_size

from hyper_parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    init_device_mesh,
    parallelize_module,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.platform import get_platform

from model import Llama3Model


def parallelize_llama3(
    model: Llama3Model,
    tp_mesh: DeviceMesh,
    *,
    enable_sequence_parallel: bool = True,
    enable_loss_parallel: bool = False,
) -> Llama3Model:
    """Apply 1-D tensor parallelism to ``model`` (TorchTitan-style TP plan)."""
    if not enable_sequence_parallel:
        raise NotImplementedError(
            "This demo only implements the TorchTitan-style path with sequence parallelism enabled."
        )

    tp_world = tp_mesh.size()
    cfg = model.cfg
    if cfg.n_heads % tp_world != 0:
        raise ValueError(f"n_heads ({cfg.n_heads}) must divide TP size ({tp_world}).")
    if cfg.n_kv_heads % tp_world != 0:
        raise ValueError(f"n_kv_heads ({cfg.n_kv_heads}) must divide TP size ({tp_world}).")

    sp_layout = Shard(1)

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=False,
            ),
            "norm": SequenceParallel(sequence_dim=1, use_local_output=False),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    rowwise_output_plan = RowwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=sp_layout,
        use_local_output=False,
    )
    norm_plan = SequenceParallel(sequence_dim=1, use_local_output=False)

    for block in model.layers:
        block.attention.tp_mesh_size = tp_world
        block.attention.tp_mesh = tp_mesh
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(sp_layout, Replicate(), Replicate()),
                desired_input_layouts=(Replicate(), Replicate(), Replicate()),
                use_local_output=False,
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
            "feed_forward": PrepareModuleInput(
                input_layouts=(sp_layout,),
                desired_input_layouts=(Replicate(),),
                use_local_output=False,
            ),
            "feed_forward.w1": ColwiseParallel(use_local_output=False),
            "feed_forward.w2": rowwise_output_plan,
            "feed_forward.w3": ColwiseParallel(use_local_output=False),
        }
        parallelize_module(block, tp_mesh, layer_plan)

    return model


def build_tp_mesh(device_type: str = "npu") -> DeviceMesh:
    """Build a 1-D TP mesh covering all ranks in the current process group."""
    world = get_group_size()
    # After ``dist.init()``, still pass ``init_backend=True`` so the mesh creates TP
    # process groups required by ``RowwiseParallel`` embedding output redistribution.
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(world,),
        mesh_dim_names=("tp",),
        init_backend=True,
    )


def build_dp_tp_mesh(
    tp_size: int,
    device_type: str = "npu",
) -> tuple[DeviceMesh, DeviceMesh, DeviceMesh]:
    """Build a 2-D ``(dp, tp)`` mesh and return ``(root_mesh, tp_mesh, dp_mesh)``.

    Args:
        tp_size: Tensor-parallel degree (must divide ``get_group_size()``).
        device_type: Device type passed to :func:`init_device_mesh`.

    Returns:
        Tuple of root mesh, 1-D TP submesh, and 1-D DP/FSDP submesh.

    Raises:
        ValueError: If ``tp_size`` is invalid or does not divide the world size.
    """
    world = get_group_size()
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1.")
    if world % tp_size != 0:
        raise ValueError(f"world_size ({world}) must be divisible by tp_size ({tp_size}).")
    dp_size = world // tp_size
    root_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
        init_backend=True,
    )
    return root_mesh, root_mesh["tp"], root_mesh["dp"]


def broadcast_state_dict_from_rank0(model: nn.Cell) -> None:
    """Broadcast parameters from rank 0 so every TP rank starts from the same weights."""
    if get_group_size() <= 1:
        return
    plat = get_platform()
    for _, p in model.parameters_and_names():
        if p is None:
            continue
        plat.broadcast(p, 0)
