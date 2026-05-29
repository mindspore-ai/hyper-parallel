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
"""Expert-parallel placement for ``MoEDemoModel`` (see ``model.py``).

Provides :func:`parallelize_moe_ep` which applies :class:`ExpertParallel`
sharding to every ``feed_forward.experts`` sub-module across all transformer
layers, and optional tensor parallelism on the attention / embedding layers.
"""
from __future__ import annotations

import torch.distributed as dist
from torch import nn

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
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel


def parallelize_moe_ep(
    model: nn.Module,
    ep_mesh: DeviceMesh,
) -> nn.Module:
    """Apply Expert Parallelism to all MoE feed-forward expert modules.

    Each ``layer.feed_forward.experts`` (:class:`GroupedExperts`) is sharded
    with :class:`ExpertParallel` — expert weights are split on dim 0 and
    tokens are routed via differentiable all-to-all.

    Args:
        model: :class:`MoEDemoModel` instance.
        ep_mesh: 1-D DeviceMesh with mesh_dim_name ``"ep"``.

    Returns:
        The same ``model`` instance after in-place parallelization.

    Raises:
        ValueError: If ``num_experts`` is not divisible by the EP mesh size.
    """
    ep_world = ep_mesh.size()
    cfg = model.cfg
    if cfg.num_experts % ep_world != 0:
        raise ValueError(
            f"num_experts ({cfg.num_experts}) must divide EP size ({ep_world})."
        )

    for layer in model.layers:
        ExpertParallel().apply(layer.feed_forward.experts, ep_mesh)

    return model


def parallelize_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
) -> nn.Module:
    """Apply Tensor Parallelism to attention and embedding layers.

    Follows the same TorchTitan-style TP plan as the Llama3 demo:
    row-wise embedding, sequence-parallel norms, Colwise/Rowwise attention
    linears, and replicated MoE expert computation.

    Args:
        model: :class:`MoEDemoModel` instance.
        tp_mesh: 1-D DeviceMesh with mesh_dim_name ``"tp"``.

    Returns:
        The same ``model`` instance after in-place parallelization.

    Raises:
        ValueError: If ``n_heads`` or ``n_kv_heads`` is not divisible by TP size.
    """
    tp_world = tp_mesh.size()
    cfg = model.cfg
    if cfg.n_heads % tp_world != 0:
        raise ValueError(
            f"n_heads ({cfg.n_heads}) must divide TP size ({tp_world})."
        )
    if cfg.n_kv_heads % tp_world != 0:
        raise ValueError(
            f"n_kv_heads ({cfg.n_kv_heads}) must divide TP size ({tp_world})."
        )

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
                output_layouts=Replicate(),
                use_local_output=True,
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
                input_layouts=(sp_layout, Replicate()),
                desired_input_layouts=(Replicate(), Replicate()),
                use_local_output=False,
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
        }
        parallelize_module(block, tp_mesh, layer_plan)

    return model


def build_ep_mesh(device_type: str = "npu") -> DeviceMesh:
    """Build a 1-D EP mesh covering all ranks in the current process group.

    Args:
        device_type: ``"npu"`` or ``"cuda"``.

    Returns:
        1-D DeviceMesh with mesh_dim_name ``"ep"``.

    Raises:
        RuntimeError: If the distributed process group has not been initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError("Initialize distributed (e.g. torchrun) before build_ep_mesh.")
    world = dist.get_world_size()
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(world,),
        mesh_dim_names=("ep",),
    )


def broadcast_state_dict_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters from rank 0 so every EP rank starts from the same weights.

    Args:
        model: The model whose parameters should be synchronized.
    """
    if not dist.is_initialized():
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
