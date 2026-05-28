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
"""Tensor-parallel placement for ``Llama3Model`` (see ``model.py``).

Follows the structure of TorchTitan ``torchtitan/models/llama3/parallelize.py`` ``apply_tp``:
embedding row-wise output on the sequence axis, sequence-parallel norms, Colwise/Rowwise Linear
pairs for attention and SwiGLU, and optional loss-parallel output sharding.
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


def parallelize_llama3(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    enable_sequence_parallel: bool = True,
    enable_loss_parallel: bool = False,
) -> nn.Module:
    """Apply 1-D tensor parallelism (TorchTitan-style TP plan).

    Works on :class:`~model.Llama3Model` and PP stage chunks from ``pipeline.py``
    (missing submodule paths such as ``tok_embeddings`` on middle stages are skipped).

    Requires ``n_heads % tp_world_size == 0`` and ``n_kv_heads % tp_world_size == 0``.

    Args:
        model: Model or PP stage chunk with ``cfg`` and ``layers`` attributes.
        tp_mesh: One-dimensional :class:`~hyper_parallel.core.dtensor.device_mesh.DeviceMesh`.
        enable_sequence_parallel: When ``True`` (default), use sequence parallelism on norms and
            chain Shard(1) activations like TorchTitan ``apply_tp``. ``False`` is not implemented.
        enable_loss_parallel: If ``True``, leave vocab shard on logits for fused CE (optional).

    Returns:
        The same ``model`` instance after in-place parallelization.
    """
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
                input_layouts=(sp_layout, Replicate()),
                desired_input_layouts=(Replicate(), Replicate()),
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
    if not dist.is_initialized():
        raise RuntimeError("Initialize distributed (e.g. torchrun) before build_tp_mesh.")
    world = dist.get_world_size()
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(world,),
        mesh_dim_names=("tp",),
    )


def broadcast_state_dict_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters from rank 0 so every TP rank starts from the same weights."""
    if not dist.is_initialized():
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
