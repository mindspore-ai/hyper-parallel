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
"""TorchTitan-style TP plan for the accuracy test ``Llama3Model``.

Self-contained copy of ``examples/torch/llama3/parallelize.py``: keeps the row-wise
embedding output, sequence-parallel norms, and Colwise/Rowwise attention/SwiGLU
plans so the parallel network is numerically equivalent to the standalone one.
"""
from __future__ import annotations

import torch.distributed as dist
from torch import nn

from hyper_parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

from tests.torch.accuracy.model import Llama3Model


def parallelize_llama3(
    model: Llama3Model,
    tp_mesh: DeviceMesh,
    *,
    enable_sequence_parallel: bool = True,
    enable_loss_parallel: bool = False,
) -> Llama3Model:
    """Apply 1-D tensor parallelism to ``model`` (TorchTitan-style TP+SP plan).

    Requires ``n_heads % tp_world_size == 0`` and ``n_kv_heads % tp_world_size == 0``.
    """
    if not enable_sequence_parallel:
        raise NotImplementedError(
            "This accuracy harness only supports the sequence-parallel TP path."
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


def broadcast_state_dict_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters from rank 0 so every rank starts from the same weights.

    Buffers are intentionally **not** broadcast: ``Llama3Model.freqs_cis`` is a complex tensor
    (HCCL does not support complex dtype broadcast), and it is deterministically recomputed from
    the same RoPE formula on every rank so a broadcast would be redundant.
    """
    if not dist.is_initialized():
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
