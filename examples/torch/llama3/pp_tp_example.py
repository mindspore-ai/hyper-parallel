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
"""Llama3 demo: pipeline parallelism + tensor parallelism (1F1B) on Torch.

Layout: 2-D ``DeviceMesh`` ``(pp, tp)``.  Each PP rank holds one stage chunk;
``parallelize_llama3`` shards that chunk across ``mesh["tp"]``.  Stage 0 accepts
a :class:`~hyper_parallel.DTensor` token batch (Replicate on TP); the last stage
computes per-micro-batch cross-entropy.

Run (from this directory), e.g. 4 ranks with ``pp=2``, ``tp=2``::

    torchrun --nproc_per_node=4 pp_tp_example.py

Optional environment variables:

* ``LLAMA3_PP_SIZE`` — pipeline width (default ``2``).
* ``LLAMA3_TP_SIZE`` — tensor-parallel width (default ``2``).
* ``LLAMA3_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).

Requirements:
    * ``world_size == LLAMA3_PP_SIZE * LLAMA3_TP_SIZE``.
    * ``n_heads`` / ``n_kv_heads`` divisible by ``LLAMA3_TP_SIZE``.
    * ``seq_len`` divisible by ``LLAMA3_TP_SIZE`` (sequence parallel).
"""
from __future__ import annotations

import os

import torch

from demo_utils import init_dist, train_steps
from model import Llama3DemoConfig
from parallelize import parallelize_llama3
from pipeline import (
    build_llama3_pp_chunk,
    build_pipeline_stage,
    split_batch_dim0,
)

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B


def _parallel_sizes_from_env(world: int) -> tuple[int, int]:
    """Read ``LLAMA3_PP_SIZE`` and ``LLAMA3_TP_SIZE`` and validate ``world``."""
    pp_raw = os.environ.get("LLAMA3_PP_SIZE", "2").strip()
    tp_raw = os.environ.get("LLAMA3_TP_SIZE", "2").strip()
    try:
        pp_size = int(pp_raw)
        tp_size = int(tp_raw)
    except ValueError as exc:
        raise ValueError(
            f"LLAMA3_PP_SIZE and LLAMA3_TP_SIZE must be integers, got "
            f"pp={pp_raw!r}, tp={tp_raw!r}"
        ) from exc
    if pp_size < 1 or tp_size < 1:
        raise ValueError("LLAMA3_PP_SIZE and LLAMA3_TP_SIZE must be >= 1.")
    if world != pp_size * tp_size:
        raise ValueError(
            f"world_size ({world}) must equal pp ({pp_size}) * tp ({tp_size})."
        )
    return pp_size, tp_size


def main() -> None:
    """Entry point: run a Llama3 pipeline + tensor-parallel training demo."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    pp_size, tp_size = _parallel_sizes_from_env(world)

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size, tp_size),
        mesh_dim_names=("pp", "tp"),
    )
    pp_mesh = mesh["pp"]
    tp_mesh = mesh["tp"]
    pp_rank = pp_mesh.get_local_rank()

    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    if cfg.n_layers < pp_size:
        raise ValueError(f"n_layers ({cfg.n_layers}) must be >= pp_size ({pp_size}).")
    if cfg.n_heads % tp_size != 0 or cfg.n_kv_heads % tp_size != 0:
        raise ValueError("n_heads and n_kv_heads must be divisible by TP size.")

    micro_batch_num = 4
    batch_size = 8
    seq_len = 16
    if seq_len % tp_size != 0:
        raise ValueError("seq_len must be divisible by TP size for sequence parallel.")
    if batch_size % micro_batch_num != 0:
        raise ValueError("batch_size must divide micro_batch_num.")

    # Same unsharded weights on all TP ranks within a PP stage before TP sharding.
    torch.manual_seed(42 + pp_rank)
    stage_module = build_llama3_pp_chunk(cfg, pp_rank, pp_size).to(device=device)
    parallelize_llama3(stage_module, tp_mesh)

    is_last = pp_rank == pp_size - 1
    pipeline_stage = build_pipeline_stage(
        stage_module,
        pp_rank=pp_rank,
        pp_size=pp_size,
        device=device,
        pp_mesh=pp_mesh,
        use_microbatch_loss=is_last,
    )
    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)

    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)

    for step in range(train_steps()):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(2000 + step)
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

        if is_last:
            stage_module.set_micro_targets(split_batch_dim0(targets, micro_batch_num))

        if pp_rank == 0:
            d_tokens = DTensor.from_local(tokens, tp_mesh, (Replicate(),))
            losses = schedule.run(d_tokens)
        else:
            losses = schedule.run()

        with SkipDTensorDispatch():
            optimizer.step()

        if is_last and tp_mesh.get_local_rank() == 0:
            mean_loss = sum(
                loss.item() if not hasattr(loss, "to_local") else loss.to_local().item()
                for loss in losses
            ) / len(losses)
            print(
                f"[pp_tp step {step}] rank={rank} mean micro-batch loss = {mean_loss:.4f} "
                f"(pp={pp_size}, tp={tp_size})"
            )


if __name__ == "__main__":
    main()
