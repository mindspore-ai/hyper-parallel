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
"""Llama3 demo: pipeline parallelism (1F1B) on the Torch backend.

Each rank owns one pipeline stage (embedding + layer slice on rank 0; LM head on
the last rank).  :class:`~hyper_parallel.core.pipeline_parallel.scheduler.Schedule1F1B`
drives forward/backward over micro-batches with automatic P2P.

Run (from this directory), e.g. 2 ranks with ``pp=2``::

    torchrun --nproc_per_node=2 pp_example.py

Optional environment variables:

* ``LLAMA3_PP_SIZE`` — pipeline width (default ``2``). Must equal ``world_size``.
* ``LLAMA3_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).
"""
from __future__ import annotations

import os

import torch

from demo_utils import init_dist, train_steps
from model import Llama3DemoConfig
from pipeline import (
    build_llama3_pp_chunk,
    build_pipeline_stage,
    count_llama3_parameters,
    split_batch_dim0,
)

from hyper_parallel import init_device_mesh
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B


def _pp_size_from_env(world: int) -> int:
    """Read pipeline width from ``LLAMA3_PP_SIZE`` and validate against ``world``."""
    raw = os.environ.get("LLAMA3_PP_SIZE", "2").strip()
    try:
        pp_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"LLAMA3_PP_SIZE must be an integer, got {raw!r}") from exc
    if pp_size < 1:
        raise ValueError("LLAMA3_PP_SIZE must be >= 1.")
    if world != pp_size:
        raise ValueError(
            f"This example expects world_size == LLAMA3_PP_SIZE "
            f"({pp_size}), got world_size={world}."
        )
    return pp_size


def main() -> None:
    """Entry point: run a Llama3 pipeline-parallel training demo."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    pp_size = _pp_size_from_env(world)

    pp_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size,),
        mesh_dim_names=("pp",),
    )
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

    micro_batch_num = 4
    batch_size = 8
    seq_len = 16
    if batch_size % micro_batch_num != 0:
        raise ValueError("batch_size must divide micro_batch_num.")

    torch.manual_seed(42 + pp_rank)
    stage_module = build_llama3_pp_chunk(cfg, pp_rank, pp_size).to(device=device)
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

    if rank == 0:
        ref_params = count_llama3_parameters(cfg)
        print(
            f"[pp] world={world}, pp={pp_size}, micro_batches={micro_batch_num}, "
            f"full-model params≈{ref_params:,}, local stage params="
            f"{sum(p.numel() for p in stage_module.parameters()):,}"
        )

    for step in range(train_steps()):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(1000 + step)
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

        if is_last:
            stage_module.set_micro_targets(split_batch_dim0(targets, micro_batch_num))

        if pp_rank == 0:
            losses = schedule.run(tokens)
        else:
            losses = schedule.run()

        optimizer.step()

        if is_last:
            mean_loss = sum(loss.item() for loss in losses) / len(losses)
            print(f"[pp step {step}] rank={rank} mean micro-batch loss = {mean_loss:.4f}")


if __name__ == "__main__":
    main()
