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
"""Llama3 demo: PP + FSDP2 + TP + CP + SP on 8 ranks (Torch backend, HyperParallel).

Builds a 4-D ``DeviceMesh`` ``(pp, fsdp, cp, tp)`` and stacks all five parallelisms
on Llama3-style PP stage chunks from ``pipeline.py``:

* **PP**     — ``Schedule1F1B`` over ``mesh["pp"]``; each PP rank owns one stage chunk.
* **TP + SP** — ``parallelize_llama3(stage, mesh["tp"])`` (TorchTitan-style plan).
* **CP**     — ``ContextParallel`` on every ``layer.attention.sdpa_core`` over ``mesh["cp"]``.
* **FSDP2**  — ``fully_shard`` on each decoder block over ``mesh["fsdp"]`` (not the stage root;
  avoids PP scheduler FSDP MetaStep conflicts on Torch).

Default layout (``world_size = 8``)::

    (pp=2, fsdp=2, cp=2, tp=1) -> 2 * 2 * 2 * 1 = 8 ranks

On 8 GPUs at least one mesh axis must be ``1``; default keeps **PP / FSDP / CP**
at width ``2`` and sets **TP** to ``1`` (``parallelize_llama3`` still runs; sequence
parallel is a no-op at width 1). Each training step uses ``micro_batch_num=1`` so
the PP ``Schedule1F1B`` schedule interacts cleanly with per-step FSDP unshard.
For ``tp=2`` as well, use ``16`` ranks ``(pp=2, fsdp=2, cp=2, tp=2)``.

Run (from this directory or the repo root)::

    torchrun --nnodes=1 --nproc_per_node=8 pp_fsdp_tp_cp_sp_example.py

Optional environment variables (defaults in parentheses); must satisfy
``pp * fsdp * cp * tp == world_size``:

* ``LLAMA3_PP_SIZE``   — pipeline parallel width (``2``).
* ``LLAMA3_FSDP_SIZE`` — FSDP2 shard width within each PP stage (``2``).
* ``LLAMA3_CP_SIZE``   — context-parallel width on attention SDPA (``2``).
* ``LLAMA3_TP_SIZE``   — tensor-parallel width with sequence parallel (``1`` on 8 GPUs).
* ``LLAMA3_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).

Constraints:

* ``world_size == pp * fsdp * cp * tp``.
* ``n_layers >= pp``; ``n_heads`` / ``n_kv_heads`` divisible by ``tp``.
* ``seq_len % cp == 0``; ``(seq_len // cp) % tp == 0``.
* batch size divisible by PP micro-batch count.
"""
# pylint: disable=C0413
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
import torch.distributed as dist

from hyper_parallel import ContextParallel, DTensor, SkipDTensorDispatch, fully_shard, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B

from demo_utils import init_dist, train_steps
from model import Llama3DemoConfig
from parallelize import parallelize_llama3
from pipeline import (
    build_llama3_pp_chunk,
    build_pipeline_stage,
    split_batch_dim0,
)

_ENV_DEFAULTS = {
    "LLAMA3_PP_SIZE": "2",
    "LLAMA3_FSDP_SIZE": "2",
    "LLAMA3_CP_SIZE": "2",
    "LLAMA3_TP_SIZE": "1",
}


def _read_positive_int(name: str, default: str) -> int:
    """Parse a positive integer from environment variable ``name``."""
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1 (got {value}).")
    return value


def _mesh_sizes_from_env(world: int) -> tuple[int, int, int, int]:
    """Read ``(pp, fsdp, cp, tp)`` from the environment and validate the product."""
    pp_size = _read_positive_int("LLAMA3_PP_SIZE", _ENV_DEFAULTS["LLAMA3_PP_SIZE"])
    fsdp_size = _read_positive_int("LLAMA3_FSDP_SIZE", _ENV_DEFAULTS["LLAMA3_FSDP_SIZE"])
    cp_size = _read_positive_int("LLAMA3_CP_SIZE", _ENV_DEFAULTS["LLAMA3_CP_SIZE"])
    tp_size = _read_positive_int("LLAMA3_TP_SIZE", _ENV_DEFAULTS["LLAMA3_TP_SIZE"])
    product = pp_size * fsdp_size * cp_size * tp_size
    if product != world:
        raise ValueError(
            f"world_size ({world}) must equal LLAMA3_PP_SIZE * LLAMA3_FSDP_SIZE * "
            f"LLAMA3_CP_SIZE * LLAMA3_TP_SIZE "
            f"({pp_size} * {fsdp_size} * {cp_size} * {tp_size} = {product})."
        )
    return pp_size, fsdp_size, cp_size, tp_size


def _apply_context_parallel(stage_module: torch.nn.Module, cp_mesh) -> None:
    """Register Colossal CP hooks on every attention SDPA core in a PP chunk."""
    cp_plan = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)
    for layer in stage_module.layers:
        cp_plan.apply(layer.attention.sdpa_core, cp_mesh)


def _apply_fsdp(stage_module: torch.nn.Module, fsdp_mesh) -> None:
    """Apply ``fully_shard`` per decoder block (not the stage root).

    The stage root stays a plain ``nn.Module`` so ``Schedule1F1B`` does not inject
    explicit FSDP unshard/reshard MetaSteps (which conflict with Torch autograd
    hooks when combined with TP).  Each ``HSDPModule`` layer still unshards inside
    its own ``forward``.
    """
    for layer in stage_module.layers:
        fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=True)
        if hasattr(layer, "set_reduce_op_type"):
            layer.set_reduce_op_type("sum")


def _validate_demo_constraints(
    cfg: Llama3DemoConfig,
    pp_size: int,
    fsdp_size: int,
    cp_size: int,
    tp_size: int,
    seq_len: int,
    batch_size: int,
    micro_batch_num: int,
) -> int:
    """Validate model/mesh layout; return local CP sequence length."""
    if cfg.n_layers < pp_size:
        raise ValueError(f"n_layers ({cfg.n_layers}) must be >= pp_size ({pp_size}).")
    if cfg.n_heads % tp_size != 0 or cfg.n_kv_heads % tp_size != 0:
        raise ValueError("n_heads and n_kv_heads must be divisible by TP size.")
    if fsdp_size < 2:
        raise ValueError(
            f"LLAMA3_FSDP_SIZE must be >= 2 (got {fsdp_size}). "
            "fsdp=1 causes DTensor mesh_shape mismatch with TP."
        )
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must divide CP size ({cp_size}).")
    seq_per_cp = seq_len // cp_size
    if seq_per_cp % tp_size != 0:
        raise ValueError(
            f"(seq_len / cp) = {seq_per_cp} must divide TP size ({tp_size}) for sequence parallel."
        )
    if batch_size % micro_batch_num != 0:
        raise ValueError("batch_size must divide micro_batch_num.")
    return seq_per_cp


def _build_scheduled_stage(
    cfg: Llama3DemoConfig,
    *,
    pp_rank: int,
    pp_size: int,
    cp_rank: int,
    seq_per_cp: int,
    device: torch.device,
    pp_mesh,
    fsdp_mesh,
    cp_mesh,
    tp_mesh,
    micro_batch_num: int,
) -> tuple[torch.nn.Module, Schedule1F1B, torch.optim.Optimizer, bool]:
    """Build a parallelized PP stage chunk and wrap it in ``Schedule1F1B``."""
    torch.manual_seed(42 + pp_rank)
    stage_module = build_llama3_pp_chunk(cfg, pp_rank, pp_size).to(device=device)
    stage_module.set_rope_seq_start(cp_rank * seq_per_cp)

    parallelize_llama3(stage_module, tp_mesh)
    _apply_context_parallel(stage_module, cp_mesh)
    _apply_fsdp(stage_module, fsdp_mesh)

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
    return stage_module, schedule, optimizer, is_last


def _run_training_loop(
    cfg: Llama3DemoConfig,
    *,
    rank: int,
    pp_rank: int,
    pp_size: int,
    fsdp_size: int,
    cp_size: int,
    tp_size: int,
    cp_rank: int,
    cp_slice: slice,
    batch_size: int,
    seq_len: int,
    micro_batch_num: int,
    stage_module: torch.nn.Module,
    schedule: Schedule1F1B,
    optimizer: torch.optim.Optimizer,
    is_last: bool,
    tp_mesh,
    fsdp_mesh,
    device: torch.device,
) -> None:
    """Execute the PP-scheduled training loop and log losses on the last stage."""
    for step in range(train_steps()):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(3000 + step)
        global_tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        global_targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        dist.broadcast(global_tokens, src=0)
        dist.broadcast(global_targets, src=0)

        tokens_local = global_tokens[:, cp_slice]
        targets_local = global_targets[:, cp_slice]

        if is_last:
            stage_module.set_micro_targets(split_batch_dim0(targets_local, micro_batch_num))

        if pp_rank == 0:
            d_tokens = DTensor.from_local(tokens_local, tp_mesh, (Replicate(),))
            losses = schedule.run(d_tokens)
        else:
            losses = schedule.run()

        with SkipDTensorDispatch():
            optimizer.step()

        if is_last and tp_mesh.get_local_rank() == 0 and cp_rank == 0 and fsdp_mesh.get_local_rank() == 0:
            mean_loss = sum(
                loss.item() if not hasattr(loss, "to_local") else loss.to_local().item()
                for loss in losses
            ) / len(losses)
            if not all(torch.isfinite(loss.detach()) for loss in losses):
                raise RuntimeError(f"rank {rank}: non-finite loss at step {step}")
            print(
                f"[pp_fsdp_tp_cp_sp step {step}] mean micro-batch loss = {mean_loss:.4f} "
                f"(pp={pp_size}, fsdp={fsdp_size}, cp={cp_size}, tp={tp_size})"
            )


def main() -> None:
    """Run PP-scheduled steps composing PP + FSDP2 + TP + CP + SP."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    pp_size, fsdp_size, cp_size, tp_size = _mesh_sizes_from_env(world)

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size, fsdp_size, cp_size, tp_size),
        mesh_dim_names=("pp", "fsdp", "cp", "tp"),
    )
    pp_mesh = mesh["pp"]
    fsdp_mesh = mesh["fsdp"]
    cp_mesh = mesh["cp"]
    tp_mesh = mesh["tp"]
    pp_rank = pp_mesh.get_local_rank()
    cp_rank = mesh.get_local_rank("cp")

    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    micro_batch_num = 1
    batch_size = 2
    seq_len = 32
    seq_per_cp = _validate_demo_constraints(
        cfg, pp_size, fsdp_size, cp_size, tp_size, seq_len, batch_size, micro_batch_num
    )

    stage_module, schedule, optimizer, is_last = _build_scheduled_stage(
        cfg,
        pp_rank=pp_rank,
        pp_size=pp_size,
        cp_rank=cp_rank,
        seq_per_cp=seq_per_cp,
        device=device,
        pp_mesh=pp_mesh,
        fsdp_mesh=fsdp_mesh,
        cp_mesh=cp_mesh,
        tp_mesh=tp_mesh,
        micro_batch_num=micro_batch_num,
    )

    if rank == 0:
        print(
            f"[pp_fsdp_tp_cp_sp] world={world}, pp={pp_size}, fsdp={fsdp_size}, "
            f"cp={cp_size}, tp={tp_size}, micro_batches={micro_batch_num}, "
            f"seq_len={seq_len}, local_cp_seq={seq_per_cp}"
        )

    _run_training_loop(
        cfg,
        rank=rank,
        pp_rank=pp_rank,
        pp_size=pp_size,
        fsdp_size=fsdp_size,
        cp_size=cp_size,
        tp_size=tp_size,
        cp_rank=cp_rank,
        cp_slice=slice(cp_rank * seq_per_cp, (cp_rank + 1) * seq_per_cp),
        batch_size=batch_size,
        seq_len=seq_len,
        micro_batch_num=micro_batch_num,
        stage_module=stage_module,
        schedule=schedule,
        optimizer=optimizer,
        is_last=is_last,
        tp_mesh=tp_mesh,
        fsdp_mesh=fsdp_mesh,
        device=device,
    )


if __name__ == "__main__":
    main()
