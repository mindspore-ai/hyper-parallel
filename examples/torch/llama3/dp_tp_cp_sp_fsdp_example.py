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
"""Llama3 demo: DP + TP + CP + SP + FSDP2 on 8 ranks (Torch backend, HyperParallel).

Builds a 4-D ``DeviceMesh`` ``(dp, fsdp, cp, tp)`` and stacks all four parallelisms on the
Llama3-style decoder from ``model.py``:

* **TP + SP** — ``parallelize_llama3(model, mesh["tp"])`` applies the TorchTitan-style plan
  (``ColwiseParallel`` / ``RowwiseParallel`` linears + ``SequenceParallel`` norms with
  ``Shard(1)`` activations on the sequence axis).
* **CP**     — ``ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)`` is registered on
  every ``layer.attention.sdpa_core`` over ``mesh["cp"]`` (Colossal CP on BSHD Q/K/V).
* **FSDP2 + DP (HSDP)** — ``fully_shard`` is applied per-layer and on the root model over
  the 2-D slice ``mesh[("dp", "fsdp")]``: parameters are sharded inside the ``fsdp`` group
  and replicated across the ``dp`` group (Hybrid Sharded Data Parallel). When ``dp == 1``
  this collapses to plain FSDP2 over the ``fsdp`` axis.

Layout (default ``world_size = 8``)::

    (dp=1, fsdp=2, cp=2, tp=2) ->  1 * 2 * 2 * 2 = 8 ranks

Each ``(dp, fsdp)`` "DP domain" sees ``world / (cp * tp)`` ranks; the global token batch
is split across DP domains, while every CP/TP rank inside one domain holds the
``S/cp``-token / ``H/tp``-head local slice. RoPE is aligned to the global token positions
via ``Llama3Model.forward(..., rope_seq_start=...)``.

Run (from this directory or the repo root)::

    torchrun --nnodes=1 --nproc_per_node=8 dp_tp_cp_sp_fsdp_example.py

Optional environment variables (all default to ``2`` except ``LLAMA3_DP_SIZE = 1``);
must satisfy ``dp * fsdp * cp * tp == world_size``:

* ``LLAMA3_DP_SIZE``   — outer (HSDP replicate) data-parallel width. Default ``1``.
* ``LLAMA3_FSDP_SIZE`` — FSDP2 shard width. Default ``2``.
* ``LLAMA3_CP_SIZE``   — context-parallel width on attention SDPA. Default ``2``.
* ``LLAMA3_TP_SIZE``   — tensor-parallel width (with sequence parallel). Default ``2``.
* ``LLAMA3_DEVICE_TYPE`` — ``npu`` or ``cuda``. Default ``npu``.

Constraints:

* ``world_size == dp * fsdp * cp * tp``.
* ``n_heads`` and ``n_kv_heads`` divisible by ``tp``.
* ``seq_len`` divisible by ``cp``; ``(seq_len // cp)`` divisible by ``tp`` (sequence parallel
  shards the local CP window evenly across TP ranks).
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
import torch.nn.functional as F

from hyper_parallel import (
    ContextParallel,
    SkipDTensorDispatch,
    fully_shard,
    init_device_mesh,
)

from model import Llama3DemoConfig, Llama3Model
from parallelize import broadcast_state_dict_from_rank0, parallelize_llama3


_ENV_DEFAULTS = {
    "LLAMA3_DP_SIZE": "1",
    "LLAMA3_FSDP_SIZE": "2",
    "LLAMA3_CP_SIZE": "2",
    "LLAMA3_TP_SIZE": "2",
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
    """Read ``(dp, fsdp, cp, tp)`` from the environment and validate ``dp*fsdp*cp*tp == world``."""
    dp_size = _read_positive_int("LLAMA3_DP_SIZE", _ENV_DEFAULTS["LLAMA3_DP_SIZE"])
    fsdp_size = _read_positive_int("LLAMA3_FSDP_SIZE", _ENV_DEFAULTS["LLAMA3_FSDP_SIZE"])
    cp_size = _read_positive_int("LLAMA3_CP_SIZE", _ENV_DEFAULTS["LLAMA3_CP_SIZE"])
    tp_size = _read_positive_int("LLAMA3_TP_SIZE", _ENV_DEFAULTS["LLAMA3_TP_SIZE"])
    product = dp_size * fsdp_size * cp_size * tp_size
    if product != world:
        raise ValueError(
            f"world_size ({world}) must equal LLAMA3_DP_SIZE * LLAMA3_FSDP_SIZE * "
            f"LLAMA3_CP_SIZE * LLAMA3_TP_SIZE "
            f"({dp_size} * {fsdp_size} * {cp_size} * {tp_size} = {product})."
        )
    return dp_size, fsdp_size, cp_size, tp_size


def init_dist() -> tuple[int, int, str]:
    """Initialize the process group and bind one device per rank.

    Returns:
        ``(rank, world_size, device_type)``.
    """
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    device_type = os.environ.get("LLAMA3_DEVICE_TYPE", "npu").strip().lower()
    if device_type == "npu":
        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)
    else:
        raise ValueError(f"Unsupported LLAMA3_DEVICE_TYPE={device_type!r} (use npu or cuda).")
    return rank, world, device_type


def main() -> None:
    """Run one Llama3 forward + backward step composing DP + TP + CP + SP + FSDP2."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    dp_size, fsdp_size, cp_size, tp_size = _mesh_sizes_from_env(world)

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(dp_size, fsdp_size, cp_size, tp_size),
        mesh_dim_names=("dp", "fsdp", "cp", "tp"),
    )
    tp_mesh = mesh["tp"]
    cp_mesh = mesh["cp"]
    # HSDP slice: parameters shard inside ``fsdp`` and replicate across ``dp``.
    hsdp_mesh = mesh[("dp", "fsdp")]
    cp_rank = mesh.get_local_rank("cp")

    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    if cfg.n_heads % tp_size != 0 or cfg.n_kv_heads % tp_size != 0:
        raise ValueError("n_heads and n_kv_heads must be divisible by TP size.")

    batch_size = 2
    seq_len = 32
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must divide CP size ({cp_size}).")
    seq_per_cp = seq_len // cp_size
    if seq_per_cp % tp_size != 0:
        raise ValueError(
            f"(seq_len / cp) = {seq_per_cp} must divide TP size ({tp_size}) so the "
            "Rowwise embedding sequence shards inside each CP window stay even."
        )

    torch.manual_seed(42 + rank)
    model = Llama3Model(cfg).to(device=device)
    broadcast_state_dict_from_rank0(model)

    # 1) TP + SP plan on ``mesh["tp"]`` (TorchTitan-style apply_tp).
    parallelize_llama3(model, tp_mesh)

    # 2) Context parallel on every attention's BSHD SDPA core, over ``mesh["cp"]``.
    cp_plan = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)
    for layer in model.layers:
        cp_plan.apply(layer.attention.sdpa_core, cp_mesh)

    # 3) FSDP2 (+ outer DP replicate) via HSDP slice ``mesh[("dp", "fsdp")]``.
    for layer in model.layers:
        fully_shard(layer, mesh=hsdp_mesh)
    fully_shard(model, mesh=hsdp_mesh)
    # DTensor + fully_shard path uses SUM-typed gradient reduction; align with the
    # fsdp_tp example so the smoke loop runs with identical batches across DP/FSDP ranks.
    model.set_reduce_op_type("sum")

    # Each (dp, fsdp) rank inside one CP+TP "view" gets one DP-domain slice of the
    # global batch. With identical seeds + a single ``broadcast`` call, every rank
    # observes the same global batch -- this is a smoke test, not a numerical baseline.
    torch.manual_seed(2026)
    global_tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    global_targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(global_tokens, src=0)
    dist.broadcast(global_targets, src=0)

    cp_slice = slice(cp_rank * seq_per_cp, (cp_rank + 1) * seq_per_cp)
    tokens_local = global_tokens[:, cp_slice]
    targets_local = global_targets[:, cp_slice]
    rope_seq_start = cp_rank * seq_per_cp

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens_local, rope_seq_start=rope_seq_start)
        loss = F.cross_entropy(
            logits.float().reshape(-1, cfg.vocab_size),
            targets_local.reshape(-1),
        )
        with SkipDTensorDispatch():
            loss.backward()
            optimizer.step()

        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"rank {rank}: non-finite loss {loss.item()}")

        if rank == 0:
            print(
                f"[dp_tp_cp_sp_fsdp step {step}] loss={loss.item():.4f} "
                f"(dp={dp_size}, fsdp={fsdp_size}, cp={cp_size}, tp={tp_size}, world={world}, "
                f"local_seq={logits.shape[1]})"
            )


if __name__ == "__main__":
    main()
