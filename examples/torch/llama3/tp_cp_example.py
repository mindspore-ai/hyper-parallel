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
"""Llama3 demo: TorchTitan-style TP (``parallelize_llama3``) + context parallel on attention.

Uses :class:`model.Llama3Model` (``model.py``) with a 2-D ``DeviceMesh`` ``(tp, cp)``:

* ``parallelize_llama3(model, mesh["tp"])`` — same plan as ``tensor_parallel_example.py`` (sequence
  parallel + Colwise/Rowwise on the **TP** submesh only).
* ``ContextParallel(..., ulysses_degree=1).apply(..., mesh["cp"])`` — Colossal CP on
  **every** ``layer.attention.sdpa_core`` (BSHD Q/K/V hooks), aligned with
  ``tests/torch/context_parallel/_test_context_parallel.py`` ``test_tp_cp_combination_npu``.

Each CP rank feeds the token slice ``global[:, cp * S/cp : (cp+1) * S/cp]``; ``Llama3Model.forward``
uses ``rope_seq_start`` so RoPE matches that global window (``freqs_cis`` is sliced after
``tok_embeddings`` using the **local** sequence length).

Run (default ``tp=2``, ``cp=2`` → ``world_size=4``)::

    torchrun --nproc_per_node=4 tp_cp_example.py

Environment:

* ``LLAMA3_TP_SIZE`` / ``LLAMA3_CP_SIZE`` — defaults ``2``; require ``world_size == tp * cp``.
* ``LLAMA3_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).

Constraints: ``seq_len % cp == 0``, ``(seq_len // cp) % tp == 0``, and ``n_heads`` / ``n_kv_heads``
divisible by ``tp``.
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

from hyper_parallel import ContextParallel, SkipDTensorDispatch, init_device_mesh

from model import Llama3DemoConfig, Llama3Model
from parallelize import broadcast_state_dict_from_rank0, parallelize_llama3


def _tp_cp_sizes_from_env(world: int) -> tuple[int, int]:
    """Read ``(tp, cp)`` from the environment and validate ``tp * cp == world``.

    Args:
        world: Current distributed world size.

    Returns:
        ``(tp_size, cp_size)`` positive integers whose product is ``world``.

    Raises:
        ValueError: If env vars are invalid or ``tp * cp != world``.
    """
    raw_tp = os.environ.get("LLAMA3_TP_SIZE", "2").strip()
    raw_cp = os.environ.get("LLAMA3_CP_SIZE", "2").strip()
    try:
        tp_size = int(raw_tp)
        cp_size = int(raw_cp)
    except ValueError as exc:
        raise ValueError(
            f"LLAMA3_TP_SIZE and LLAMA3_CP_SIZE must be integers, got {raw_tp!r} and {raw_cp!r}."
        ) from exc
    if tp_size < 1 or cp_size < 1:
        raise ValueError("LLAMA3_TP_SIZE and LLAMA3_CP_SIZE must be >= 1.")
    if tp_size * cp_size != world:
        raise ValueError(
            f"world_size ({world}) must equal LLAMA3_TP_SIZE * LLAMA3_CP_SIZE "
            f"({tp_size} * {cp_size} = {tp_size * cp_size})."
        )
    return tp_size, cp_size


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
    """Run one Llama3 forward with TP + CP and a short CE + backward step."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    tp_size, cp_size = _tp_cp_sizes_from_env(world)

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(tp_size, cp_size),
        mesh_dim_names=("tp", "cp"),
    )
    cp_rank = mesh.get_local_rank("cp")

    batch_size = 1
    seq_len = 8
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must divide CP size ({cp_size}).")
    seq_per_cp = seq_len // cp_size
    if seq_per_cp % tp_size != 0:
        raise ValueError(
            f"(seq_len / cp) = {seq_per_cp} must divide TP size ({tp_size}) so Rowwise embedding "
            "sequence shards are even."
        )
    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    if cfg.n_heads % tp_size != 0 or cfg.n_kv_heads % tp_size != 0:
        raise ValueError("n_heads and n_kv_heads must divide TP size.")

    torch.manual_seed(42 + rank)
    model = Llama3Model(cfg).to(device=device)
    broadcast_state_dict_from_rank0(model)
    parallelize_llama3(model, mesh["tp"])

    cp_plan = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)
    for layer in model.layers:
        cp_plan.apply(layer.attention.sdpa_core, mesh["cp"])

    torch.manual_seed(2026)
    global_tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(global_tokens, src=0)

    tokens_cp = global_tokens[:, cp_rank * seq_per_cp : (cp_rank + 1) * seq_per_cp]
    targets_cp = tokens_cp
    rope_seq_start = cp_rank * seq_per_cp

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens_cp, rope_seq_start=rope_seq_start)
    loss = F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size),
        targets_cp.reshape(-1),
    )
    with SkipDTensorDispatch():
        loss.backward()
        optimizer.step()

    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"rank {rank}: non-finite loss {loss.item()}")

    if rank == 0:
        print(
            f"tp_cp_example OK: loss={loss.item():.4f} local_seq={logits.shape[1]}, "
            f"tp={tp_size}, cp={cp_size}, logits_shape={tuple(logits.shape)}"
        )


if __name__ == "__main__":
    main()
