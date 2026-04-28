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
"""Minimal Llama3 tensor-parallel demo (TorchTitan-style TP plan on HyperParallel).

Run (from repo root or this directory)::

    torchrun --nproc_per_node=2 tensor_parallel_example.py

Requirements:
    * Sequence length must divide the TP world size (sequence parallel).
    * ``n_heads`` and ``n_kv_heads`` must divide the TP world size.

The layout follows ``torchtitan/models/llama3/parallelize.py`` ``apply_tp`` (embedding
row-wise + sequence-parallel norms + Colwise/Rowwise linears).
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

from hyper_parallel import SkipDTensorDispatch

from model import Llama3DemoConfig, Llama3Model
from parallelize import broadcast_state_dict_from_rank0, build_tp_mesh, parallelize_llama3


def init_dist() -> tuple[int, int]:
    """Initialize process group and bind one NPU per rank."""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)
    return rank, world


def main() -> None:
    rank, world = init_dist()
    device = torch.device("npu", rank)

    # Heads must divide TP size; sequence length must divide TP size for Shard(1).
    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    if cfg.n_heads % world != 0 or cfg.n_kv_heads % world != 0:
        raise ValueError("n_heads and n_kv_heads must be divisible by world_size.")

    torch.manual_seed(42 + rank)
    model = Llama3Model(cfg).to(device=device)

    # Same global weights on every rank before TP sharding.
    broadcast_state_dict_from_rank0(model)

    tp_mesh = build_tp_mesh(device_type="npu")
    parallelize_llama3(model, tp_mesh)

    batch_size = 2
    seq_len = 16
    if seq_len % world != 0:
        raise ValueError("seq_len must be divisible by TP world_size for sequence parallel.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

        logits = model(tokens)
        loss = F.cross_entropy(
            logits.float().reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        )
        # TP forward must keep DTensor op dispatch enabled; skip only during raw-tensor backward/opt.
        with SkipDTensorDispatch():
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
