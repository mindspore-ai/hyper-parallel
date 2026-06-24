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
"""Llama3 demo: tensor parallelism + fully_shard (FSDP2-style) on MindSpore + Ascend.

Layout mirrors ``examples/torch/llama3/fsdp_tp_example.py``: a 2-D device mesh ``(dp, tp)``
where ``parallelize_llama3`` uses the 1-D ``mesh["tp"]`` slice and ``fully_shard`` uses the
1-D ``mesh["dp"]`` slice.

Run (4 ranks, ``tp=2``, ``dp=2`` by default)::

    msrun --worker_num=4 --local_worker_num=4 --log_dir=./msrun_log --join=True fsdp_tp_example.py

Optional environment variables:

* ``LLAMA3_TP_SIZE`` — TP degree (default ``2``). ``world_size`` must be divisible by it.
"""
from __future__ import annotations

import os

import mindspore as ms
from mindspore import communication as dist
from mindspore import mint, nn, ops
from mindspore._c_expression import NoFallbackGuard
from mindspore.communication import get_group_size, get_rank

from model import Llama3DemoConfig, Llama3Model
from parallelize import (
    broadcast_state_dict_from_rank0,
    build_dp_tp_mesh,
    parallelize_llama3,
)

from hyper_parallel import SkipDTensorDispatch, fully_shard
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

enable_mindspore_backward_compat()


def _tp_size_from_env(world: int) -> int:
    """Read tensor-parallel width from ``LLAMA3_TP_SIZE`` and validate against ``world``."""
    raw = os.environ.get("LLAMA3_TP_SIZE", "2").strip()
    try:
        tp = int(raw)
    except ValueError as exc:
        raise ValueError(f"LLAMA3_TP_SIZE must be an integer, got {raw!r}") from exc
    if tp < 1:
        raise ValueError("LLAMA3_TP_SIZE must be >= 1.")
    if world % tp != 0:
        raise ValueError(f"world_size ({world}) must be divisible by LLAMA3_TP_SIZE ({tp}).")
    return tp


def _sync_parameter_names_from_fqn(cell: nn.Cell) -> None:
    """Sync ``Parameter.name`` to FQN so ``nn.Adam`` can build a unique ``ParameterTuple``."""
    for fqn, p in cell.parameters_and_names(expand=True):
        if p is None or not fqn:
            continue
        if p.name != fqn:
            p.name = fqn


def main() -> None:
    """Entry point: initialize distributed backend, build a Llama3 model with TP+FSDP, and run a training loop."""
    dist.init()
    rank = get_rank()
    world = get_group_size()
    tp_size = _tp_size_from_env(world)
    dp_size = world // tp_size

    tp_mesh, dp_mesh = build_dp_tp_mesh(tp_size=tp_size, device_type="npu")[1:]

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

    ms.set_seed(42 + rank)
    model = Llama3Model(cfg)

    broadcast_state_dict_from_rank0(model)
    parallelize_llama3(model, tp_mesh)

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    model.set_reduce_op_type("sum")
    _sync_parameter_names_from_fqn(model)

    batch_size = 2
    seq_len = 16
    if seq_len % tp_size != 0:
        raise ValueError("seq_len must be divisible by TP size for sequence parallel.")

    optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

    # Same minibatch on every rank (smoke test; identical data on all DP ranks).
    ms.set_seed(2026)
    tokens = mint.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=ms.int32)
    targets = mint.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=ms.int32)

    for step in range(2):
        model.zero_grad()
        logits = model(tokens)
        loss = mint.nn.functional.cross_entropy(
            ops.cast(logits, ms.float32).reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        )
        with SkipDTensorDispatch():
            loss.backward()
            grads = tuple(param.grad for param in model.trainable_params())
            with NoFallbackGuard():
                optimizer(grads)

        if rank == 0:
            print(
                f"[fsdp_tp step {step}] loss = {loss.item():.4f} "
                f"(dp={dp_size}, tp={tp_size}, world={world})"
            )


if __name__ == "__main__":
    main()
