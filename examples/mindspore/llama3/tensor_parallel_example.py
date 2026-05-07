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
"""Minimal Llama3 tensor-parallel demo on MindSpore + Ascend (HyperParallel).

Run (see README for ``msrun`` launcher flags; world size must match TP degree).
"""
# pylint: disable=C0413
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
from mindspore import communication as dist
from mindspore import mint, nn, ops
from mindspore._c_expression import NoFallbackGuard
from mindspore.communication import get_group_size, get_rank

from hyper_parallel import SkipDTensorDispatch

from model import Llama3DemoConfig, Llama3Model
from parallelize import broadcast_state_dict_from_rank0, build_tp_mesh, parallelize_llama3


def _zero_grad(cell: nn.Cell) -> None:
    for p in cell.trainable_params():
        p.grad = None


def _sync_parameter_names_from_fqn(cell: nn.Cell) -> None:
    """``nn.Adam`` builds a :class:`~mindspore.common.parameter.ParameterTuple` that requires unique
    ``Parameter.name`` values. After tensor-parallel ``distribute_module`` / styles, some shards may
    still report short names (e.g. ``weight``); sync each parameter's ``name`` to its FQN from
    :meth:`~mindspore.nn.Cell.parameters_and_names`.
    """
    for fqn, p in cell.parameters_and_names(expand=True):
        if p is None or not fqn:
            continue
        if p.name != fqn:
            p.name = fqn


def main() -> None:
    dist.init()
    rank = get_rank()

    cfg = Llama3DemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )
    world = get_group_size()
    if cfg.n_heads % world != 0 or cfg.n_kv_heads % world != 0:
        raise ValueError("n_heads and n_kv_heads must be divisible by world_size.")

    ms.set_seed(42 + rank)
    model = Llama3Model(cfg)

    broadcast_state_dict_from_rank0(model)

    tp_mesh = build_tp_mesh(device_type="npu")
    parallelize_llama3(model, tp_mesh)
    _sync_parameter_names_from_fqn(model)

    batch_size = 2
    seq_len = 16
    if seq_len % world != 0:
        raise ValueError("seq_len must be divisible by TP world_size for sequence parallel.")

    optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

    for step in range(2):
        _zero_grad(model)
        tokens = mint.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=ms.int32)
        targets = mint.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=ms.int32)

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
            print(f"[step {step}] loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
