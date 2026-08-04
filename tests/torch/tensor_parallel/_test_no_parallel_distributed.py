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
"""Distributed NPU worker tests for ``NoParallel`` (vs CPU single-device reference).

Launched from ``test_tp_styles_distributed.test_tp_styles_two_card_wave_one`` (2 ranks).

Scenarios:

1. **Replicated Linear forward** — NoParallel on a plain ``nn.Linear``; every rank
   computes the full matmul and produces a replicated output that matches the CPU
   single-device reference.
2. **Input redistribution from Shard** — Upstream ``SequenceParallel`` produces a
   ``Shard(1)`` DTensor; ``NoParallel(desired_input_layout=Replicate())`` must
   redistribute the sharded input back to ``Replicate()`` before the replicated
   module runs.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  -- Ascend NPU

from hyper_parallel import (
    NoParallel,
    SequenceParallel,
    init_device_mesh,
    parallelize_module,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.torch.utils import init_dist


def _make_tp_mesh_1d():
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def _npu_precision_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-4,
        atol=1e-5,
    )


def test_no_parallel_linear_and_redistribute_npu():
    """
    Feature: NoParallel Linear forward + SP→NoParallel redistribute (one torchrun)
    Description:
        1. Replicated Linear forward vs CPU F.linear
        2. SequenceParallel then NoParallel desired_input=Replicate vs CPU
    Expectation: Both scenarios match CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # --- replicated Linear ---
    torch.manual_seed(50)
    torch.npu.manual_seed(50)
    in_f, out_f, batch = 32, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    sharded = parallelize_module(linear, mesh, NoParallel(use_local_output=False))
    with torch.no_grad():
        y_hp = sharded(x.npu())
    assert isinstance(y_hp, DTensor), "output should be a DTensor"
    assert y_hp.placements == (Replicate(),), "output should be Replicate()"
    _npu_precision_close(y_hp.to_local(), y_ref)

    # --- redistribute sharded input ---
    torch.manual_seed(53)
    torch.npu.manual_seed(53)
    bsz, seq_len, hidden, out_f = 4, 16, 32, 24
    assert seq_len % world_size == 0

    class NormThenLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden, elementwise_affine=True)
            self.linear = nn.Linear(hidden, out_f, bias=True)

        def forward(self, x):
            return self.linear(self.norm(x))

    norm_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    linear_cpu = nn.Linear(hidden, out_f, bias=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = linear_cpu(norm_cpu(x_cpu))

    model = NormThenLinear().npu()
    with torch.no_grad():
        model.norm.load_state_dict(norm_cpu.state_dict())
        model.linear.weight.copy_(linear_cpu.weight.npu())
        model.linear.bias.copy_(linear_cpu.bias.npu())

    parallelize_module(model, mesh, {
        "norm": SequenceParallel(sequence_dim=1, use_local_output=False),
        "linear": NoParallel(desired_input_layout=Replicate(),
                             use_local_output=True),
    })
    chunk = seq_len // world_size
    sl = slice(rank * chunk, (rank + 1) * chunk)
    with torch.no_grad():
        y_local = model(x_cpu[:, sl, :].npu())
    _npu_precision_close(y_local, y_ref)
