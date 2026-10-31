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
"""Native GDN Ulysses must preserve complete-sequence gradients and packed resets."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu

import hyper_parallel as hp
from hyper_parallel.components.functional.gated_delta_net_ascendc import chunk_gated_delta_rule_ascendc
from hyper_parallel.distributed.context_parallel.gated_delta_net import (
    GatedDeltaNetUlyssesCP, _differentiable_all_to_all_shard,
)
from tests.torch.context_parallel._test_linear_tnd import _GDN, _assert_same, _execute, _isolation


def _core_inputs(tokens):
    """Fix effective inputs independently of backend norm and gate implementations."""
    generator = torch.Generator().manual_seed(17381)
    shape = (1, tokens, 96, 128)
    query, key = [torch.randn(shape, generator=generator) for _ in range(2)]
    query, key = [(value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
                  for value in (query, key)]
    value = (torch.randn(shape, generator=generator) * .2).bfloat16()
    gate = -(torch.rand(shape[:-1], generator=generator) * .015 + .005)
    beta = (torch.rand(shape[:-1], generator=generator) * .6 + .2).bfloat16()
    grad_output = (torch.randn(shape, generator=generator) * .1).bfloat16()
    return (query, key, value, gate, beta), grad_output


def _check_core(mesh, length, offsets):
    """Check native full-sequence output and all five gradients after Ulysses exchange."""
    rank, size = dist.get_rank(), dist.get_world_size()
    source, grad_output = _core_inputs(length * size)
    full = [value.npu().requires_grad_() for value in source]
    output, _ = chunk_gated_delta_rule_ascendc(*full, cu_seqlens=offsets)
    gradients = torch.autograd.grad(output, full, grad_output.npu())
    token_slice = slice(rank * length, (rank + 1) * length)
    expected = [value[:, token_slice].detach().cpu() for value in (output, *gradients)]
    del output, gradients, full
    local = [value[:, token_slice].npu().requires_grad_() for value in source]
    exchanged = [_differentiable_all_to_all_shard(value, mesh, split_dim=2, concat_dim=1) for value in local]
    output, _ = chunk_gated_delta_rule_ascendc(*exchanged, cu_seqlens=offsets)
    output = _differentiable_all_to_all_shard(output, mesh, split_dim=1, concat_dim=2)
    gradients = torch.autograd.grad(output, local, grad_output[:, token_slice].npu())
    for name, actual, reference in zip(("output", "q", "k", "v", "g", "beta"), (output, *gradients), expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0,
                                   msg=lambda message, field=name: f"native full/Ulysses {field}: {message}")


def _check_layer(mesh, length):
    """Check projection views, raw gates, two metadata lifetimes and document isolation."""
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(451)
    model = _GDN(96, 128).bfloat16().npu()
    model.A_log.data = model.A_log.data.float()
    model.dt_bias.data = model.dt_bias.data.float()
    wrapper = GatedDeltaNetUlyssesCP(model, mesh, backend="ascendc")
    local = (torch.randn(length, 32) * .05).bfloat16().npu()
    grad_output = (torch.randn(length, 32) * .01).bfloat16().npu()
    tails = (0, length + length // 2 + 1, 3 * length, size * length - 1, size * length)
    aligned = tuple(range(0, size * length + 1, length))
    inputs, gradients = [local, local * .75], [grad_output, grad_output * .5]
    expected = _execute(model, wrapper, inputs, gradients, [tails, aligned])
    actual = _execute(model, wrapper, inputs, gradients, [tails, aligned], live=True)
    _assert_same(actual, expected)
    _isolation(wrapper, local, grad_output, tails, rank, length)


def test_gdn_ascendc_ulysses() -> None:
    """Qualify dense/TND native core and packed full-layer checkpoint ownership."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl", timeout=timedelta(seconds=600))
    try:
        size = dist.get_world_size()
        mesh = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
        length = 4096
        _check_core(mesh, length, None)
        _check_core(mesh, length, (0, length + 1, 2 * length + 63, size * length - 1, size * length))
        _check_layer(mesh, length)
    finally:
        dist.destroy_process_group()
