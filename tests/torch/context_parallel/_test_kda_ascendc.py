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
"""Native dense/TND Ulysses must preserve resets, gradients and saved calls."""
# pylint: disable=forbidden-backend-import
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.components.functional.kimi_delta_attention_ascendc import ascendc_chunk_kda
from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.components.modules.kimi_delta_attention import KimiDeltaAttention
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionCoreCP, KimiDeltaAttentionLayerCP, KimiDeltaAttentionUlyssesCP,
)
from tests.torch.context_parallel._test_linear_tnd import _assert_same, _execute, _isolation


def _local_core_inputs(full, rank, offsets):
    """Build differentiable rank-local inputs and their packed metadata."""
    local = [(value[:, rank * 4096:(rank + 1) * 4096] if index < 5 else value)
             .detach().clone().requires_grad_() for index, value in enumerate(full)]
    info = None if offsets is None else PackedSequenceInfo(offsets, rank * 4096, (rank + 1) * 4096)
    if info is not None:
        local = [value.squeeze(0) if index < 5 else value for index, value in enumerate(local)]
    return local, info


def _check_core(mesh, offsets):
    """Compare the two public Ulysses core entries to one native complete sequence."""
    size, rank = dist.get_world_size(), dist.get_rank()
    torch.manual_seed(83)
    shape = (1, 4096 * size, 4, 128)
    source = [torch.randn(shape).bfloat16() for _ in range(4)]
    source += [torch.randn(shape[:-1]).bfloat16(), torch.zeros(4), torch.full((4, 128), -5.)]
    full = [value.npu().requires_grad_() for value in source]
    grad_output = torch.randn(shape).bfloat16().npu()
    expected = ascendc_chunk_kda(*full[:5], a_log=full[5], dt_bias=full[6], cu_seqlens=offsets)
    expected_grads = torch.autograd.grad(expected, full, grad_output)
    local, info = _local_core_inputs(full, rank, offsets)
    executors = (KimiDeltaAttentionUlyssesCP(mesh, backend="ascendc"),
                 KimiDeltaAttentionCoreCP(mesh, ulysses_degree=size, backend="ascendc"))

    def run(*values: torch.Tensor) -> torch.Tensor:
        """Preserve packed metadata through non-reentrant checkpoint replay."""
        return executor(*values[:5], a_log=values[5], dt_bias=values[6], sequence_info=info)

    seed = grad_output[:, rank * 4096:(rank + 1) * 4096].contiguous()
    if info is not None:
        seed = seed.squeeze(0)
    for executor, checkpointed in zip(executors, (False, True)):
        actual = checkpoint(run, *local, use_reentrant=False) if checkpointed else run(*local)
        grads = torch.autograd.grad(actual, local, seed)
        reference_output = expected[:, rank * 4096:(rank + 1) * 4096]
        torch.testing.assert_close(actual, reference_output.squeeze(0) if info else reference_output, rtol=0, atol=0)
        for index, (grad, reference) in enumerate(zip(grads, expected_grads)):
            if index < 5:
                reference = reference[:, rank * 4096:(rank + 1) * 4096]
                if info is not None:
                    reference = reference.squeeze(0)
            else:
                dist.all_reduce(grad)
            torch.testing.assert_close(grad, reference, rtol=0, atol=0)


def _check_layer(mesh):
    """Exercise packed ShortConv, parameter views and two checkpointed live calls."""
    size, rank = dist.get_world_size(), dist.get_rank()
    torch.manual_seed(83)
    model = KimiDeltaAttention(hidden_size=32, num_heads=4, conv_kernel_size=4).bfloat16().npu()
    model.A_log.data = model.A_log.data.float()
    model.dt_bias.data = model.dt_bias.data.float()
    wrapper = KimiDeltaAttentionLayerCP(model, mesh, backend="ascendc", ulysses_degree=size)
    local = (torch.randn(4096, 32)*.05).bfloat16().npu()
    do = torch.randn_like(local)*.01
    offsets = (0, 4097, 4096*size-1, 4096*size)
    aligned = tuple(range(0, 4096*size+1, 4096))
    inputs, gradients = [local, local*.75], [do, do*.5]
    expected = _execute(model, wrapper, inputs, gradients, [offsets, aligned])
    actual = _execute(model, wrapper, inputs, gradients, [offsets, aligned], live=True)
    _assert_same(actual, expected)
    _isolation(wrapper, local, do, offsets, rank, 4096)


def test_kda_ascendc_ulysses() -> None:
    """Check dense/packed core parity and the complete packed layer lifecycle."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    try:
        size = dist.get_world_size()
        mesh = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
        _check_core(mesh, None)
        _check_core(mesh, (0, 4097, 4096*size-1, 4096*size))
        _check_layer(mesh)
    finally:
        dist.destroy_process_group()
