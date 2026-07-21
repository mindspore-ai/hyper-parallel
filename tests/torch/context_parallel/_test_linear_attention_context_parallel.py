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
"""Distributed precision test for Qwen3.5 linear-attention CP modes."""

import os

import pytest
import torch
import torch.distributed as dist

import hyper_parallel as hp
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.context_parallel import LinearAttentionContextParallel
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.models.qwen3_5.model import Qwen3_5GatedDeltaNet


def _grad_norm(parameters, *, reduce_replicated: bool) -> torch.Tensor:
    total = torch.zeros((), device="npu", dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if reduce_replicated:
            grad = grad.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        total += grad.square().sum()
    return total.sqrt()


def _run_mode(mode: str, mesh) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    config = dict(
        hidden_size=1024,
        num_v_heads=16,
        num_k_heads=16,
        head_k_dim=128,
        head_v_dim=128,
    )
    torch.manual_seed(20260721)
    reference = Qwen3_5GatedDeltaNet(
        **config,
        gdn_backend="eager",
        conv_backend="eager",
    ).to(device="npu", dtype=torch.bfloat16)
    candidate = Qwen3_5GatedDeltaNet(**config).to(
        device="npu", dtype=torch.bfloat16
    )
    candidate.load_state_dict(reference.state_dict())
    if mode == "ulysses":
        candidate.gdn_backend = "triton"
        candidate.conv_backend = "triton"
    LinearAttentionContextParallel(mode=mode).apply(candidate, mesh)

    full_seq = 1024
    local_seq = full_seq // world
    torch.manual_seed(20260722)
    full_hidden = torch.randn(
        1, full_seq, config["hidden_size"], device="npu", dtype=torch.bfloat16
    ).requires_grad_(True)
    local_hidden = full_hidden[
        :, rank * local_seq:(rank + 1) * local_seq
    ].detach().clone().contiguous().requires_grad_(True)

    expected = reference(full_hidden)
    local_input = DTensor.from_local(local_hidden, mesh, (Shard(1),))
    with SkipDTensorDispatch():
        actual = candidate(local_input)
    expected_local = expected[:, rank * local_seq:(rank + 1) * local_seq]
    torch.testing.assert_close(
        actual.float(), expected_local.float(), rtol=5e-2, atol=1e-2
    )

    torch.manual_seed(20260723)
    grad_output = torch.randn_like(actual)
    gathered = [torch.empty_like(grad_output) for _ in range(world)]
    dist.all_gather(gathered, grad_output.contiguous())
    expected.backward(torch.cat(gathered, dim=1))
    actual.backward(grad_output)
    expected_input_grad = full_hidden.grad[
        :, rank * local_seq:(rank + 1) * local_seq
    ]
    torch.testing.assert_close(
        local_hidden.grad.float(), expected_input_grad.float(), rtol=8e-2, atol=2e-2
    )

    expected_norm = _grad_norm(reference.parameters(), reduce_replicated=False)
    actual_norm = _grad_norm(candidate.parameters(), reduce_replicated=True)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-3, atol=1e-2)
    dist.barrier()


def test_linear_attention_cp_modes_npu():
    """Validate Ulysses and optimized P2P on four NPU ranks."""
    pytest.importorskip("torch_npu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    hp.init_process_group(backend="hccl")
    if dist.get_world_size() != 4:
        pytest.skip("linear-attention CP precision test requires four ranks")
    mesh = hp.init_device_mesh("npu", (4,), mesh_dim_names=("cp",))
    for mode in ("ulysses", "p2p"):
        _run_mode(mode, mesh)
