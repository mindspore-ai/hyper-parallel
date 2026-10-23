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
"""Real collective checks for KDA local-M state exchange and DP submeshes."""
import copy
from datetime import timedelta
import os
from unittest.mock import patch

import torch
import torch.distributed as dist

import hyper_parallel as hp
from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import KimiDeltaAttentionLayerCP
from hyper_parallel.distributed.context_parallel import kimi_delta_attention_state as state_cp
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_state import (
    build_kda_state_cp,
    split_kda_mesh,
)


def _cpu_merge(states, matrices, rank, forward):
    """Exercise real Gloo transport with a test-only FP32 merge backend."""
    output = torch.zeros_like(states[0])
    indices = range(rank) if forward else range(states.shape[0] - 1, rank, -1)
    for index in indices:
        matrix = matrices[index] if forward else matrices[index].transpose(-1, -2)
        output = matrix @ output + states[index]
    return output


def _cpu_scan(maps, incoming, indices, forward, prefix_count=None):
    """Substitute only device arithmetic while checking real grouped transport."""
    result, prefix = incoming, incoming if prefix_count == 0 else None
    for step, index in enumerate(indices, 1):
        matrix = maps[index, ..., 128:]
        result = (matrix if forward else matrix.transpose(-1, -2)) @ result + maps[index, ..., :128]
        if step == prefix_count:
            prefix = result
    return result, prefix


def _check(mesh, width):
    """Compare ordered states and per-call M storage with FP64 references."""
    rank, size = mesh.get_local_rank(), mesh.size()
    protocol = build_kda_state_cp(mesh, "allgather" if width == size else "grouped_allgather_p2p",
                                  1 if width == size else width)
    generator = torch.Generator().manual_seed(724 + sum(mesh.rank_list))
    shape = (size, 1, 96, 128, 128)
    matrices = torch.randn(shape, generator=generator) * .012
    states = torch.randn(shape, generator=generator) * .01
    gradients = torch.randn(shape, generator=generator) * .01
    device = "npu" if dist.get_backend() == "hccl" else "cpu"
    local_m = [matrices[rank].to(device), (matrices[rank] * .95).to(device)]
    output = [protocol.forward((states[rank] * (1 + step)).to(device), local_m[step]) for step in range(2)]
    for step in (1, 0):
        state, cache = output[step]
        if len(cache) != 1 or cache[0] is not local_m[step]:
            raise AssertionError("Gather cache must reuse this invocation's local M")
        actual = protocol.backward(gradients[rank].to(device), cache)
        expected = torch.zeros_like(states[0], dtype=torch.float64)
        adjoint = torch.zeros_like(expected)
        maps = matrices * (.95 if step else 1.)
        for index in range(rank):
            expected = maps[index].double() @ expected + (states[index] * (1 + step)).double()
        for index in range(size - 1, rank, -1):
            adjoint = gradients[index].double() + maps[index].double().transpose(-1, -2) @ adjoint
        torch.testing.assert_close(state.cpu().double(), expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual.cpu().double(), adjoint, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(local_m[step].cpu(), maps[rank], atol=0, rtol=0)
    if any(isinstance(value, torch.Tensor) for value in vars(protocol).values()):
        raise AssertionError("The shared protocol must not retain any invocation tensors")
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"PASS local-M state CP{size}/g{width} ranks={mesh.rank_list}", flush=True)


def _run(device):
    """Run state CP checks on WORLD and root-preserving CP subgroups."""
    torch.set_num_threads(1)
    if device == "npu":
        # The CPU transport qualification must remain usable without torch-npu.
        import torch_npu  # pylint: disable=C0415
        torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch.npu.matmul.allow_hf32 = False
        torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl" if device == "npu" else "gloo", timeout=timedelta(seconds=180))
    size = dist.get_world_size()
    root = hp.init_device_mesh(device, (size,), mesh_dim_names=("cp",))
    _check(root, size)
    _check(root, 2)
    # cp is the outer axis, so DP replicas own interleaved global ranks.
    dp_root = hp.init_device_mesh(device, (size // 2, 2), mesh_dim_names=("cp", "dp"))
    _check(dp_root["cp"], min(2, size // 2))
    if size >= 8:
        _check(root, 4)
        hybrid = split_kda_mesh(root, 2, ("kda_state", "kda_ulysses"))
        _check(hybrid["kda_state"], 2)
    dist.destroy_process_group()


def test_kda_ag_state_npu():
    """Qualify the fused merge and owner chain on real NPU groups."""
    _run("npu")


def test_kda_ag_state_gloo():
    """Qualify root-preserving subgroup order with real CPU communication."""
    with patch.object(state_cp, "merge_affine_summaries", _cpu_merge), \
            patch.object(state_cp, "_scan_maps", _cpu_scan):
        _run("cpu")


def test_kda_hybrid_layout() -> None:
    """Match the serial layer, including halo and replicated parameter gradients."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = hp.init_device_mesh("cpu", (size,), mesh_dim_names=("cp",))
    torch.manual_seed(412)
    reference = KimiDeltaAttention(hidden_size=8, num_heads=4, num_v_heads=4,
                                   head_k_dim=4, head_v_dim=4, conv_kernel_size=3,
                                   chunk_size=4).float().train()
    model = copy.deepcopy(reference)
    executor = KimiDeltaAttentionLayerCP(model, mesh, ulysses_degree=2,
                                               chunk_size=4, backend="eager")
    torch.manual_seed(523)
    full = torch.randn(1, 32, 8, requires_grad=True)
    grad_output = torch.randn_like(full)
    local = full.detach().chunk(size, dim=1)[rank].contiguous().requires_grad_()
    expected = reference(full)
    actual = executor(local)
    actual.backward(grad_output.chunk(size, dim=1)[rank].contiguous())
    expected.backward(grad_output)
    torch.testing.assert_close(actual, expected.detach().chunk(size, dim=1)[rank], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(local.grad, full.grad.chunk(size, dim=1)[rank], atol=2e-6, rtol=2e-5)
    for (_, parameter), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-6, rtol=2e-5)
    dist.destroy_process_group()
