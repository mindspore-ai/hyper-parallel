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
"""Distributed mixed-CP forward/backward oracle; supports CPU/Gloo and NPU/HCCL.

Run with torchrun --nproc-per-node=8 this_file.py --backend hccl --gdn-backend triton.
CPU smoke: torchrun --nproc-per-node=2 this_file.py --backend gloo.
"""

import argparse
import copy
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.gated_delta_net import GatedDeltaNetP2PCP
from hyper_parallel.distributed.context_parallel.wrappers import sdpa_hf_cp_wrapper


class _GatedNorm(nn.Module):
    """Small gated RMSNorm with the same GDN interface."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states, gate):
        normalized = hidden_states.float() * torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * self.weight.float() * F.silu(gate.float())).to(hidden_states.dtype)


class _GDN(nn.Module):
    """Tiny GDN with fewer heads than CP ranks, including depthwise-conv halo."""

    def __init__(self, dim):
        super().__init__()
        self.num_k_heads = 1
        self.num_v_heads = 2
        self.head_k_dim = dim
        self.head_v_dim = dim
        self.key_dim = dim
        self.value_dim = 2 * dim
        self.conv_dim = 4 * dim
        self.in_proj_qkv = nn.Linear(16, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(16, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(16, 2, bias=False)
        self.in_proj_a = nn.Linear(16, 2, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, 4, padding=3, groups=self.conv_dim)
        self.A_log = nn.Parameter(torch.zeros(2))  # pylint: disable=invalid-name
        self.dt_bias = nn.Parameter(torch.zeros(2))
        self.norm = _GatedNorm(dim)
        self.out_proj = nn.Linear(self.value_dim, 16, bias=False)


class _Attention(nn.Module):
    """HF-style projections and SDPA with one KV head (no CP head divisibility)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.v_proj = nn.Linear(16, 8, bias=False)
        self.out_proj = nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states):
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, 2, 8).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, 1, 8).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, 1, 8).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key.expand_as(query), value.expand_as(query), is_causal=True)
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, 16))


class _SingletonMesh:
    """CP=1 metadata for the independent full-sequence eager reference."""

    ndim = 1

    @staticmethod
    def size():
        return 1

    @staticmethod
    def get_local_rank():
        return 0


def run_case(mesh, device, backend, recompute):
    """Compare mixed-CP outputs, input gradients and every parameter gradient."""
    torch.manual_seed(123)
    dtype = torch.bfloat16 if backend == "triton" else torch.float32
    gdn = _GDN(128 if backend == "triton" else 8).to(device=device, dtype=dtype)
    attention = _Attention().to(device=device, dtype=dtype)
    reference_gdn, reference_attention = copy.deepcopy(gdn), copy.deepcopy(attention)
    local_gdn = GatedDeltaNetP2PCP(gdn, mesh, backend=backend)
    full_gdn = GatedDeltaNetP2PCP(reference_gdn, _SingletonMesh(), backend="eager")
    attention.forward = sdpa_hf_cp_wrapper(attention, None, None, mesh, None)
    full_input = torch.randn(1, 64 * mesh.size(), 16, device=device, dtype=dtype, requires_grad=True)
    local_input = full_input.detach().chunk(mesh.size(), dim=1)[mesh.get_local_rank()].clone().requires_grad_()

    reference = reference_attention(full_input + full_gdn(full_input))
    if recompute:
        hidden = checkpoint(local_gdn, local_input, use_reentrant=False)
        actual = checkpoint(attention, local_input + hidden, use_reentrant=False)
    else:
        actual = attention(local_input + local_gdn(local_input))
    expected = reference.chunk(mesh.size(), dim=1)[mesh.get_local_rank()]
    tolerance = {"atol": 3e-2, "rtol": 3e-2} if backend == "triton" else {"atol": 2e-5, "rtol": 2e-4}
    torch.testing.assert_close(actual, expected, **tolerance)
    # Sum local objectives; do not divide by CP here. This tests the collective
    # Jacobians independently of the trainer's global token-loss normalization.
    reference.float().square().sum().backward()
    actual.float().square().sum().backward()
    torch.testing.assert_close(
        local_input.grad, full_input.grad.chunk(mesh.size(), dim=1)[mesh.get_local_rank()], **tolerance,
    )
    for local_module, reference_module in ((gdn, reference_gdn), (attention, reference_attention)):
        for (name, parameter), (_, reference_parameter) in zip(
            local_module.named_parameters(), reference_module.named_parameters(),
        ):
            if parameter.grad is None or reference_parameter.grad is None:
                raise AssertionError(f"Missing gradient for {name}")
            dist.all_reduce(parameter.grad, group=mesh.get_group())
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=lambda msg, name=name: f"{name}: {msg}",
                                       **tolerance)
    if dist.get_rank() == 0:
        print(f"PASS mixed CP={mesh.size()} backend={backend} recompute={recompute}", flush=True)


def run_distributed(communication_backend, gdn_backend):
    """Initialize workers and run ordinary and activation-checkpointed graphs."""
    torch.set_num_threads(1)
    if communication_backend == "hccl":
        import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
        # Optional hardware backend; CPU/Gloo validation must not require it.
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cpu")
    dist.init_process_group(communication_backend, timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("cp",))
        for recompute in (False, True):
            run_case(mesh, device, gdn_backend, recompute)
    finally:
        dist.destroy_process_group()


def test_mixed_cp_triton():
    """Eight-rank NPU forward/backward parity, including checkpoint recompute."""
    run_distributed("hccl", "triton")


def main():
    """Select the hardware/backend for standalone torchrun validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--gdn-backend", choices=("eager", "triton"), default="eager")
    args = parser.parse_args()
    run_distributed(args.backend, args.gdn_backend)


if __name__ == "__main__":
    main()
