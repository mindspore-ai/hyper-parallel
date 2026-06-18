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
"""Distributed NPU worker tests for FSDP + Expert Parallelism (torchrun, hccl).

Launched from ``test_fsdp_ep_distributed.py`` via ``parallel_run``.  Each test
function verifies that FSDP + EP distributed computation produces numerically
identical results to the standalone baseline.

Test strategy:
  - All ranks create the same model using an identical seed.
  - All ranks create the same input tensor using an identical seed.
  - Standalone: each rank runs the full model independently (same result everywhere).
  - FSDP + EP distributed: experts are sharded via ExpertParallel, parameters are
    further sharded via FSDP on the DP dimension.
  - Both forward outputs and input gradients are compared with ``rtol=1e-3,
    atol=1e-3`` tolerances.

Configuration for all tests:
  - 4 cards total: 2 FSDP ranks × 2 EP ranks (2D mesh [fsdp, ep])
  - num_experts: 4 (2 per EP rank)
  - top_k: 2
  - dim: 64, hidden_dim: 128
  - batch_size: 4, seq_len: 16
"""
import torch
import torch.distributed as dist

import torch_npu  # noqa: F401 — Ascend NPU

from hyper_parallel import init_device_mesh, fully_shard
from hyper_parallel.platform.torch.common import FeedForward, MoE
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from tests.torch.utils import init_dist


def _npu_precision_close(
    result: torch.Tensor,
    reference: torch.Tensor,
    label: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Assert that *result* is close to *reference* with NPU-appropriate tolerances.

    Args:
        result: Tensor produced by the distributed path.
        reference: Tensor produced by the standalone path.
        label: Short description for the assertion message.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    result_cpu = result.detach().cpu()
    reference_cpu = reference.detach().cpu()
    max_diff = (result_cpu - reference_cpu).abs().max().item()
    assert torch.allclose(result_cpu, reference_cpu, rtol=rtol, atol=atol), (
        f"{label} mismatch: "
        f"max_diff={max_diff:.6f}, rtol={rtol}, atol={atol}, "
        f"result_shape={tuple(result.shape)}, ref_shape={tuple(reference.shape)}"
    )


def _run_forward_backward(moe: MoE, x: torch.Tensor) -> tuple:
    """Run a forward + backward pass and return (output, input_grad).

    Args:
        moe: The MoE model.
        x: Input tensor (will have ``requires_grad=True`` set internally).

    Returns:
        Tuple of ``(output, input_grad)`` where *output* is the result of
        ``moe(x)`` and *input_grad* is the gradient of the loss w.r.t. *x*.
    """
    x_in = x.clone().requires_grad_(True)
    out = moe(x_in)
    loss = out.sum()
    loss.backward()
    return out.detach(), x_in.grad.clone()


def test_fsdp_ep_forward_backward_npu():
    """
    Feature: FSDP + EP forward and backward precision on NPU.
    Description:
        1. Build identical standalone MoE on all ranks (seed=42).
        2. Build identical FSDP + EP MoE on all ranks:
           - 2D mesh [fsdp=2, ep=2]
           - EP shards experts on EP dim (2 experts per EP rank)
           - FSDP shards parameters on FSDP dim
        3. Forward + backward with the same input (seed=100).
        4. FSDP + EP output and input gradient must match standalone within tolerance.
    Expectation: Outputs and gradients match standalone within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4 (2 FSDP × 2 EP)
        - num_experts: 4 (2 per EP rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    rank, device_id = init_dist()
    _ = dist.get_world_size()
    device = torch.device(f"npu:{device_id}")

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = 4
    top_k = 2

    fsdp_size = 2
    ep_size = 2

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(fsdp_size, ep_size),
        mesh_dim_names=("fsdp", "ep"),
    )

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    standalone_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    standalone_moe = standalone_moe.to(device)

    torch.manual_seed(100)
    torch.npu.manual_seed(100)
    x_ref = torch.randn(bs, slen, dim, device=device)

    standalone_out, standalone_x_grad = _run_forward_backward(standalone_moe, x_ref)

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    fsdp_ep_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    fsdp_ep_moe = fsdp_ep_moe.to(device)

    ExpertParallel().apply(fsdp_ep_moe.experts, mesh["ep"])

    fully_shard(fsdp_ep_moe, mesh=mesh["fsdp"])

    fsdp_ep_out, fsdp_ep_x_grad = _run_forward_backward(fsdp_ep_moe, x_ref)

    _npu_precision_close(fsdp_ep_out, standalone_out, label=f"rank{rank} FSDP+EP forward output")
    _npu_precision_close(fsdp_ep_x_grad, standalone_x_grad, label=f"rank{rank} FSDP+EP input gradient")


def test_fsdp_ep_three_step_training_npu():
    """
    Feature: FSDP + EP 3-step training convergence.
    Description:
        1. Build standalone MoE and FSDP+EP MoE with identical seeds.
        2. Train both for 3 steps on the same input batch.
        3. Verify FSDP+EP loss matches standalone loss at every step.
    Expectation: FSDP+EP loss matches standalone loss within rtol=1e-3, atol=1e-3.

    Note:
        Two sources inflate the FSDP+EP gradient relative to standalone:
          * EP AlltoAllV duplicates each token ep_size times so every expert
            processes ep_size × more tokens (and accumulates ep_size × gradient).
          * fully_shard's default reduce-op is SUM (not AVG) whenever any managed
            parameter is a DTensor, which is the case here because ExpertParallel
            converts expert weights to DTensors.  reduce_scatter on the FSDP dim
            therefore sums (not averages) across fsdp_size replicas.
        The combined factor is fsdp_size × ep_size = world_size, so dividing the
        backward loss by world_size matches the standalone loss trajectory.

    Configuration:
        - num_proc: 4 (2 FSDP × 2 EP)
        - num_experts: 4 (2 per EP rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    rank, device_id = init_dist()
    device = torch.device(f"npu:{device_id}")

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = 4
    top_k = 2

    fsdp_size = 2
    ep_size = 2
    grad_scale = fsdp_size * ep_size

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(fsdp_size, ep_size),
        mesh_dim_names=("fsdp", "ep"),
    )

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    standalone_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    standalone_moe = standalone_moe.to(device)
    standalone_optimizer = torch.optim.SGD(standalone_moe.parameters(), lr=0.01)

    torch.manual_seed(100)
    torch.npu.manual_seed(100)
    x = torch.randn(bs, slen, dim, device=device)

    standalone_losses = []
    for _ in range(3):
        standalone_optimizer.zero_grad()
        out = standalone_moe(x)
        loss = out.sum()
        loss.backward()
        standalone_optimizer.step()
        standalone_losses.append(loss.item())

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    fsdp_ep_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    fsdp_ep_moe = fsdp_ep_moe.to(device)

    ExpertParallel().apply(fsdp_ep_moe.experts, mesh["ep"])

    fully_shard(fsdp_ep_moe, mesh=mesh["fsdp"])

    fsdp_optimizer = torch.optim.SGD(fsdp_ep_moe.parameters(), lr=0.01)

    fsdp_losses = []
    for _ in range(3):
        fsdp_optimizer.zero_grad()
        out = fsdp_ep_moe(x)
        loss = out.sum()
        (loss / grad_scale).backward()
        fsdp_optimizer.step()
        fsdp_losses.append(loss.item())

    for i, (sl, fl) in enumerate(zip(standalone_losses, fsdp_losses)):
        _npu_precision_close(
            torch.tensor(fl), torch.tensor(sl),
            label=f"rank{rank} step{i} FSDP+EP vs standalone loss",
        )

    dist.barrier()
