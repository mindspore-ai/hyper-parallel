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
"""Distributed NPU worker tests for Expert Parallelism (torchrun, hccl).

Launched from ``test_ep_distributed.py`` via ``parallel_run``.  Each test
function verifies that the EP distributed computation produces numerically
identical results to the standalone (single-rank, all-experts-local) baseline.

Test strategy:
  - All ranks create the same model using an identical seed.
  - All ranks create the same input tensor using an identical seed.
  - Standalone: each rank runs the full model independently (same result everywhere).
  - EP distributed: experts are sharded via ``ExpertParallel``; tokens are
    routed across ranks via differentiable all-to-all.
  - Both forward outputs and input gradients are compared with ``rtol=1e-3,
    atol=1e-3`` tolerances.
"""
from copy import deepcopy
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.platform.torch.common import FeedForward, MoE
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from tests.torch.utils import _DEVICE_TYPE, init_backend


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _build_standalone_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    use_grouped_mm: bool = False,
) -> MoE:
    """Create a MoE model with a fixed seed on *device*.

    Args:
        dim: Token embedding dimension.
        hidden_dim: Expert hidden dimension.
        num_experts: Total number of experts.
        top_k: Experts selected per token.
        device: Target device.
        use_grouped_mm: Whether to use grouped matmul kernel.

    Returns:
        Initialised MoE module on *device*.
    """
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k,
              use_grouped_mm=use_grouped_mm)
    moe = moe.to(device)
    return moe


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


# ---------------------------------------------------------------------------
# Test: EP forward / backward precision (4 ranks, 4 experts)
# ---------------------------------------------------------------------------

def test_ep_forward_backward_npu():
    """
    Feature: ExpertParallel forward and backward precision on NPU.
    Description:
        1. Build identical standalone MoE on all ranks (seed=42).
        2. Build identical EP MoE on all ranks; apply ExpertParallel sharding.
        3. Forward + backward with the same input (seed=100).
        4. EP output and input gradient must match standalone within tolerance.
    Expectation: Outputs and gradients match standalone within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one expert per rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = world_size  # one expert per rank
    top_k = 2

    # ---- Standalone reference ----
    standalone_moe = _build_standalone_moe(dim, hidden_dim, num_experts, top_k, device)

    torch.manual_seed(100)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(100)
    x_ref = torch.randn(bs, slen, dim, device=device)

    standalone_out, standalone_x_grad = _run_forward_backward(standalone_moe, x_ref)

    # ---- EP distributed ----
    ep_moe = _build_standalone_moe(dim, hidden_dim, num_experts, top_k, device)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(world_size,), mesh_dim_names=("ep",))
    ExpertParallel().apply(ep_moe.experts, mesh)

    ep_out, ep_x_grad = _run_forward_backward(ep_moe, x_ref)

    # ---- Compare ----
    _npu_precision_close(ep_out, standalone_out, label=f"rank{rank} forward output")
    _npu_precision_close(ep_x_grad, standalone_x_grad, label=f"rank{rank} input gradient")


# ---------------------------------------------------------------------------
# Test: EP with multiple experts per rank (8 experts, 4 ranks)
# ---------------------------------------------------------------------------

def test_ep_multi_experts_per_rank_npu():
    """
    Feature: ExpertParallel with multiple local experts per rank.
    Description:
        1. Build standalone MoE with 8 experts on all ranks (seed=44).
        2. Apply ExpertParallel on 4-rank mesh (2 experts per rank).
        3. Forward + backward with the same input (seed=102).
        4. EP output and gradient must match standalone.
    Expectation: Outputs and gradients match standalone within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 8 (two experts per rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = world_size * 2  # 2 experts per rank
    top_k = 2

    # ---- Standalone reference ----
    torch.manual_seed(44)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(44)
    standalone_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    standalone_moe = standalone_moe.to(device)

    torch.manual_seed(102)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(102)
    x_ref = torch.randn(bs, slen, dim, device=device)

    standalone_out, standalone_x_grad = _run_forward_backward(standalone_moe, x_ref)

    # ---- EP distributed (2 local experts per rank) ----
    torch.manual_seed(44)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(44)
    ep_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    ep_moe = ep_moe.to(device)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(world_size,), mesh_dim_names=("ep",))
    ExpertParallel().apply(ep_moe.experts, mesh)

    ep_out, ep_x_grad = _run_forward_backward(ep_moe, x_ref)

    # ---- Compare ----
    _npu_precision_close(ep_out, standalone_out, label=f"rank{rank} multi-expert forward output")
    _npu_precision_close(ep_x_grad, standalone_x_grad, label=f"rank{rank} multi-expert input gradient")


# ---------------------------------------------------------------------------
# Test: EP with npu_grouped_matmul kernel (4 ranks, 4 experts)
# ---------------------------------------------------------------------------

def test_ep_grouped_mm_npu():
    """
    Feature: ExpertParallel with npu_grouped_matmul kernel precision.
    Description:
        1. Build standalone MoE with use_grouped_mm=False (for-loop reference).
        2. Build EP MoE with use_grouped_mm=True (npu_grouped_matmul kernel).
        3. Apply ExpertParallel on 4-rank mesh.
        4. Forward + backward with the same input (seed=103).
        5. grouped_mm EP output and gradient must match for-loop reference.
    Expectation: Outputs and gradients match within rtol=1e-2, atol=1e-2.

    Note:
        npu_grouped_matmul has slightly lower precision than for-loop for
        bfloat16/float16; tolerances are relaxed to 1e-2.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one expert per rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = world_size
    top_k = 2

    # ---- for-loop reference (EP, no grouped_mm) ----
    torch.manual_seed(45)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(45)
    ref_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k,
                  use_grouped_mm=False)
    ref_moe = ref_moe.to(device)
    mesh_ref = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(world_size,), mesh_dim_names=("ep",))
    ExpertParallel().apply(ref_moe.experts, mesh_ref)

    torch.manual_seed(103)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(103)
    x_ref = torch.randn(bs, slen, dim, device=device)

    ref_out, ref_x_grad = _run_forward_backward(ref_moe, x_ref)

    # ---- grouped_mm EP ----
    torch.manual_seed(45)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(45)
    gmm_moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k,
                  use_grouped_mm=True)
    gmm_moe = gmm_moe.to(device)
    mesh_gmm = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(world_size,), mesh_dim_names=("ep",))
    ExpertParallel().apply(gmm_moe.experts, mesh_gmm)

    gmm_out, gmm_x_grad = _run_forward_backward(gmm_moe, x_ref)

    # ---- Compare (relaxed tolerance for grouped_mm) ----
    _npu_precision_close(
        gmm_out, ref_out, label=f"rank{rank} grouped_mm forward output", rtol=1e-2, atol=1e-2
    )
    _npu_precision_close(
        gmm_x_grad, ref_x_grad, label=f"rank{rank} grouped_mm input gradient", rtol=1e-2, atol=1e-2
    )


# ---------------------------------------------------------------------------
# Local-shard helpers
# ---------------------------------------------------------------------------

def _run_standalone_global(
    moe: MoE,
    x_global: torch.Tensor,
) -> tuple:
    """Run forward + backward on the **full** global token sequence.

    Args:
        moe: Standalone (single-device) MoE model.
        x_global: Global input of shape ``[bs, slen, dim]``.

    Returns:
        Tuple ``(output, input_grad)`` where both tensors have shape
        ``[bs, slen, dim]``.
    """
    x_in = x_global.clone().requires_grad_(True)
    out = moe(x_in)
    out.sum().backward()
    return out.detach(), x_in.grad.clone()


def _run_ep_local_shard(
    moe: MoE,
    x_global: torch.Tensor,
    rank: int,
    ep_degree: int,
    mesh,
) -> tuple:
    """Run forward + backward on this rank's **local** token slice.

    The local slice is obtained via ``DTensor.from_local(..., [Shard(1)]).to_local()``
    to validate the DTensor sharding pipeline.  The EP MoE output is
    ``[bs, local_slen, dim]`` — no final AllGather since each rank only
    owns its own output tokens.

    Args:
        moe: EP-sharded MoE model (experts partitioned via ExpertParallel).
        x_global: Global input of shape ``[bs, slen, dim]``.  All ranks must
            hold the same tensor (generated with an identical seed).
        rank: Current rank index.
        ep_degree: EP group size (== number of ranks in the EP mesh).
        mesh: 1-D EP DeviceMesh.

    Returns:
        Tuple ``(output, input_grad)`` where both tensors have shape
        ``[bs, local_slen, dim]``.

    Raises:
        ValueError: If ``slen`` is not evenly divisible by ``ep_degree``.
    """
    slen = x_global.shape[1]
    if slen % ep_degree != 0:
        raise ValueError(
            f"slen={slen} is not divisible by ep_degree={ep_degree}"
        )
    local_slen = slen // ep_degree
    start = rank * local_slen
    x_slice = x_global[:, start:start + local_slen, :].contiguous()

    # Wrap as DTensor Shard(1) then extract local shard — validates the
    # DTensor sharding pipeline used at the EP MoE boundary.
    x_dt = DTensor.from_local(x_slice, mesh, [Shard(1)])
    x_local = x_dt.to_local()

    x_in = x_local.clone().requires_grad_(True)
    out = moe(x_in)
    out.sum().backward()
    return out.detach(), x_in.grad.clone()


def _run_local_shard_ep(
    rank: int,
    world_size: int,
    device: torch.device,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    bs: int,
    slen: int,
    seed_model: int,
    seed_input: int,
    use_grouped_mm: bool = False,
    shared_expert_hidden: int = 0,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Core logic shared by local-shard EP precision tests.

    Comparison strategy:

    - **Standalone reference**: runs on the **full** global sequence
      ``x_global`` of shape ``[bs, slen, dim]``.  Output and gradient have
      the same shape.
    - **EP distributed**: each rank receives its own local slice via
      ``DTensor.from_local(x_slice, mesh, [Shard(1)]).to_local()``, passes
      it to the EP MoE, and produces output of shape ``[bs, local_slen, dim]``
      (no final AllGather — each rank owns only its output tokens).
    - **Comparison**: EP output ≈ ``standalone_out[:, start:end, :]`` and
      EP input gradient ≈ ``standalone_x_grad[:, start:end, :]``.

    Gradient equivalence holds because MoE has no cross-token dependencies:
    ``d(loss_global)/d(x_i) == d(loss_local)/d(x_i)`` for each token i in
    the local slice, where ``loss = out.sum()``.

    Args:
        rank: Current rank.
        world_size: EP group size (== num_proc).
        device: Target device.
        dim: Token embedding dimension.
        hidden_dim: Expert hidden dimension.
        num_experts: Total number of experts.
        top_k: Experts selected per token.
        bs: Batch size.
        slen: Global sequence length (must be divisible by world_size).
        seed_model: Random seed for model initialization.
        seed_input: Random seed for input generation (same on all ranks).
        use_grouped_mm: Enable grouped matmul kernel in EP model.
        shared_expert_hidden: If > 0, add a shared FeedForward with this
            hidden dimension.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
    """
    local_slen = slen // world_size
    start = rank * local_slen

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(world_size,), mesh_dim_names=("ep",))

    # Build standalone and EP models with identical weights (same seed).
    torch.manual_seed(seed_model)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(seed_model)
    shared_ref = FeedForward(dim=dim, hidden_dim=shared_expert_hidden) if shared_expert_hidden > 0 else None
    standalone_moe = MoE(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
        top_k=top_k, shared_expert=deepcopy(shared_ref) if shared_ref else None,
    ).to(device)

    torch.manual_seed(seed_model)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(seed_model)
    shared_ep = FeedForward(dim=dim, hidden_dim=shared_expert_hidden) if shared_expert_hidden > 0 else None
    ep_moe = MoE(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
        top_k=top_k, use_grouped_mm=use_grouped_mm,
        shared_expert=deepcopy(shared_ep) if shared_ep else None,
    ).to(device)
    ExpertParallel().apply(ep_moe.experts, mesh)

    # Global input — identical on all ranks (same seed).
    torch.manual_seed(seed_input)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(seed_input)
    x_global = torch.randn(bs, slen, dim, device=device)

    # Standalone reference: full global sequence → [bs, slen, dim] output.
    standalone_out, standalone_x_grad = _run_standalone_global(standalone_moe, x_global)

    # EP: local slice via DTensor → [bs, local_slen, dim] output.
    ep_out, ep_x_grad = _run_ep_local_shard(ep_moe, x_global, rank, world_size, mesh)

    # Extract the corresponding local slice from standalone results.
    ref_out_local = standalone_out[:, start:start + local_slen, :].contiguous()
    ref_grad_local = standalone_x_grad[:, start:start + local_slen, :].contiguous()

    _npu_precision_close(
        ep_out, ref_out_local,
        label=f"rank{rank} local-shard forward output",
        rtol=rtol, atol=atol,
    )
    _npu_precision_close(
        ep_x_grad, ref_grad_local,
        label=f"rank{rank} local-shard input gradient",
        rtol=rtol, atol=atol,
    )


# ---------------------------------------------------------------------------
# Test: EP local-shard forward/backward (4 ranks, 4 experts, top_k=2)
# ---------------------------------------------------------------------------

def test_ep_local_shard_forward_backward_npu():
    """
    Feature: ExpertParallel with local-shard token input, forward/backward precision.
    Description:
        1. Standalone MoE runs on full global sequence (bs=4, slen=16).
        2. EP MoE receives each rank's local slice (slen=4) via DTensor Shard(1).
        3. EP output [bs, 4, dim] must match standalone_out[:, rank*4:(rank+1)*4, :].
        4. EP input gradient must match the corresponding slice of standalone gradient.
    Expectation: Outputs and gradients match within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one expert per rank)
        - top_k: 2, dim: 64, hidden_dim: 128
        - batch_size: 4, slen: 16 (local_slen=4 per rank)
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)
    _run_local_shard_ep(
        rank=rank, world_size=world_size, device=device,
        dim=64, hidden_dim=128, num_experts=world_size, top_k=2,
        bs=4, slen=16, seed_model=50, seed_input=200,
    )


# ---------------------------------------------------------------------------
# Test: EP local-shard multi-expert per rank (4 ranks, 8 experts, top_k=2)
# ---------------------------------------------------------------------------

def test_ep_local_shard_multi_expert_npu():
    """
    Feature: ExpertParallel local-shard input with multiple experts per rank.
    Description:
        1. Standalone MoE runs on full global sequence (slen=16, 8 experts).
        2. EP MoE receives each rank's local slice (slen=4) via DTensor Shard(1).
        3. EP output must match the corresponding slice of standalone output.
        4. EP gradient must match the corresponding slice of standalone gradient.
    Expectation: Outputs and gradients match within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 8 (two experts per rank)
        - top_k: 2, dim: 64, hidden_dim: 128
        - batch_size: 4, slen: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)
    _run_local_shard_ep(
        rank=rank, world_size=world_size, device=device,
        dim=64, hidden_dim=128, num_experts=world_size * 2, top_k=2,
        bs=4, slen=16, seed_model=51, seed_input=201,
    )


# ---------------------------------------------------------------------------
# Test: EP local-shard with grouped_mm (4 ranks, 4 experts, top_k=2)
# ---------------------------------------------------------------------------

def test_ep_local_shard_grouped_mm_npu():
    """
    Feature: ExpertParallel local-shard input with npu_grouped_matmul.
    Description:
        1. Standalone for-loop MoE runs on the full global sequence (reference).
        2. grouped_mm EP receives each rank's local slice via DTensor Shard(1).
        3. EP output must match the corresponding slice of standalone output
           within relaxed tolerance (grouped_mm has lower numerical precision).
    Expectation: Outputs and gradients match within rtol=1e-2, atol=1e-2.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one per rank)
        - top_k: 2, dim: 64, hidden_dim: 128
        - batch_size: 4, slen: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)
    _run_local_shard_ep(
        rank=rank, world_size=world_size, device=device,
        dim=64, hidden_dim=128, num_experts=world_size, top_k=2,
        bs=4, slen=16, seed_model=52, seed_input=202,
        use_grouped_mm=True, rtol=1e-2, atol=1e-2,
    )


# ---------------------------------------------------------------------------
# Test: EP local-shard top_k=1 (4 ranks, 4 experts)
# ---------------------------------------------------------------------------

def test_ep_local_shard_top1_npu():
    """
    Feature: ExpertParallel local-shard input with top_k=1.
    Description:
        1. Standalone MoE runs on full global sequence (top_k=1).
        2. EP MoE receives each rank's local slice via DTensor Shard(1).
        3. EP output must match the corresponding slice of standalone output.
        4. EP gradient must match the corresponding slice of standalone gradient.
    Expectation: Outputs and gradients match within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one per rank), top_k: 1
        - dim: 64, hidden_dim: 128, batch_size: 4, slen: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)
    _run_local_shard_ep(
        rank=rank, world_size=world_size, device=device,
        dim=64, hidden_dim=128, num_experts=world_size, top_k=1,
        bs=4, slen=16, seed_model=53, seed_input=203,
    )


# ---------------------------------------------------------------------------
# Test: EP local-shard with shared expert (4 ranks, 4 experts, top_k=2)
# ---------------------------------------------------------------------------

def test_ep_local_shard_shared_expert_npu():
    """
    Feature: ExpertParallel local-shard input with a shared (always-active) expert.
    Description:
        1. Standalone MoE (4 experts + shared expert) runs on full global sequence.
        2. EP MoE receives each rank's local slice via DTensor Shard(1).
        3. EP output must match the corresponding slice of standalone output.
        4. EP gradient must match the corresponding slice of standalone gradient.
    Expectation: Outputs and gradients match within rtol=1e-3, atol=1e-3.

    Configuration:
        - num_proc: 4
        - num_experts: 4 (one per rank), top_k: 2
        - shared_expert hidden_dim: 128
        - dim: 64, hidden_dim: 128, batch_size: 4, slen: 16
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(_DEVICE_TYPE)
    _run_local_shard_ep(
        rank=rank, world_size=world_size, device=device,
        dim=64, hidden_dim=128, num_experts=world_size, top_k=2,
        bs=4, slen=16, seed_model=54, seed_input=204,
        shared_expert_hidden=128,
    )
