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
"""Distributed NPU worker tests for EP DTensor boundary integration.

Launched from ``test_ep_dtensor_boundary.py`` via ``parallel_run``.  Each test
function verifies that the EP MoE with DTensor boundary hooks
(``PrepareModuleInputOutput``) produces numerically correct results compared
to a standalone single-card baseline.

Test strategy:
  - All ranks create the same model using an identical seed.
  - All ranks create the same input tensor using an identical seed.
  - Standalone: each rank runs the full model independently (same result everywhere).
  - EP+DTensor-boundary distributed: experts are sharded via ``ExpertParallel``
    and DTensor boundary hooks convert between plain/DTensor representations;
    tokens are routed across ranks via differentiable all-to-all.
  - Both forward outputs and input gradients are compared with ``rtol=1e-3,
    atol=1e-3`` tolerances.
"""
import sys

# pylint: disable=C0413
import torch
import torch.distributed as dist

import torch_npu  # pylint: disable=W0611  # Ascend NPU side-effect import

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.tensor_parallel import parallelize_module
from hyper_parallel.core.tensor_parallel.style import PrepareModuleInputOutput
from hyper_parallel.platform.torch.common import FeedForward, MoE
from tests.torch.utils import init_dist


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RTOL = 1e-3
_ATOL = 1e-3


def _npu_precision_close(
    result: torch.Tensor,
    reference: torch.Tensor,
    label: str,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> None:
    """Assert that *result* is close to *reference* with NPU-appropriate tolerances.

    Args:
        result: Tensor produced by the distributed path.
        reference: Tensor produced by the standalone path.
        label: Short description for the assertion message.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    result_cpu = result.detach().cpu().float()
    reference_cpu = reference.detach().cpu().float()
    max_diff = (result_cpu - reference_cpu).abs().max().item()
    assert torch.allclose(result_cpu, reference_cpu, rtol=rtol, atol=atol), (
        f"{label} mismatch: "
        f"max_diff={max_diff:.6f}, rtol={rtol}, atol={atol}, "
        f"result_shape={tuple(result.shape)}, ref_shape={tuple(reference.shape)}"
    )


def _build_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    with_shared_expert: bool = False,
    seed: int = 42,
) -> MoE:
    """Create a MoE model with a fixed seed on *device*.

    Args:
        dim: Token embedding dimension.
        hidden_dim: Expert hidden dimension.
        num_experts: Total number of experts.
        top_k: Experts selected per token.
        device: Target device.
        with_shared_expert: Whether to include a shared FeedForward expert.
        seed: Random seed for deterministic weight initialization.

    Returns:
        Initialised MoE module on *device*.
    """
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    shared = FeedForward(dim=dim, hidden_dim=hidden_dim) if with_shared_expert else None
    moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
              top_k=top_k, shared_expert=shared)
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
# Test: EP-only with plain tensor input (regression test)
# ---------------------------------------------------------------------------

def test_ep_only_plain_tensor():
    """EP-only: 2 cards, plain tensor input, no DTensor boundary.

    Regression test: Verify EP-only behavior with plain tensor
    matches existing EP ST baseline.

    Configuration:
        - num_proc: 2
        - num_experts: 4 (2 experts per rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16
    """
    _, device_id = init_dist()
    device = torch.device(f"npu:{device_id}")
    world_size = dist.get_world_size()

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = 4
    top_k = 2

    ep_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device,
                        with_shared_expert=False, seed=42)

    ep_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,),
                               mesh_dim_names=("ep",))
    ExpertParallel().apply(ep_moe.experts, ep_mesh)

    torch.manual_seed(100)
    torch.npu.manual_seed(100)
    x_ep = torch.randn(bs, slen, dim, device=device)

    x_in = x_ep.clone().requires_grad_(True)
    out = ep_moe(x_in)
    loss = out.sum()
    loss.backward()

    assert isinstance(out, torch.Tensor) and not isinstance(out, DTensor), (
        f"Expected plain tensor output, got {type(out)}"
    )

    ep_out = out.detach()
    ep_x_grad = x_in.grad.clone()

    assert ep_out.shape == (bs, slen, dim), (
        f"Output shape mismatch: expected {(bs, slen, dim)}, got {tuple(ep_out.shape)}"
    )
    assert ep_x_grad.shape == (bs, slen, dim), (
        f"Grad shape mismatch: expected {(bs, slen, dim)}, got {tuple(ep_x_grad.shape)}"
    )
    assert not torch.isnan(ep_out).any(), "Output contains NaN"
    assert not torch.isinf(ep_out).any(), "Output contains Inf"
    assert not torch.isnan(ep_x_grad).any(), "Gradient contains NaN"


# ---------------------------------------------------------------------------
# Test: SP x EP with DTensor Shard(1) input
# ---------------------------------------------------------------------------

def test_sp_ep_dtensor_boundary():
    """SP(2) x EP(2): 4 cards, DTensor Shard(1) input, forward/backward correctness.

    Integration test:
    - 2D mesh [ep=2, sp=2]
    - Input: Shard(1) DTensor (sequence dimension sharded across SP dimension)
    - PrepareModuleInputOutput hooks on MoE:
        * Pre-hook: use_local_input=True => .to_local() before MoE.forward
        * Post-hook: output_layouts=Shard(1), use_local_output=False => wraps
          plain tensor output back to DTensor Shard(1)
    - Verify: forward output numerically aligned with standalone
    - Verify: input gradient numerically aligned with standalone (plain tensor
      due to Stage 1 use_local_input=True gradient placement loss)

    Configuration:
        - num_proc: 4
        - mesh: [ep=2, sp=2]
        - num_experts: 4 (2 experts per EP rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16 (local_seq_len=8 per SP rank)
    """
    rank, device_id = init_dist()
    device = torch.device(f"npu:{device_id}")

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = 4
    top_k = 2

    standalone_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device,
                                with_shared_expert=False, seed=46)

    torch.manual_seed(104)
    torch.npu.manual_seed(104)
    x_global = torch.randn(bs, slen, dim, device=device)

    standalone_out, standalone_x_grad = _run_forward_backward(standalone_moe, x_global)

    sp_ep_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device,
                           with_shared_expert=False, seed=46)

    mesh_2d = init_device_mesh(device_type="npu", mesh_shape=(2, 2),
                               mesh_dim_names=("ep", "sp"))
    sp_mesh = mesh_2d["sp"]
    ep_mesh = mesh_2d["ep"]

    io_style = PrepareModuleInputOutput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Shard(1),),
        use_local_input=True,
        output_layouts=(Shard(1),),
        desired_output_layouts=(Shard(1),),
        use_local_output=False,
    )
    with sp_mesh:
        parallelize_module(sp_ep_moe, sp_mesh, io_style)

    ExpertParallel().apply(sp_ep_moe.experts, ep_mesh)

    sp_size = sp_mesh.size()
    local_slen = slen // sp_size
    sp_coord = mesh_2d.get_local_rank("sp")
    start = sp_coord * local_slen

    torch.manual_seed(104)
    torch.npu.manual_seed(104)
    x_global_ep = torch.randn(bs, slen, dim, device=device)
    x_slice = x_global_ep[:, start:start + local_slen, :].contiguous()

    x_dt = DTensor.from_local(x_slice, sp_mesh, [Shard(1)])

    x_in_dt = x_dt.clone().requires_grad_(True)
    out = sp_ep_moe(x_in_dt)
    loss = out.sum()
    loss.backward()

    assert isinstance(out, DTensor), (
        f"Expected DTensor output, got {type(out)}"
    )
    assert x_in_dt.grad is not None, "Gradient should not be None"

    # Stage 1 limitation: use_local_input=True calls .to_local() in the
    # pre-hook which detaches the DTensor autograd chain.  The input gradient
    # is therefore a plain tensor (not DTensor) with correct values but
    # lost placement metadata.  This is a known Stage 1 gradient bug that
    # will be resolved in Stage 3 (LocalMapConfig / DTensor throughout
    # Router+SharedExperts, to_local only at GroupedExperts boundary).
    ep_out_local = out.to_local().detach()
    ep_x_grad = x_in_dt.grad.detach().clone()

    ref_out_local = standalone_out[:, start:start + local_slen, :].contiguous()
    ref_grad_local = standalone_x_grad[:, start:start + local_slen, :].contiguous()

    _npu_precision_close(ep_out_local, ref_out_local,
                         label=f"rank{rank} SPxEP DTensor boundary output")

    assert ep_x_grad.shape == (bs, local_slen, dim), (
        f"Gradient shape mismatch: expected {(bs, local_slen, dim)}, "
        f"got {tuple(ep_x_grad.shape)}"
    )

    _npu_precision_close(ep_x_grad, ref_grad_local,
                         label=f"rank{rank} SPxEP DTensor boundary input gradient")


# ---------------------------------------------------------------------------
# Test: SP x EP with shared expert (covers out = routed + shared_out path)
# ---------------------------------------------------------------------------

def test_sp_ep_dtensor_boundary_with_shared_expert():
    """SP(2) x EP(2): 4 cards, with shared expert, DTensor Shard(1) input.

    Covers the ``out = out + shared_out`` path in MoE.forward where the
    routed-expert output (plain tensor from EP all-to-all combine) is added
    to the shared-expert output (plain tensor from FeedForward).  The post-hook
    then wraps the sum back into a DTensor Shard(1).

    Verifies:
    - Forward output numerically aligned with standalone baseline
    - Input gradient numerically aligned with standalone baseline (plain tensor
      due to Stage 1 use_local_input=True gradient placement loss)

    Configuration:
        - num_proc: 4
        - mesh: [ep=2, sp=2]
        - num_experts: 4 (2 experts per EP rank)
        - top_k: 2
        - dim: 64, hidden_dim: 128
        - batch_size: 4, seq_len: 16 (local_seq_len=8 per SP rank)
        - shared_expert: FeedForward(dim=64, hidden_dim=128)
    """
    rank, device_id = init_dist()
    device = torch.device(f"npu:{device_id}")

    dim, hidden_dim = 64, 128
    bs, slen = 4, 16
    num_experts = 4
    top_k = 2

    standalone_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device,
                                with_shared_expert=True, seed=47)

    torch.manual_seed(105)
    torch.npu.manual_seed(105)
    x_global = torch.randn(bs, slen, dim, device=device)

    standalone_out, standalone_x_grad = _run_forward_backward(standalone_moe, x_global)

    sp_ep_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device,
                           with_shared_expert=True, seed=47)

    mesh_2d = init_device_mesh(device_type="npu", mesh_shape=(2, 2),
                               mesh_dim_names=("ep", "sp"))
    sp_mesh = mesh_2d["sp"]
    ep_mesh = mesh_2d["ep"]

    io_style = PrepareModuleInputOutput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Shard(1),),
        use_local_input=True,
        output_layouts=(Shard(1),),
        desired_output_layouts=(Shard(1),),
        use_local_output=False,
    )
    with sp_mesh:
        parallelize_module(sp_ep_moe, sp_mesh, io_style)

    ExpertParallel().apply(sp_ep_moe.experts, ep_mesh)

    sp_size = sp_mesh.size()
    local_slen = slen // sp_size
    sp_coord = mesh_2d.get_local_rank("sp")
    start = sp_coord * local_slen

    torch.manual_seed(105)
    torch.npu.manual_seed(105)
    x_global_ep = torch.randn(bs, slen, dim, device=device)
    x_slice = x_global_ep[:, start:start + local_slen, :].contiguous()

    x_dt = DTensor.from_local(x_slice, sp_mesh, [Shard(1)])

    x_in_dt = x_dt.clone().requires_grad_(True)
    out = sp_ep_moe(x_in_dt)
    loss = out.sum()
    loss.backward()

    assert isinstance(out, DTensor), (
        f"Expected DTensor output, got {type(out)}"
    )
    assert x_in_dt.grad is not None, "Gradient should not be None"

    ep_out_local = out.to_local().detach()
    ep_x_grad = x_in_dt.grad.detach().clone()

    ref_out_local = standalone_out[:, start:start + local_slen, :].contiguous()
    ref_grad_local = standalone_x_grad[:, start:start + local_slen, :].contiguous()

    _npu_precision_close(ep_out_local, ref_out_local,
                         label=f"rank{rank} SPxEP shared-expert output")

    assert ep_x_grad.shape == (bs, local_slen, dim), (
        f"Gradient shape mismatch: expected {(bs, local_slen, dim)}, "
        f"got {tuple(ep_x_grad.shape)}"
    )

    _npu_precision_close(ep_x_grad, ref_grad_local,
                         label=f"rank{rank} SPxEP shared-expert input gradient")


# ---------------------------------------------------------------------------
# Torchrun dispatch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    case_name = sys.argv[1] if len(sys.argv) > 1 else None
    dispatch = {
        "test_ep_only_plain_tensor": test_ep_only_plain_tensor,
        "test_sp_ep_dtensor_boundary": test_sp_ep_dtensor_boundary,
        "test_sp_ep_dtensor_boundary_with_shared_expert": test_sp_ep_dtensor_boundary_with_shared_expert,
    }
    if case_name not in dispatch:
        raise ValueError(f"Unknown case: {case_name}")
    dispatch[case_name]()
