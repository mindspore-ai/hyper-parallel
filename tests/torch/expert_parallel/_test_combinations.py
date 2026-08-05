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
"""Distributed worker for combined strategy tests."""

import os

import torch
import torch.distributed as dist
from torch import nn

# pylint: disable=wrong-import-position
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
rank = os.environ.get("RANK")
if rank is not None:
    os.environ["RANK_ID"] = rank
local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None:
    os.environ["DEVICE_ID"] = local_rank
world_size = os.environ.get("WORLD_SIZE")
if world_size is not None:
    os.environ["RANK_SIZE"] = world_size

from hyper_parallel import ContextParallel, fully_shard, init_device_mesh
from hyper_parallel.core.expert_parallel import (
    ExpertParallel,
    TensorParallel,
    ExpertTensorParallel,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.platform.torch.common import FeedForward, MoE
from tests.torch.expert_parallel.templates import get_template
from tests.torch.expert_parallel.validator import (
    validate_template,
    validate_mesh_dimensions,
    validate_moe_compatibility,
)
from tests.torch.expert_parallel._test_ep_distributed import (
    _npu_precision_close,
    _run_forward_backward,
)
from tests.torch.utils import _DEVICE_TYPE, init_backend

# Common parameter set for EP-only tests (reused by base/grouped_mm/shared)
EP_ONLY_PARAMS = [
    (2, 1, 64, 128, 2, 8),
    (2, 2, 64, 128, 2, 8),
    (4, 1, 64, 128, 4, 16),
    (4, 2, 64, 128, 4, 16),
    (8, 1, 64, 128, 8, 32),
    (8, 2, 64, 128, 8, 32),
]

TP_ONLY_PARAMS = [
    (2, 1, 64, 64, 2, 8),
    (2, 2, 64, 64, 2, 8),
    (2, 1, 64, 128, 2, 8),
    (2, 2, 64, 128, 2, 8),
    (4, 1, 64, 64, 4, 16),
    (4, 2, 64, 64, 4, 16),
    (4, 1, 64, 128, 4, 16),
    (4, 2, 64, 128, 4, 16),
]

MULTI_CARD_CONFIGS = {
    "dp-ep": {"template": "dp-ep", "params": (4, 2, 64, 128, 4, 16)},
    "ep-tp": {"template": "ep-tp", "params": (4, 2, 64, 128, 4, 16)},
    "dp-ep-tp": {"template": "dp-ep-tp", "params": (8, 2, 64, 128, 8, 32)},
    "dp-ep-cp": {"template": "dp-ep-cp", "params": (8, 2, 64, 128, 8, 32)},
}


def _select_orthogonal_fsdp_placement(param: torch.Tensor) -> Shard:
    """Choose an FSDP shard dimension not already occupied by EP or TP."""
    sharded_dims = {
        placement.dim
        for placement in getattr(param, "placements", ())
        if placement.is_shard()
    }
    for shard_dim in range(param.ndim):
        if shard_dim not in sharded_dims:
            return Shard(shard_dim)
    raise ValueError(
        "DP+EP+TP combination test requires one parameter dimension that is "
        f"not already sharded, but got placements={getattr(param, 'placements', ())}"
    )


def _run_moe_test(
    template_name: str,
    num_experts: int,
    top_k: int,
    dim: int,
    hidden_dim: int,
    bs: int,
    slen: int,
    use_grouped_mm: bool = False,
    shared_expert: bool = False,
    check_gradient: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Run a single MoE test with configurable tolerance and gradient checks."""
    rank_rank = dist.get_rank()
    rank_world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(_DEVICE_TYPE)

    config = get_template(template_name)
    dp, ep, tp, cp = config["dp"], config["ep"], config["tp"], config["cp"]

    validate_template(template_name, rank_world_size, num_experts, hidden_dim)

    # Build mesh
    mesh_names = []
    mesh_shape = []
    for name, dimval in [("dp", dp), ("ep", ep), ("tp", tp), ("cp", cp)]:
        if dimval > 1:
            mesh_names.append(name)
            mesh_shape.append(dimval)
    if not mesh_shape:
        mesh_shape = [1]
        mesh_names = ["dp"]
    mesh = init_device_mesh(
        _DEVICE_TYPE, tuple(mesh_shape), mesh_dim_names=tuple(mesh_names)
    )

    # Standalone MoE
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    shared_ref = FeedForward(dim=dim, hidden_dim=hidden_dim) if shared_expert else None
    standalone_moe = MoE(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_grouped_mm=False,
        shared_expert=shared_ref,
    ).to(device)

    # Parallel MoE
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    shared_ep = FeedForward(dim=dim, hidden_dim=hidden_dim) if shared_expert else None
    parallel_moe = MoE(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_grouped_mm=use_grouped_mm,
        shared_expert=shared_ep,
    ).to(device)

    # Apply parallelism based on template
    if template_name == "ep-only":
        ExpertParallel().apply(parallel_moe.experts, mesh["ep"])
    elif template_name == "tp-only":
        TensorParallel().apply(parallel_moe.experts, mesh["tp"])
    elif template_name in ["dp-ep", "ep-tp", "dp-ep-tp", "dp-ep-cp"]:
        if tp > 1:
            ExpertTensorParallel().apply(parallel_moe.experts, mesh["ep", "tp"])
        else:
            ExpertParallel().apply(parallel_moe.experts, mesh["ep"])
        if dp > 1:
            shard_placement_fn = _select_orthogonal_fsdp_placement if tp > 1 else None
            fully_shard(
                parallel_moe,
                mesh=mesh["dp"],
                shard_placement_fn=shard_placement_fn,
            )
            parallel_moe.set_reduce_op_type("sum")

    # Input
    torch.manual_seed(43)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(43)
    x = torch.randn(bs, slen, dim, device=device)

    if check_gradient:
        standalone_out, standalone_grad = _run_forward_backward(standalone_moe, x)
        x_in = x.clone().requires_grad_(True)
        out = parallel_moe(x_in)
        out.sum().backward()
        parallel_out = out.detach()
        parallel_grad = x_in.grad.clone()
    else:
        with torch.no_grad():
            standalone_out = standalone_moe(x).detach()
            parallel_out = parallel_moe(x).detach()

    _npu_precision_close(
        parallel_out,
        standalone_out,
        label=f"rank{rank_rank} {template_name} forward",
        rtol=rtol, atol=atol,
    )
    if check_gradient:
        _npu_precision_close(
            parallel_grad,
            standalone_grad,
            label=f"rank{rank_rank} {template_name} gradient",
            rtol=rtol, atol=atol,
        )


def _validate_slen_divisible_by_cp(slen: int, cp: int) -> None:
    """Validate that sequence length is divisible by CP degree."""
    if slen % cp != 0:
        raise ValueError(
            f"slen={slen} must be divisible by cp={cp} for CP attention tests. "
            f"Please adjust slen to a multiple of {cp}."
        )


class _CoreAttention(nn.Module):
    """Scaled dot-product attention over tensors in BSHD layout."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention while preserving the BSHD layout."""
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scale = query.shape[-1] ** -0.5
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.matmul(probabilities, value).transpose(1, 2)


class _Attention(nn.Module):
    """Self-attention with an exposed QKV core for ContextParallel hooks."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Initialize projections and the core attention module."""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.core_attn = _CoreAttention()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project inputs, run core attention, and merge attention heads."""
        batch_size, seq_len, dim = inputs.shape
        qkv_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        query = self.query_proj(inputs).reshape(qkv_shape)
        key = self.key_proj(inputs).reshape(qkv_shape)
        value = self.value_proj(inputs).reshape(qkv_shape)
        output = self.core_attn(query, key, value)
        return self.output_proj(output.reshape(batch_size, seq_len, dim))


class _AttentionMoEBlock(nn.Module):
    """Self-attention followed by a Mixture-of-Experts layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        use_grouped_mm: bool,
    ) -> None:
        """Initialize the attention and MoE submodules."""
        super().__init__()
        self.attn = _Attention(dim, num_heads)
        self.moe = MoE(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            use_grouped_mm=use_grouped_mm,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run self-attention followed by the MoE layer."""
        return self.moe(self.attn(inputs))


def _build_attention_moe_block(
    dim: int,
    num_heads: int,
    num_experts: int,
    top_k: int,
    hidden_dim: int,
    device: torch.device,
    use_grouped_mm: bool = False,
) -> _AttentionMoEBlock:
    """Build a block with Self-Attention + MoE for CP testing."""
    return _AttentionMoEBlock(
        dim=dim,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=hidden_dim,
        use_grouped_mm=use_grouped_mm,
    ).to(device)


def _run_cp_attention_test(
    template_name: str,
    num_experts: int,
    top_k: int,
    dim: int,
    num_heads: int,
    hidden_dim: int,
    bs: int,
    slen: int,
    cp: int,
    use_grouped_mm: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Run CP+Attention test: validates CP communication correctness."""
    # Explicit validation: slen must be divisible by cp
    _validate_slen_divisible_by_cp(slen, cp)

    rank_rank = dist.get_rank()
    rank_world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(_DEVICE_TYPE)

    config = get_template(template_name)
    dp, ep, _, cp_dim = config["dp"], config["ep"], config["tp"], config["cp"]

    if cp_dim != cp:
        raise ValueError(f"Template cp={cp_dim} does not match provided cp={cp}")
    if dp * ep * cp != rank_world_size:
        raise ValueError(
            f"dp*ep*cp = {dp}*{ep}*{cp} = {dp * ep * cp} "
            f"does not match world_size={rank_world_size}"
        )

    mesh = init_device_mesh(
        _DEVICE_TYPE,
        (dp, ep, cp),
        mesh_dim_names=("dp", "ep", "cp"),
    )

    # Standalone Block
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    standalone_block = _build_attention_moe_block(
        dim, num_heads, num_experts, top_k, hidden_dim, device, use_grouped_mm
    )

    # Parallel Block
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    parallel_block = _build_attention_moe_block(
        dim, num_heads, num_experts, top_k, hidden_dim, device, use_grouped_mm
    )
    ContextParallel(
        seq_dim=1,
        head_dim=2,
        ulysses_degree=1,
        use_local_output=True,
    ).apply(parallel_block.attn.core_attn, mesh["cp"])
    ExpertParallel().apply(parallel_block.moe.experts, mesh["ep"])
    if dp > 1:
        fully_shard(parallel_block, mesh=mesh["dp"])
        parallel_block.set_reduce_op_type("sum")

    # Input
    torch.manual_seed(43)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(43)
    x_global = torch.randn(bs, slen, dim, device=device)

    # Standalone: full sequence
    x_ref = x_global.clone().requires_grad_(True)
    standalone_out = standalone_block(x_ref)
    standalone_loss = standalone_out.sum()
    standalone_loss.backward()
    standalone_grad = x_ref.grad.clone()

    # CP parallel: local slice
    local_slen = slen // cp
    cp_rank = mesh.get_local_rank("cp")
    start = cp_rank * local_slen
    x_slice = x_global[:, start:start + local_slen, :].contiguous()

    x_in = x_slice.clone().requires_grad_(True)
    parallel_out = parallel_block(x_in)
    parallel_loss = parallel_out.sum()
    parallel_loss.backward()
    parallel_grad = x_in.grad.clone()

    # Compare with standalone slice
    ref_out_local = standalone_out[:, start:start + local_slen, :].contiguous()
    ref_grad_local = standalone_grad[:, start:start + local_slen, :].contiguous()

    _npu_precision_close(
        parallel_out,
        ref_out_local,
        label=f"rank{rank_rank} CP+Attention forward (slen={slen}, cp={cp})",
        rtol=rtol, atol=atol,
    )
    _npu_precision_close(
        parallel_grad,
        ref_grad_local,
        label=f"rank{rank_rank} CP+Attention gradient (slen={slen}, cp={cp})",
        rtol=rtol, atol=atol,
    )


def _run_validation_tests() -> None:
    """Run validation function checks (no distributed communication)."""
    validate_mesh_dimensions(1, 2, 1, 1, 2)
    validate_mesh_dimensions(1, 2, 2, 1, 4)
    validate_moe_compatibility(4, 2, 128, 2)
    validate_moe_compatibility(8, 2, 64, 2)

    invalid_mesh = [(1, 3, 1, 1, 2), (3, 2, 1, 1, 6)]
    for case in invalid_mesh:
        try:
            validate_mesh_dimensions(*case)
            raise AssertionError(f"Expected ValueError for mesh case: {case}")
        except ValueError:
            pass

    invalid_moe = [(3, 2, 128, 1), (4, 2, 63, 2)]
    for case in invalid_moe:
        try:
            validate_moe_compatibility(*case)
            raise AssertionError(f"Expected ValueError for moe case: {case}")
        except ValueError:
            pass

    # Validate CP constraint: slen must be divisible by cp
    # Valid case: should pass
    _validate_slen_divisible_by_cp(32, 2)
    # Invalid case: should raise ValueError
    try:
        _validate_slen_divisible_by_cp(31, 2)
        raise AssertionError("Expected ValueError for slen not divisible by cp")
    except ValueError:
        pass

    rank_valid = dist.get_rank() if dist.is_initialized() else 0
    if rank_valid == 0:
        print("All validation tests passed.")


# Pytest entry points called by parallel_run
def test_ep_only_base() -> None:
    """Run EP-only base tests."""
    init_backend(_DEVICE_TYPE)
    for params in EP_ONLY_PARAMS:
        num_experts, top_k, dim, hidden_dim, bs, slen = params
        _run_moe_test(
            "ep-only", num_experts, top_k, dim, hidden_dim,
            bs, slen, use_grouped_mm=False, shared_expert=False
        )


def test_ep_only_grouped_mm() -> None:
    """Check grouped_mm forward precision against the differentiable loop reference."""
    init_backend(_DEVICE_TYPE)
    for params in EP_ONLY_PARAMS:
        num_experts, top_k, dim, hidden_dim, bs, slen = params
        _run_moe_test(
            "ep-only", num_experts, top_k, dim, hidden_dim,
            bs, slen, use_grouped_mm=True, shared_expert=False,
            check_gradient=False,
            rtol=1e-2, atol=1e-2
        )


def test_ep_only_shared() -> None:
    """Run EP-only tests with shared expert."""
    init_backend(_DEVICE_TYPE)
    for params in EP_ONLY_PARAMS:
        num_experts, top_k, dim, hidden_dim, bs, slen = params
        _run_moe_test(
            "ep-only", num_experts, top_k, dim, hidden_dim,
            bs, slen, use_grouped_mm=False, shared_expert=True
        )


def test_tp_only() -> None:
    """Run TP-only tests."""
    init_backend(_DEVICE_TYPE)
    for params in TP_ONLY_PARAMS:
        num_experts, top_k, dim, hidden_dim, bs, slen = params
        _run_moe_test(
            "tp-only", num_experts, top_k, dim, hidden_dim,
            bs, slen, use_grouped_mm=False, shared_expert=False
        )


def test_validation() -> None:
    """Run validation tests (no distributed communication)."""
    init_backend(_DEVICE_TYPE)
    _run_validation_tests()


def test_dp_ep() -> None:
    """Run DP+EP tests (requires 4 cards)."""
    if int(os.environ.get("WORLD_SIZE", 1)) < 4:
        print("Skipping dp-ep: requires 4 cards")
        return
    init_backend(_DEVICE_TYPE)
    num_experts, top_k, dim, hidden_dim, bs, slen = MULTI_CARD_CONFIGS["dp-ep"]["params"]
    _run_moe_test(
        "dp-ep", num_experts, top_k, dim, hidden_dim,
        bs, slen, use_grouped_mm=False, shared_expert=False
    )


def test_ep_tp() -> None:
    """Run EP+TP tests (requires 4 cards)."""
    if int(os.environ.get("WORLD_SIZE", 1)) < 4:
        print("Skipping ep-tp: requires 4 cards")
        return
    init_backend(_DEVICE_TYPE)
    num_experts, top_k, dim, hidden_dim, bs, slen = MULTI_CARD_CONFIGS["ep-tp"]["params"]
    _run_moe_test(
        "ep-tp", num_experts, top_k, dim, hidden_dim,
        bs, slen, use_grouped_mm=False, shared_expert=False
    )


def test_dp_ep_tp() -> None:
    """Run DP+EP+TP tests (requires 8 cards)."""
    if int(os.environ.get("WORLD_SIZE", 1)) < 8:
        print("Skipping dp-ep-tp: requires 8 cards")
        return
    init_backend(_DEVICE_TYPE)
    num_experts, top_k, dim, hidden_dim, bs, slen = MULTI_CARD_CONFIGS["dp-ep-tp"]["params"]
    _run_moe_test(
        "dp-ep-tp", num_experts, top_k, dim, hidden_dim,
        bs, slen, use_grouped_mm=False, shared_expert=False
    )


def run_dp_ep_cp() -> None:
    """Run DP+EP+CP tests (CP dimension compatibility only)."""
    if int(os.environ.get("WORLD_SIZE", 1)) < 8:
        print("Skipping dp-ep-cp: requires 8 cards")
        return
    init_backend(_DEVICE_TYPE)
    num_experts, top_k, dim, hidden_dim, bs, slen = MULTI_CARD_CONFIGS["dp-ep-cp"]["params"]
    _run_moe_test(
        "dp-ep-cp", num_experts, top_k, dim, hidden_dim,
        bs, slen, use_grouped_mm=False, shared_expert=False
    )


def test_dp_ep_cp_with_attention() -> None:
    """Run DP+EP+CP with real Attention module (requires 8 cards)."""
    if int(os.environ.get("WORLD_SIZE", 1)) < 8:
        print("Skipping dp_ep_cp_with_attention: requires 8 cards")
        return
    init_backend(_DEVICE_TYPE)

    test_configs = [
        (4, 2, 64, 4, 128, 2, 32),
        (4, 2, 64, 4, 128, 2, 16),
        (8, 2, 64, 4, 128, 2, 32),
    ]

    for num_experts, top_k, dim, num_heads, hidden_dim, bs, slen in test_configs:
        _run_cp_attention_test(
            "dp-ep-cp",
            num_experts,
            top_k,
            dim,
            num_heads,
            hidden_dim,
            bs,
            slen,
            cp=2,
            use_grouped_mm=False,
            rtol=1e-3,
            atol=1e-3,
        )
