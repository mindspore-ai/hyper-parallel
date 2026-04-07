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
"""Unified ContextParallel NPU tests — hccl backend, run via torchrun.

Redesign goals vs context_parallel_npu_test.py:
  - All formats covered: BSH, SBH, BSND, BNSD, TND
  - Causal masks: sparse_mode=0 (explicit), sparse_mode=2 (leftUpCausal), sparse_mode=3 (TND)
  - Most tests use a numerical baseline; documented exceptions keep targeted checks
  - No profiler, no warmup (those belong in perf tests)
  - AsyncContextParallel tests on NPU with precision comparison

Test groups:
  Group 1  (U*)  Pure Ulysses,     cp=2, npu_fusion_attention / F.sdpa
  Group 2  (C*)  Pure Colossal AI, cp=2, npu_fusion_attention / F.sdpa
  Group 3  (H*)  Hybrid CP,        cp=4, ds=2, co=2
  Group 4  (L*)  Load Balancing,   cp=2, Colossal causal
  Group 5  (A*)  AsyncContextParallel NPU, cp=2/4
  Group 6  (I*)  API & Integration, cp=2/4

Run 2-card tests:
    HYPER_PARALLEL_PLATFORM=torch torchrun --nproc-per-node=2 \\
        --master_addr=127.0.0.1 --master_port=13000 \\
        -m pytest -s tests/torch/context_parallel/_test_context_parallel.py \\
        -k "not hybrid and not tp and not a3"

Run 4-card tests:
    HYPER_PARALLEL_PLATFORM=torch torchrun --nproc-per-node=4 \\
        --master_addr=127.0.0.1 --master_port=13100 \\
        -m pytest -s tests/torch/context_parallel/_test_context_parallel.py \\
        -k "hybrid or tp or a3"
"""
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu  # type: ignore[import-untyped]
import pytest

from hyper_parallel import init_device_mesh, ContextParallel, AsyncContextParallel, parallelize_module
from tests.torch.utils import init_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_dist_npu():
    """Init process group (hccl) and set NPU device per rank.

    Returns:
        rank (int): global rank of this process.
    """
    rank, _ = init_dist()
    return rank


def _assert_close(actual, expected, rank, label="", atol=1e-2, rtol=1e-2):
    """Compare two NPU tensors via CPU float32 conversion.

    Args:
        actual:   CP output tensor (NPU, float16).
        expected: Single-card reference tensor (NPU, float16).
        rank:     Global rank for logging.
        label:    Test label for the printed message.
        atol:     Absolute tolerance (default 1e-2 for float16).
        rtol:     Relative tolerance (default 1e-2 for float16).
    """
    act_np = actual.detach().cpu().float().numpy()
    exp_np = expected.detach().cpu().float().numpy()
    max_diff = np.max(np.abs(act_np - exp_np))
    print(f"[Rank {rank}] {label} max_diff={max_diff:.2e}")
    assert np.allclose(act_np, exp_np, atol=atol, rtol=rtol), \
        f"Rank {rank}: {label} max_diff={max_diff:.4e}"


def _sync_workers():
    """Synchronize ranks between merged sub-cases."""
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class BshFaAttn(nn.Module):
    """npu_fusion_attention non-causal, BSH layout: [B, S, H*D].

    head_num must be supplied at construction time (hidden dim is H*D,
    not separable from shape alone).
    """

    def __init__(self, head_num: int) -> None:
        """Store the head count for BSH flash attention."""
        super().__init__()
        self.head_num = head_num

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run non-causal flash attention on BSH inputs.

        Args:
            q: Query tensor in BSH layout.
            k: Key tensor in BSH layout.
            v: Value tensor in BSH layout.

        Returns:
            The attention output tensor.
        """
        scale = (q.shape[-1] // self.head_num) ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, self.head_num, "BSH",
            scale=scale, sparse_mode=0,
        )[0]


class SbhFaAttn(nn.Module):
    """npu_fusion_attention non-causal, SBH layout: [S, B, H*D]."""

    def __init__(self, head_num: int) -> None:
        """Store the head count for SBH flash attention."""
        super().__init__()
        self.head_num = head_num

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run non-causal flash attention on SBH inputs.

        Args:
            q: Query tensor in SBH layout.
            k: Key tensor in SBH layout.
            v: Value tensor in SBH layout.

        Returns:
            The attention output tensor.
        """
        scale = (q.shape[-1] // self.head_num) ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, self.head_num, "SBH",
            scale=scale, sparse_mode=0,
        )[0]


class BsndFaAttn(nn.Module):
    """npu_fusion_attention non-causal, BSND layout: [B, S, H, D]."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run non-causal flash attention on BSND inputs.

        Args:
            q: Query tensor in BSND layout.
            k: Key tensor in BSND layout.
            v: Value tensor in BSND layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[2]
        scale = q.shape[-1] ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "BSND",
            scale=scale, sparse_mode=0,
        )[0]


class BsndFaCausalLeftupAttn(nn.Module):
    """npu_fusion_attention causal, BSND layout, sparse_mode=2 (leftUpCausal).

    Requires a fixed 2048×2048 compressed causal mask.
    Upper-right triangle = True (mask future tokens), lower-left = False (attend past).
    In Colossal CP, the dispatcher (NPUFlashAttentionScoreDistributedOp._compute_sparse_params)
    adjusts pre_tockens/next_tockens per rank to achieve globally-correct causal attention.
    """

    def __init__(self, head_num: int) -> None:
        """Create the fixed left-up causal mask for BSND attention."""
        super().__init__()
        self.head_num = head_num
        self.register_buffer(
            "atten_mask",
            torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1),
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run left-up causal flash attention on BSND inputs.

        Args:
            q: Query tensor in BSND layout.
            k: Key tensor in BSND layout.
            v: Value tensor in BSND layout.

        Returns:
            The attention output tensor.
        """
        scale = q.shape[-1] ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, self.head_num, "BSND",
            atten_mask=self.atten_mask,
            scale=scale, sparse_mode=2,
            pre_tockens=65536, next_tockens=0,
        )[0]


class BnsdFaAttn(nn.Module):
    """npu_fusion_attention non-causal, BNSD layout: [B, H, S, D]."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run non-causal flash attention on BNSD inputs.

        Args:
            q: Query tensor in BNSD layout.
            k: Key tensor in BNSD layout.
            v: Value tensor in BNSD layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[1]
        scale = q.shape[-1] ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "BNSD",
            scale=scale, sparse_mode=0,
        )[0]


class BnsdFaCausalExplicitAttn(nn.Module):
    """npu_fusion_attention causal, BNSD layout, sparse_mode=0 with explicit upper-tri mask.

    Builds the [S_q, S_k] bool mask dynamically: True = mask out (future tokens).
    Correct for arbitrary sequence lengths (unlike sparse_mode=2 which needs fixed 2048×2048).
    For Pure Ulysses: after ATA, S_q == S_k == full S, so the mask is the standard upper-tri.
    """

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run causal flash attention with an explicit upper-triangular mask.

        Args:
            q: Query tensor in BNSD layout.
            k: Key tensor in BNSD layout.
            v: Value tensor in BNSD layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[1]
        s_q, s_k = q.shape[2], k.shape[2]
        scale = q.shape[-1] ** -0.5
        atten_mask = torch.triu(
            torch.ones(s_q, s_k, dtype=torch.bool, device=q.device), diagonal=1
        )
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "BNSD",
            atten_mask=atten_mask, scale=scale, sparse_mode=0,
        )[0]


class BnsdFaCausalColossalAttn(nn.Module):
    """npu_fusion_attention causal, BNSD, sparse_mode=0, for Colossal CP mode.

    Builds a standard (S, S) upper-triangular causal mask using k.shape[2] to
    obtain the true global sequence length S.

    Note: with load_balance=True, q.shape[2] inside forward() returns S/2
    (not the full S) because the framework wraps q_half with a co_submesh of
    size cp_size/2. k.shape[seq_dim] always returns the true global S because
    K/V are Replicate. Always use k.shape[seq_dim] for the mask dimension in
    load-balanced Colossal AI mode.

    The dispatcher's _adjust_atten_mask_for_seq_split() slices the (S, S)
    mask to (local_s, S) per rank for globally-correct causal attention.
    """

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run Colossal causal flash attention with a global explicit mask.

        Args:
            q: Local query tensor in BNSD layout.
            k: Global key tensor in BNSD layout.
            v: Global value tensor in BNSD layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[1]
        s = k.shape[2]  # K is always Replicate → true global S in all Colossal modes
        scale = q.shape[-1] ** -0.5
        atten_mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "BNSD",
            atten_mask=atten_mask, scale=scale, sparse_mode=0,
        )[0]


class BnsdFaCausalLeftupAttn(nn.Module):
    """npu_fusion_attention causal, BNSD layout, sparse_mode=2 (leftUpCausal).

    In Colossal CP, the dispatcher adjusts pre_tockens/next_tockens per rank
    to achieve globally-correct causal attention.
    """

    def __init__(self) -> None:
        """Create the fixed left-up causal mask for BNSD attention."""
        super().__init__()
        self.register_buffer(
            "atten_mask",
            torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1),
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run left-up causal flash attention on BNSD inputs.

        Args:
            q: Query tensor in BNSD layout.
            k: Key tensor in BNSD layout.
            v: Value tensor in BNSD layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[1]
        scale = q.shape[-1] ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "BNSD",
            atten_mask=self.atten_mask,
            scale=scale, sparse_mode=2,
            pre_tockens=65536, next_tockens=0,
        )[0]


class NpuFlashAttentionTND(nn.Module):
    """npu_fusion_attention in TND format: [T, H, D].

    actual_seq_qlen/kvlen store global cumulative sample lengths.
    The distributed op adjusts them per-rank for Colossal CP automatically.
    """

    def __init__(
        self,
        actual_seq_qlen: Sequence[int],
        actual_seq_kvlen: Sequence[int],
        sparse_mode: int = 0,
        atten_mask: Optional[Tensor] = None,
    ) -> None:
        """Store TND attention metadata and an optional causal mask."""
        super().__init__()
        self.actual_seq_qlen = actual_seq_qlen
        self.actual_seq_kvlen = actual_seq_kvlen
        self.sparse_mode = sparse_mode
        self.atten_mask = atten_mask

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Run flash attention on TND inputs.

        Args:
            q: Query tensor in TND layout.
            k: Key tensor in TND layout.
            v: Value tensor in TND layout.

        Returns:
            The attention output tensor.
        """
        head_num = q.shape[1]
        scale = q.shape[-1] ** -0.5
        return torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            q, k, v, head_num, "TND",
            scale=scale,
            sparse_mode=self.sparse_mode,
            atten_mask=self.atten_mask,
            actual_seq_qlen=self.actual_seq_qlen,
            actual_seq_kvlen=self.actual_seq_kvlen,
        )[0]


class SimpleAttnBSHD(nn.Module):
    """Minimal BSHD matmul attention: [B, S, H, D] input/output."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute plain scaled dot-product attention in BSHD layout.

        Args:
            q: Query tensor in BSHD layout.
            k: Key tensor in BSHD layout.
            v: Value tensor in BSHD layout.

        Returns:
            The attention output tensor.
        """
        # BSHD → BHSD
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2)  # BHSD → BSHD


# ---------------------------------------------------------------------------
# Group 1: Pure Ulysses (cp=2)
# ---------------------------------------------------------------------------

def _test_ulysses_bnsd_fa_noncausal():
    """U1: Pure Ulysses CP=2, BNSD, npu_fusion_attention non-causal.

    Reference: single-card NpuFA on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(42)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaAttn()
    ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "U1_ulysses_bnsd_fa_noncausal")


def _test_ulysses_bnsd_fa_causal_explicit():
    """U2: Pure Ulysses CP=2, BNSD, npu_fusion_attention causal (sparse_mode=0, explicit mask).

    After ATA each rank holds full S but H/cp heads; standard upper-tri mask is correct.
    Reference: single-card causal NpuFA on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(123)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaCausalExplicitAttn()
    ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaCausalExplicitAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "U2_ulysses_bnsd_fa_causal_explicit")


def _test_ulysses_bnsd_fa_causal_leftup():
    """U2b: Pure Ulysses CP=2, BNSD, npu_fusion_attention causal (sparse_mode=2 leftUpCausal).

    In Ulysses mode, after ATA each rank holds full S (not sharded), so the standard
    causal mask with pre_tockens=65536 is correct—no dispatcher adjustment needed.
    Reference: single-card BnsdFaCausalLeftupAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(456)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaCausalLeftupAttn().to("npu")
    ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaCausalLeftupAttn().to("npu")
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "U2b_ulysses_bnsd_fa_causal_leftup")


def test_ulysses_bsnd_fa_noncausal():
    """U3: Pure Ulysses CP=2, BSND [B,S,H,D], npu_fusion_attention non-causal.

    seq_dim=1 (S), head_dim=2 (H).
    Reference: single-card BsndFaAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(789)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s]

    core_attn = BsndFaAttn()
    ContextParallel(seq_dim=1, head_dim=2).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BsndFaAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "U3_ulysses_bsnd_fa_noncausal")


def _test_ulysses_bnsd_sdpa_noncausal():
    """U4: Pure Ulysses CP=2, BNSD, F.scaled_dot_product_attention non-causal.

    Reference: single-card F.sdpa on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(42)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    class _FsdpaNonCausal(nn.Module):
        """Local non-causal SDPA wrapper for the Ulysses BNSD suite."""

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Run non-causal SDPA on local tensors.

            Args:
                q: Local query tensor.
                k: Local key tensor.
                v: Local value tensor.

            Returns:
                The attention output tensor.
            """
            return F.scaled_dot_product_attention(q, k, v)

    core_attn = _FsdpaNonCausal()
    ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = F.scaled_dot_product_attention(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "U4_ulysses_bnsd_sdpa_noncausal")


def _test_ulysses_bnsd_sdpa_causal():
    """U5: Pure Ulysses CP=2, BNSD, F.scaled_dot_product_attention causal (is_causal=True).

    After ATA each rank holds full S; is_causal=True applies to the complete context.
    Reference: single-card causal F.sdpa on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(123)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    class _FsdpaCausal(nn.Module):
        """Local causal SDPA wrapper for the Ulysses BNSD suite."""

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Run causal SDPA on local tensors.

            Args:
                q: Local query tensor.
                k: Local key tensor.
                v: Local value tensor.

            Returns:
                The attention output tensor.
            """
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    core_attn = _FsdpaCausal()
    ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = F.scaled_dot_product_attention(full_q, full_k, full_v, is_causal=True)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "U5_ulysses_bnsd_sdpa_causal")


def test_ulysses_tnd_fa_noncausal():
    """U6: Pure Ulysses CP=2, TND format, npu_fusion_attention non-causal.

    After ATA each rank holds full T but H/cp heads; actual_seq_qlen unchanged.
    Reference: single-card TND NpuFA on full tensors → local slice.
    """
    rank = _init_dist_npu()
    num_samples, tps = 2, 4   # tps = tokens_per_sample
    total_len = num_samples * tps      # 8
    num_heads, head_dim = 4, 16
    cp_size = 2
    t_local = total_len // cp_size

    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(77)
    full_q = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[rank * t_local:(rank + 1) * t_local]
    local_k = full_k[rank * t_local:(rank + 1) * t_local]
    local_v = full_v[rank * t_local:(rank + 1) * t_local]

    actual_seq_qlen = [(i + 1) * tps for i in range(num_samples)]
    actual_seq_kvlen = actual_seq_qlen[:]

    core_attn = NpuFlashAttentionTND(actual_seq_qlen, actual_seq_kvlen, sparse_mode=0)
    ContextParallel(seq_dim=0, head_dim=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            full_q, full_k, full_v,
            head_num=num_heads, input_layout="TND", scale=head_dim ** -0.5,
            sparse_mode=0, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
        )[0]
    ref_local = ref_full[rank * t_local:(rank + 1) * t_local]

    assert cp_out.shape == (t_local, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "U6_ulysses_tnd_fa_noncausal")


def test_ulysses_bnsd_suite():
    """Merged Ulysses BNSD suite.

    Runs the same CP=2 / BNSD setup across FA and SDPA variants in one torchrun
    session to reduce process-launch overhead.
    """
    _test_ulysses_bnsd_fa_noncausal()
    _sync_workers()
    _test_ulysses_bnsd_fa_causal_explicit()
    _sync_workers()
    _test_ulysses_bnsd_fa_causal_leftup()
    _sync_workers()
    _test_ulysses_bnsd_sdpa_noncausal()
    _sync_workers()
    _test_ulysses_bnsd_sdpa_causal()


# ---------------------------------------------------------------------------
# Group 2: Pure Colossal AI (cp=2)
# ---------------------------------------------------------------------------

def test_colossal_bsh_fa_noncausal():
    """C1: Pure Colossal CP=2, BSH [B,S,H*D], npu_fusion_attention non-causal.

    seq_dim=1, ulysses_degree=1. head_dim is not needed for BSH Colossal
    (no head-dimension A2A), but we pass a dummy head_dim=2 for API compat.
    Reference: BshFaAttn(local_q, full_k, full_v) per rank.
    """
    rank = _init_dist_npu()
    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    hidden = num_heads * head_dim
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(11)
    full_q = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device="npu")
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s]

    core_attn = BshFaAttn(head_num=num_heads)
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    # Colossal reference: local_q attends to full K/V (row-independence property).
    ref_model = BshFaAttn(head_num=num_heads)
    with torch.no_grad():
        ref_local = ref_model(local_q, full_k, full_v)

    assert cp_out.shape == (batch, local_s, hidden)
    _assert_close(cp_out, ref_local, rank, "C1_colossal_bsh_fa_noncausal")


def test_colossal_sbh_fa_noncausal():
    """C2: Pure Colossal CP=2, SBH [S,B,H*D], npu_fusion_attention non-causal.

    seq_dim=0, ulysses_degree=1.
    Reference: SbhFaAttn(local_q, full_k, full_v) per rank.
    """
    rank = _init_dist_npu()
    seq_len, batch, num_heads, head_dim = 8, 2, 4, 16
    hidden = num_heads * head_dim
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(22)
    full_q = torch.randn(seq_len, batch, hidden, dtype=torch.float16, device="npu")
    full_k = torch.randn(seq_len, batch, hidden, dtype=torch.float16, device="npu")
    full_v = torch.randn(seq_len, batch, hidden, dtype=torch.float16, device="npu")
    local_q = full_q[rank * local_s:(rank + 1) * local_s]
    local_k = full_k[rank * local_s:(rank + 1) * local_s]
    local_v = full_v[rank * local_s:(rank + 1) * local_s]

    core_attn = SbhFaAttn(head_num=num_heads)
    ContextParallel(seq_dim=0, head_dim=2, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = SbhFaAttn(head_num=num_heads)
    with torch.no_grad():
        ref_local = ref_model(local_q, full_k, full_v)

    assert cp_out.shape == (local_s, batch, hidden)
    _assert_close(cp_out, ref_local, rank, "C2_colossal_sbh_fa_noncausal")


def test_colossal_bsnd_fa_noncausal():
    """C3: Pure Colossal CP=2, BSND [B,S,H,D], npu_fusion_attention non-causal.

    seq_dim=1, ulysses_degree=1.
    Reference: BsndFaAttn(local_q, full_k, full_v) per rank.
    """
    rank = _init_dist_npu()
    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(33)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s]

    core_attn = BsndFaAttn()
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BsndFaAttn()
    with torch.no_grad():
        ref_local = ref_model(local_q, full_k, full_v)

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "C3_colossal_bsnd_fa_noncausal")


def test_colossal_bsnd_fa_causal_leftup():
    """C3b: Pure Colossal CP=2, BSND, npu_fusion_attention causal (sparse_mode=2 leftUpCausal).

    Core test: NPUFlashAttentionScoreDistributedOp._compute_sparse_params converts
    sparse_mode=2 to BAND (mode=4) and adjusts pre/next_tockens per rank so that
    each rank's local Q only attends to the globally-correct KV window.

    Reference: single-card BsndFaCausalLeftupAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(44)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s]

    core_attn = BsndFaCausalLeftupAttn(head_num=num_heads).to("npu")
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    # Reference: globally-correct causal on full sequence → take local slice.
    ref_model = BsndFaCausalLeftupAttn(head_num=num_heads).to("npu")
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "C3b_colossal_bsnd_fa_causal_leftup")


def _test_colossal_bnsd_fa_noncausal():
    """C4: Pure Colossal CP=2, BNSD [B,H,S,D], npu_fusion_attention non-causal.

    seq_dim=2, head_dim=1, ulysses_degree=1.
    Reference: BnsdFaAttn(local_q, full_k, full_v) per rank.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(77)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaAttn()
    with torch.no_grad():
        ref_local = ref_model(local_q, full_k, full_v)

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "C4_colossal_bnsd_fa_noncausal")


def _test_colossal_bnsd_fa_causal_explicit():
    """C5: Pure Colossal CP=2, BNSD, npu_fusion_attention causal (sparse_mode=0, offset mask).

    BnsdFaCausalColossalAttn uses dist.get_rank() to build the globally-correct
    [local_s, S] causal mask: mask[qi, ki] = True when ki > q_start + qi.

    Reference: single-card BnsdFaCausalExplicitAttn on full tensors → local slice.
    Both produce globally-correct causal attention for each rank's token range.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(55)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaCausalColossalAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    # Single-card reference with standard causal mask (globally correct for full S).
    ref_model = BnsdFaCausalExplicitAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "C5_colossal_bnsd_fa_causal_explicit")


def _test_colossal_bnsd_fa_causal_leftup():
    """C5b: Pure Colossal CP=2, BNSD, npu_fusion_attention causal (sparse_mode=2 leftUpCausal).

    Verifies the same dispatcher adjustment as C3b but in BNSD layout.
    Reference: single-card BnsdFaCausalLeftupAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(66)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaCausalLeftupAttn().to("npu")
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaCausalLeftupAttn().to("npu")
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "C5b_colossal_bnsd_fa_causal_leftup")


def _test_colossal_bnsd_sdpa_noncausal():
    """C6: Pure Colossal CP=2, BNSD, F.scaled_dot_product_attention non-causal.

    Reference: F.sdpa(local_q, full_k, full_v) per rank.
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(77)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    class _FsdpaNonCausal(nn.Module):
        """Local non-causal SDPA wrapper for the Colossal BNSD suite."""

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Run non-causal SDPA on local tensors.

            Args:
                q: Local query tensor.
                k: Replicated key tensor.
                v: Replicated value tensor.

            Returns:
                The attention output tensor.
            """
            return F.scaled_dot_product_attention(q, k, v)

    core_attn = _FsdpaNonCausal()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_local = F.scaled_dot_product_attention(local_q, full_k, full_v)

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "C6_colossal_bnsd_sdpa_noncausal")


def _test_colossal_bnsd_sdpa_causal():
    """C7: Pure Colossal CP=2, BNSD, F.sdpa causal with global offset mask.

    In Colossal mode Q is local [B,H,local_s,D] and K/V are full [B,H,S,D].
    is_causal=True is wrong here (only applies [local_s×local_s] lower-tri mask).
    Instead, build an explicit [local_s, S] causal mask with global Q offset.

    Reference: same explicit global causal mask on (local_q, full_k, full_v).
    """
    rank = _init_dist_npu()
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(321)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    class _FsdpaCausalColossal(nn.Module):
        """Global causal mask for Colossal CP.

        q/k/v arrive as DTensors (Q: Shard(seq_dim), K/V: Replicate), so
        q.shape[2] is the GLOBAL sequence length S.  Build a full [S, S] causal
        mask; the SDPA dispatcher slices it to [local_s, S] for each rank via
        _adjust_attn_mask_for_sp when attn_mask.shape[-2] == global_q_len.
        """

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Run causal SDPA with the Colossal-style global mask.

            Args:
                q: Query tensor with DTensor global shape metadata.
                k: Key tensor with DTensor global shape metadata.
                v: Value tensor with DTensor global shape metadata.

            Returns:
                The attention output tensor.
            """
            s_q, s_k = q.shape[2], k.shape[2]   # global S (DTensor global shape)
            q_pos = torch.arange(s_q, device=q.device).unsqueeze(1)   # [S, 1]
            k_pos = torch.arange(s_k, device=q.device).unsqueeze(0)   # [1, S]
            causal_mask = k_pos <= q_pos   # [S, S] global causal mask, dispatcher slices per rank
            return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

    core_attn = _FsdpaCausalColossal()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    # Reference: same global causal mask, direct call (no CP).
    q_start = rank * local_s
    q_pos = torch.arange(q_start, q_start + local_s, device=local_q.device).unsqueeze(1)
    k_pos = torch.arange(seq_len, device=local_q.device).unsqueeze(0)
    causal_mask_ref = k_pos <= q_pos
    with torch.no_grad():
        ref_local = F.scaled_dot_product_attention(local_q, full_k, full_v,
                                                   attn_mask=causal_mask_ref)

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "C7_colossal_bnsd_sdpa_causal")


def test_colossal_tnd_fa_causal():
    """C8: Pure Colossal CP=2, TND format, npu_fusion_attention causal (sparse_mode=3).

    sparse_mode=3 (RightDownCausal) with 2048×2048 upper-tri bool compressed mask.
    Reference: single-card TND causal NpuFA on full tensors → local slice.
    """
    rank = _init_dist_npu()
    num_samples, tps = 2, 4
    total_len = num_samples * tps    # 8
    num_heads, head_dim = 4, 16
    cp_size = 2
    t_local = total_len // cp_size

    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to("npu")

    torch.manual_seed(13)
    full_q = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[rank * t_local:(rank + 1) * t_local]
    local_k = full_k[rank * t_local:(rank + 1) * t_local]
    local_v = full_v[rank * t_local:(rank + 1) * t_local]

    actual_seq_qlen = [(i + 1) * tps for i in range(num_samples)]
    actual_seq_kvlen = actual_seq_qlen[:]

    core_attn = NpuFlashAttentionTND(
        actual_seq_qlen, actual_seq_kvlen, sparse_mode=3, atten_mask=atten_mask,
    )
    ContextParallel(seq_dim=0, head_dim=1, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            full_q, full_k, full_v,
            head_num=num_heads, input_layout="TND", scale=head_dim ** -0.5,
            sparse_mode=3, atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
        )[0]
    ref_local = ref_full[rank * t_local:(rank + 1) * t_local]

    assert cp_out.shape == (t_local, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "C9_colossal_tnd_fa_causal")


def test_colossal_bnsd_suite():
    """Merged Colossal BNSD suite.

    Covers FA / SDPA and causal / non-causal variants for the same CP=2 BNSD
    setup in one torchrun session.
    """
    _test_colossal_bnsd_fa_noncausal()
    _sync_workers()
    _test_colossal_bnsd_fa_causal_explicit()
    _sync_workers()
    _test_colossal_bnsd_fa_causal_leftup()
    _sync_workers()
    _test_colossal_bnsd_sdpa_noncausal()
    _sync_workers()
    _test_colossal_bnsd_sdpa_causal()


# ---------------------------------------------------------------------------
# Group 3: Hybrid CP (cp=4, ds=2, co=2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def _test_hybrid_bnsd_fa_noncausal():
    """H1: Hybrid CP=4 (ds=2, co=2), BNSD, npu_fusion_attention non-causal.

    Reference: single-card BnsdFaAttn on full tensors → local slice.
    All ranks use the same seed to generate identical full tensors.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 16
    cp_size = 4
    ds = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    # Same seed on all ranks → same full tensors → consistent reference.
    torch.manual_seed(42)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ds).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "H1_hybrid_bnsd_fa_noncausal")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def _test_hybrid_bnsd_fa_causal_leftup():
    """H2: Hybrid CP=4 (ds=2, co=2), BNSD, npu_fusion_attention causal (sparse_mode=2).

    Verifies dispatcher sparse param adjustment + Ulysses ATA cooperate correctly.
    Reference: single-card BnsdFaCausalLeftupAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 16
    cp_size = 4
    ds = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(88)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    core_attn = BnsdFaCausalLeftupAttn().to("npu")
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ds).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    ref_model = BnsdFaCausalLeftupAttn().to("npu")
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "H2_hybrid_bnsd_fa_causal_leftup")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def _test_hybrid_bnsd_sdpa_noncausal():
    """H3: Hybrid CP=4 (ds=2, co=2), BNSD, F.scaled_dot_product_attention non-causal.

    Upgraded from shape-only check to single-card baseline comparison.
    Reference: single-card F.sdpa on full tensors → local slice.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 16
    cp_size = 4
    ds = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(99)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    class _FsdpaNonCausal(nn.Module):
        """Local non-causal SDPA wrapper for the Hybrid BNSD suite."""

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Run non-causal SDPA on local tensors.

            Args:
                q: Local query tensor.
                k: Local key tensor.
                v: Local value tensor.

            Returns:
                The attention output tensor.
            """
            return F.scaled_dot_product_attention(q, k, v)

    core_attn = _FsdpaNonCausal()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ds).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = F.scaled_dot_product_attention(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(cp_out, ref_local, rank, "H3_hybrid_bnsd_sdpa_noncausal")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def test_hybrid_tnd_fa_causal():
    """H4: Hybrid CP=4 (ds=2, co=2), TND format, npu_fusion_attention causal (sparse_mode=3).

    Reference: single-card TND causal NpuFA on full tensors → local slice.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    num_samples, tps = 4, 4
    total_len = num_samples * tps    # 16; divisible by cp=4
    num_heads, head_dim = 4, 16
    cp_size = 4
    ds = 2
    t_local = total_len // cp_size

    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to("npu")

    torch.manual_seed(31)
    full_q = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(total_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[rank * t_local:(rank + 1) * t_local]
    local_k = full_k[rank * t_local:(rank + 1) * t_local]
    local_v = full_v[rank * t_local:(rank + 1) * t_local]

    actual_seq_qlen = [(i + 1) * tps for i in range(num_samples)]
    actual_seq_kvlen = actual_seq_qlen[:]

    core_attn = NpuFlashAttentionTND(
        actual_seq_qlen, actual_seq_kvlen, sparse_mode=3, atten_mask=atten_mask,
    )
    ContextParallel(seq_dim=0, head_dim=1, ulysses_degree=ds).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    with torch.no_grad():
        ref_full = torch_npu.npu_fusion_attention(  # type: ignore[attr-defined]
            full_q, full_k, full_v,
            head_num=num_heads, input_layout="TND", scale=head_dim ** -0.5,
            sparse_mode=3, atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
        )[0]
    ref_local = ref_full[rank * t_local:(rank + 1) * t_local]

    assert cp_out.shape == (t_local, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "H4_hybrid_tnd_fa_causal")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def test_hybrid_bnsd_suite():
    """Merged Hybrid BNSD suite.

    Covers FA / SDPA and causal / non-causal variants for the shared
    CP=4, ds=2, co=2, BNSD setup in one torchrun session.
    """
    _test_hybrid_bnsd_fa_noncausal()
    _sync_workers()
    _test_hybrid_bnsd_fa_causal_leftup()
    _sync_workers()
    _test_hybrid_bnsd_sdpa_noncausal()


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="Hybrid tests require world_size=4",
)
def test_hybrid_2d_mesh_equiv():
    """H5: Hybrid CP=4 with caller-provided 2D mesh — numerical output matches 1D mesh path.

    A 2D mesh (ds=2, co=2) is passed directly; ContextParallel auto-flattens it.
    Verifies both meshes produce identical numerical results using single-card reference.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 16
    ds, co = 2, 2
    cp_size = ds * co
    local_s = seq_len // cp_size

    mesh_1d = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))
    mesh_2d = init_device_mesh("npu", (1, ds, co), mesh_dim_names=("dp", "ds", "co"))

    torch.manual_seed(999)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s]

    # 1D mesh path
    attn_1d = BnsdFaAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ds).apply(attn_1d, mesh_1d["cp"])
    with torch.no_grad():
        out_1d = attn_1d(local_q, local_k, local_v)

    # 2D mesh path
    attn_2d = BnsdFaAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ds).apply(
        attn_2d, mesh_2d[("ds", "co")]
    )
    with torch.no_grad():
        out_2d = attn_2d(local_q, local_k, local_v)

    # Both paths should also match single-card reference
    ref_model = BnsdFaAttn()
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, :, rank * local_s:(rank + 1) * local_s]

    assert out_1d.shape == (batch, num_heads, local_s, head_dim)
    assert out_2d.shape == (batch, num_heads, local_s, head_dim)
    _assert_close(out_1d, ref_local, rank, "H5_hybrid_2d_mesh_1d_vs_ref")
    _assert_close(out_2d, ref_local, rank, "H5_hybrid_2d_mesh_2d_vs_ref")


# ---------------------------------------------------------------------------
# Group 4: Load Balancing (cp=2, Colossal causal)
# ---------------------------------------------------------------------------

def test_lb_bsnd_fa_causal():
    """L1: Q-exchange Load Balance (BSND, sparse_mode=2) matches single-card causal reference.

    Uses BsndFaCausalLeftupAttn (with explicit 2048×2048 atten_mask).
    Reference: single-card BsndFaCausalLeftupAttn on full tensors → local slice.
    """
    rank = _init_dist_npu()
    cp_size = dist.get_world_size()
    if cp_size != 2:
        pytest.skip("L1 requires exactly 2 NPU cards")

    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))
    cp_mesh = mesh["cp"]

    device = f"npu:{rank % torch.npu.device_count()}"
    torch.manual_seed(42)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    dist.broadcast(full_q, src=0)
    dist.broadcast(full_k, src=0)
    dist.broadcast(full_v, src=0)
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s].contiguous()
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s].contiguous()
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s].contiguous()

    # Single-card reference: full-sequence causal → take local slice.
    ref_model = BsndFaCausalLeftupAttn(head_num=num_heads).to("npu")
    with torch.no_grad():
        ref_full = ref_model(full_q, full_k, full_v)
    ref_local = ref_full[:, rank * local_s:(rank + 1) * local_s]

    # Q-exchange LB
    attn_lb = BsndFaCausalLeftupAttn(head_num=num_heads).to("npu")
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1, load_balance=True).apply(
        attn_lb, cp_mesh
    )
    with torch.no_grad():
        lb_out = attn_lb(local_q, local_k, local_v)

    _assert_close(lb_out, ref_local, rank, "L1_lb_bsnd_fa_causal")


def test_lb_bnsd_fa_causal():
    """L2: Q-exchange LB (BNSD, sparse_mode=0 explicit mask) == plain Colossal causal.

    Verifies LB=True produces same result as LB=False in BNSD layout.
    Both use BnsdFaCausalColossalAttn (globally-correct offset mask via dist.get_rank()).
    """
    rank = _init_dist_npu()
    cp_size = dist.get_world_size()
    if cp_size != 2:
        pytest.skip("L2 requires exactly 2 NPU cards")

    batch, num_heads, seq_len, head_dim = 1, 4, 512, 32
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))
    cp_mesh = mesh["cp"]

    device = f"npu:{rank % torch.npu.device_count()}"
    torch.manual_seed(17)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    full_k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    full_v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    dist.broadcast(full_q, src=0)
    dist.broadcast(full_k, src=0)
    dist.broadcast(full_v, src=0)
    local_q = full_q[:, :, rank * local_s:(rank + 1) * local_s].contiguous()
    local_k = full_k[:, :, rank * local_s:(rank + 1) * local_s].contiguous()
    local_v = full_v[:, :, rank * local_s:(rank + 1) * local_s].contiguous()

    # Reference: plain Colossal AI, no LB
    attn_ref = BnsdFaCausalColossalAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1).apply(attn_ref, cp_mesh)
    with torch.no_grad():
        ref_out = attn_ref(local_q, local_k, local_v)

    # Q-exchange LB
    attn_lb = BnsdFaCausalColossalAttn()
    ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=1, load_balance=True).apply(
        attn_lb, cp_mesh
    )
    with torch.no_grad():
        lb_out = attn_lb(local_q, local_k, local_v)

    assert lb_out.shape == ref_out.shape
    _assert_close(lb_out, ref_out, rank, "L2_lb_bnsd_fa_causal")


# ---------------------------------------------------------------------------
# Group 5: AsyncContextParallel NPU
# ---------------------------------------------------------------------------

class _ProjectionBSHD(nn.Module):
    """Linear projection that outputs BSHD format: [B, S, H, D].

    AsyncContextParallel hooks fire on the projection output, so the output
    must have the correct seq_dim=1 (S) and head_dim=2 (H) dimensions.
    """

    def __init__(self, in_dim: int, num_heads: int, head_dim: int) -> None:
        """Create the shared linear projection for one Q, K, or V branch."""
        super().__init__()
        self.linear = nn.Linear(in_dim, num_heads * head_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: Tensor) -> Tensor:
        """Project hidden states and reshape them into BSHD layout.

        Args:
            x: Input hidden states in `[B, S, hidden]` layout.

        Returns:
            Projected tensor in `[B, S, H, D]` layout.
        """
        batch, seq_len, _ = x.shape
        return self.linear(x).view(batch, seq_len, self.num_heads, self.head_dim)


class _AsyncModel(nn.Module):
    """Minimal model with separate q/k/v BSHD projections for AsyncContextParallel testing.

    Output layout: BSHD [B, S, H, D].  Hidden size = H * D.
    The projections output [B, S, H, D] so that AsyncCP can apply A2A on seq_dim=1.
    """

    def __init__(self, hidden: int, num_heads: int, head_dim: int) -> None:
        """Build the q/k/v projections and BSHD attention module."""
        super().__init__()
        self.q_proj = _ProjectionBSHD(hidden, num_heads, head_dim)
        self.k_proj = _ProjectionBSHD(hidden, num_heads, head_dim)
        self.v_proj = _ProjectionBSHD(hidden, num_heads, head_dim)
        self.attn = SimpleAttnBSHD()

    def forward(self, x: Tensor) -> Tensor:
        """Project inputs into q/k/v tensors and run BSHD attention.

        Args:
            x: Input hidden states in `[B, S, hidden]` layout.

        Returns:
            Attention output in `[B, S, H, D]` layout.
        """
        q = self.q_proj(x)  # [B, S, H, D]
        k = self.k_proj(x)  # [B, S, H, D]
        v = self.v_proj(x)  # [B, S, H, D]
        return self.attn(q, k, v)


def _test_async_ulysses_forward_npu():
    """A1: AsyncContextParallel (Ulysses CP=2) forward output matches single-card baseline on NPU.

    Model: nn.Linear q/k/v projections + BSHD matmul attention.
    AsyncCP overlaps seq→head A2A with projection computation.
    Reference: same model, single-card full sequence → take local slice.
    """
    rank = _init_dist_npu()
    cp_size = dist.get_world_size()
    if cp_size < 2:
        pytest.skip("Requires at least 2 processes")
    batch, seq_len, num_heads, head_dim = 1, 8, 4, 8
    hidden = num_heads * head_dim
    local_s = seq_len // cp_size

    mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))

    device = "npu"
    torch.manual_seed(42)
    full_x = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device=device)
    local_x = full_x[:, rank * local_s:(rank + 1) * local_s]

    # Build AsyncCP model with same initial weights on all ranks.
    torch.manual_seed(0)
    model_cp = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    async_cp = AsyncContextParallel(seq_dim=1, head_dim=2)
    async_cp.apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    with torch.no_grad():
        q_out = model_cp.q_proj(local_x)   # [B, local_s, H, D]
        k_out = model_cp.k_proj(local_x)   # [B, local_s, H, D]
        v_out = model_cp.v_proj(local_x)   # [B, local_s, H, D]
        cp_out = model_cp.attn(q_out, k_out, v_out)

    # Reference: same model (same weights), single-card full sequence.
    torch.manual_seed(0)
    model_ref = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        ref_full = model_ref(full_x)  # [B, S, H, D]
    ref_local = ref_full[:, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "A1_async_ulysses_forward_npu")


def _test_async_ulysses_backward_npu():
    """A2: AsyncContextParallel (Ulysses CP=2) backward gradient check on NPU.

    Verifies:
    1. Backward runs without error.
    2. All parameter gradients are non-None and non-NaN/Inf.
    3. After all-reducing CP gradients, they match single-card reference gradients.
    """
    rank = _init_dist_npu()
    cp_size = dist.get_world_size()
    if cp_size < 2:
        pytest.skip("Requires at least 2 processes")
    batch, seq_len, num_heads, head_dim = 1, 8, 4, 8
    hidden = num_heads * head_dim
    local_s = seq_len // cp_size

    mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))

    device = "npu"
    torch.manual_seed(42)
    full_x = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device=device)
    local_x = full_x[:, rank * local_s:(rank + 1) * local_s].clone()

    # CP model
    torch.manual_seed(0)
    model_cp = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    async_cp = AsyncContextParallel(seq_dim=1, head_dim=2)
    async_cp.apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    # CP forward + backward
    q_out = model_cp.q_proj(local_x)   # [B, local_s, H, D]
    k_out = model_cp.k_proj(local_x)   # [B, local_s, H, D]
    v_out = model_cp.v_proj(local_x)   # [B, local_s, H, D]
    cp_out = model_cp.attn(q_out, k_out, v_out)
    cp_out.sum().backward()

    # Check non-NaN gradients
    for name, param in model_cp.named_parameters():
        assert param.grad is not None, f"[Rank {rank}] A2: {name}.grad is None"
        assert not torch.isnan(param.grad).any(), f"[Rank {rank}] A2: NaN in {name}.grad"
        assert not torch.isinf(param.grad).any(), f"[Rank {rank}] A2: Inf in {name}.grad"

    # All-reduce CP gradients: sum over ranks = d(full_output.sum())/dW.
    # K/V weights receive gradients from all S positions (non-causal attention).
    # Each rank only holds local K/V, so per-rank gradients are partial; the
    # all-reduce assembles the complete gradient matching the single-card full loss.
    for param in model_cp.parameters():
        dist.all_reduce(param.grad)

    # Reference model (same initial weights, single-card, full-sequence loss)
    torch.manual_seed(0)
    model_ref = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    ref_full_out = model_ref(full_x)                                  # [B, S, H, D]
    ref_full_out.sum().backward()                                      # full-sequence loss

    # Compare gradients
    for (name, param_cp), (_, param_ref) in zip(
        model_cp.named_parameters(), model_ref.named_parameters()
    ):
        _assert_close(param_cp.grad, param_ref.grad, rank, f"A2_bwd_{name}")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="A3 requires world_size=4",
)
def test_async_hybrid_forward_npu():
    """A3: AsyncContextParallel Hybrid (cp=4, ds=2, co=2) forward matches single-card on NPU.

    Reference: same model (single-card, no CP), full sequence → local slice.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, seq_len, num_heads, head_dim = 1, 16, 4, 8
    hidden = num_heads * head_dim
    cp_size = 4
    ds = 2
    local_s = seq_len // cp_size

    mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))

    device = "npu"
    torch.manual_seed(42)
    full_x = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device=device)
    local_x = full_x[:, rank * local_s:(rank + 1) * local_s]

    torch.manual_seed(0)
    model_cp = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    async_cp = AsyncContextParallel(seq_dim=1, head_dim=2, ulysses_degree=ds)
    async_cp.apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    with torch.no_grad():
        q_out = model_cp.q_proj(local_x)   # [B, local_s, H, D]
        k_out = model_cp.k_proj(local_x)   # [B, local_s, H, D]
        v_out = model_cp.v_proj(local_x)   # [B, local_s, H, D]
        cp_out = model_cp.attn(q_out, k_out, v_out)

    torch.manual_seed(0)
    model_ref = _AsyncModel(hidden, num_heads, head_dim).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        ref_full = model_ref(full_x)
    ref_local = ref_full[:, rank * local_s:(rank + 1) * local_s]

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "A3_async_hybrid_forward_npu")


def test_async_ulysses_suite():
    """Merged Async Ulysses suite.

    Runs forward and backward correctness checks in one torchrun session to
    reduce launch overhead for AsyncContextParallel validation.
    """
    _test_async_ulysses_forward_npu()
    _sync_workers()
    _test_async_ulysses_backward_npu()


# ---------------------------------------------------------------------------
# Group 6: API & Integration
# ---------------------------------------------------------------------------

def test_parallelize_module_api_npu():
    """I1: parallelize_module API with ContextParallel (Colossal mode), BSHD, CP=2.

    Reference: local_q cross-attends full K/V (Colossal semantics).
    """
    rank = _init_dist_npu()
    batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
    cp_size = 2
    local_s = seq_len // cp_size
    mesh = init_device_mesh("npu", (1, cp_size), mesh_dim_names=("dp", "cp"))

    torch.manual_seed(42)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    local_q = full_q[:, rank * local_s:(rank + 1) * local_s]
    local_k = full_k[:, rank * local_s:(rank + 1) * local_s]
    local_v = full_v[:, rank * local_s:(rank + 1) * local_s]

    class _ModelWithAttn(nn.Module):
        """Tiny wrapper model used to validate the parallelize_module API."""

        def __init__(self) -> None:
            """Wrap the core attention module for API-level parallelization."""
            super().__init__()
            self.core_attn = SimpleAttnBSHD()

        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            """Forward inputs to the wrapped attention module.

            Args:
                q: Query tensor in BSHD layout.
                k: Key tensor in BSHD layout.
                v: Value tensor in BSHD layout.

            Returns:
                The attention output tensor.
            """
            return self.core_attn(q, k, v)

    model = _ModelWithAttn()
    parallelize_module(
        model, mesh["cp"],
        {"core_attn": ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)},
    )

    with torch.no_grad():
        cp_out = model(local_q, local_k, local_v)

    # Reference: local_q attending to full K/V.
    ref_attn = SimpleAttnBSHD()
    with torch.no_grad():
        ref_local = ref_attn(local_q, full_k, full_v)

    assert cp_out.shape == (batch, local_s, num_heads, head_dim)
    _assert_close(cp_out, ref_local, rank, "I1_parallelize_module_api_npu")


@pytest.mark.skipif(
    dist.is_initialized() and dist.get_world_size() < 4,
    reason="TP+CP test requires world_size=4",
)
def test_tp_cp_combination_npu():
    """I2: TP=2 × CP=2 (Colossal mode), BSHD float16 — shape and numerical verification.

    Each rank holds S/cp tokens and H/tp heads.
    Reference: local Q/H slice attends to full K/V for the cp_rank's S slice.
    """
    rank = _init_dist_npu()
    if dist.get_world_size() < 4:
        pytest.skip("Requires world_size=4")
    batch, seq_len, num_heads, head_dim = 1, 8, 4, 16
    tp_size, cp_size = 2, 2

    mesh = init_device_mesh("npu", (tp_size, cp_size), mesh_dim_names=("tp", "cp"))
    tp_rank = mesh.get_local_rank("tp")
    cp_rank = mesh.get_local_rank("cp")

    local_s = seq_len // cp_size
    local_h = num_heads // tp_size

    torch.manual_seed(11)
    full_q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
    full_v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")

    local_q = full_q[:, cp_rank * local_s:(cp_rank + 1) * local_s,
                     tp_rank * local_h:(tp_rank + 1) * local_h]
    local_k = full_k[:, cp_rank * local_s:(cp_rank + 1) * local_s,
                     tp_rank * local_h:(tp_rank + 1) * local_h]
    local_v = full_v[:, cp_rank * local_s:(cp_rank + 1) * local_s,
                     tp_rank * local_h:(tp_rank + 1) * local_h]

    core_attn = SimpleAttnBSHD()
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, mesh["cp"])

    with torch.no_grad():
        cp_out = core_attn(local_q, local_k, local_v)

    assert cp_out.shape == (batch, local_s, local_h, head_dim), \
        f"[Rank {rank}] Expected {(batch, local_s, local_h, head_dim)}, got {cp_out.shape}"
    assert not torch.isnan(cp_out).any(), f"[Rank {rank}] NaN in TP×CP output"
    assert not torch.isinf(cp_out).any(), f"[Rank {rank}] Inf in TP×CP output"
    print(f"[Rank {rank}] I2_tp_cp_combination_npu: shape {tuple(cp_out.shape)} OK")
