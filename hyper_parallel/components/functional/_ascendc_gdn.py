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

"""AscendC GDN autograd composition, following VeOmni's stateless training path.

Reference: ByteDance-Seed/VeOmni, veomni/ops/kernels/gated_delta_rule/
_ascend/flash_gated_delta_rule.py (Apache-2.0), derived from MindSpeed-MM and
flashserve/flash-linear-attention-npu. Heavy forward and backward operations
use the fla_npu torch.ops.npu registrations, not the newer ctypes API.
KKT and triangular inversion follow VeOmni; cumulative sums reuse the equivalent
Hyper helper. This module is imported lazily on NPU only.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# auto_models/ops contains PyTorch-specific high-performance kernels.
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional._veomni_gdn_kkt import chunk_scaled_dot_kkt_fwd
from hyper_parallel.components.functional._veomni_gdn_tril import solve_tril
from hyper_parallel.components.functional._gdn_triton.cumsum import chunk_local_cumsum
from hyper_parallel.components.functional._gdn_triton.utils import input_guard, is_arch35


def _ensure_registered() -> None:
    """Load the PyTorch extension and reject an incompatible ctypes-only install."""
    import_module("fla_npu")
    required = (
        "npu_recompute_w_u_fwd", "npu_chunk_gated_delta_rule_fwd_h", "npu_chunk_fwd_o",
        "npu_chunk_bwd_dv_local", "npu_chunk_gated_delta_rule_bwd_dhu", "npu_chunk_bwd_dqkwg",
        "npu_prepare_wy_repr_bwd_da", "npu_prepare_wy_repr_bwd_full",
    )
    missing = [name for name in required if not hasattr(torch.ops.npu, name)]
    if missing:
        raise RuntimeError(
            "GDN requires VeOmni-compatible fla_npu torch.ops.npu registrations; "
            f"missing: {missing}. See VeOmni docs/examples/qwen3_5.md for the pinned build."
        )


def _head_first(tensor):
    return tensor.transpose(1, 2).contiguous()


def _recompute(key, value, beta, inverse, decay):
    return torch.ops.npu.npu_recompute_w_u_fwd(
        key, value, beta, inverse, 64, g=decay, gk=None, cu_seqlens=None, chunk_indices=None,
    )


def _states(key, weight, value, decay):
    return torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        key, weight, value, g=decay, gk=None, initial_state=None,
        output_final_state=False, chunk_size=64, save_new_value=True,
        cu_seqlens=None, chunk_indices=None, use_exp2=False, transpose_state_layout=False,
    )


class _AscendCGatedDeltaRule(torch.autograd.Function):
    """Bind the explicit AscendC backward to the chunk forward intermediates."""

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="npu")
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        decay: torch.Tensor, beta: torch.Tensor, scale: float,
    ) -> torch.Tensor:
        """Compute sequence-first output while retaining WY intermediates."""
        ctx.decay_dtype = decay.dtype
        decay = chunk_local_cumsum(decay, chunk_size=64, head_first=False)
        query, key, value = (_head_first(tensor) for tensor in (query, key, value))
        inverse = chunk_scaled_dot_kkt_fwd(
            k=key, g=decay, beta=beta, chunk_size=64, output_dtype=torch.float32,
        )
        inverse = _head_first(solve_tril(inverse, output_dtype=key.dtype))
        head_decay, head_beta = _head_first(decay), _head_first(beta).float()
        weight, updated = _recompute(key, value, head_beta, inverse, head_decay)
        state, new_value, _ = _states(key, weight, updated, head_decay)
        output = torch.ops.npu.npu_chunk_fwd_o(
            query, key, new_value, state, scale, g=head_decay, g_gamma=None,
            chunk_size=64, transpose_state_layout=False,
        )
        ctx.save_for_backward(query, key, value, head_decay, head_beta, inverse)
        ctx.scale = scale
        ctx.beta_dtype = beta.dtype
        return _head_first(output).to(query.dtype)

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="npu")
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Recompute recurrent states and differentiate the WY representation."""
        query, key, value, decay, beta, inverse = ctx.saved_tensors
        grad_output = _head_first(grad_output)
        weight, updated = _recompute(key, value, beta, inverse, decay)
        state, new_value, _ = _states(key, weight, updated, decay)
        grad_value = torch.ops.npu.npu_chunk_bwd_dv_local(
            query, key, grad_output, decay, ctx.scale, 64, g_gamma=None, A=inverse,
        )
        grad_state, _, grad_value = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            query, key, weight, grad_output, grad_value, ctx.scale, 64,
            g=decay, gK=None, h0=None, dht=None, use_exp2=False, transpose_state_layout=False,
        )
        grad_query, grad_key, grad_weight, grad_decay = torch.ops.npu.npu_chunk_bwd_dqkwg(
            query, key, new_value, decay, state, grad_output, grad_state, grad_value, 64,
            w=None, g_gamma=None, scale=ctx.scale, use_exp2=False, transpose_state_layout=False,
        )
        grad_inverse = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
            key, value, beta, inverse, grad_weight, grad_value, decay.float(), chunk_size=64,
        )
        grad_key_wy, grad_value, grad_beta, grad_decay_wy = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            key, value, beta, inverse, grad_inverse, grad_weight, grad_value, decay, 64,
        )
        grad_key.add_(grad_key_wy)
        grad_decay.add_(grad_decay_wy)
        if grad_decay.dtype != torch.float32:
            raise ValueError("AscendC GDN decay gradient must be float32")
        grad_decay = chunk_local_cumsum(
            _head_first(grad_decay), chunk_size=64, reverse=True, head_first=False,
        )
        return (
            _head_first(grad_query).to(query.dtype),
            _head_first(grad_key).to(key.dtype),
            _head_first(grad_value).to(value.dtype),
            grad_decay.to(ctx.decay_dtype),
            _head_first(grad_beta).to(ctx.beta_dtype),
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, None]:
    """Run the stateless AscendC training kernel on sequence-first tensors.

    Args:
        q: Query tensor [batch, sequence, heads, key_dim].
        k: Key tensor with the same layout as q.
        v: Value tensor [batch, sequence, heads, value_dim].
        g: Log decay [batch, sequence, heads].
        beta: Update gate with the same layout as g.
        scale: Query scale, or inverse square root of key_dim by default.
        use_qk_l2norm_in_kernel: Apply differentiable Q/K normalization as in VeOmni.

    Returns:
        Sequence-first output and None (no recurrent-state carry).
    """
    if is_arch35():
        raise NotImplementedError("This VeOmni GDN integration currently targets Ascend 910B/910_93")
    _ensure_registered()
    if use_qk_l2norm_in_kernel:
        # Keep normalization outside the custom Function so native autograd
        # differentiates the same input-dtype normalization used by VeOmni.
        q = (q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    scale = k.shape[-1] ** -0.5 if scale is None else float(scale)
    inputs = (tensor.contiguous() for tensor in (q, k, v, g, beta))
    return _AscendCGatedDeltaRule.apply(*inputs, scale), None
