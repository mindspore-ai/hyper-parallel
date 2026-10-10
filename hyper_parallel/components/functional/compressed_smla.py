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
"""Native V4 sparse attention with the model's independent KL teacher preserved."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import


@lru_cache(maxsize=1)
def _load_attention_ops() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Resolve attention forward, backward and metadata through the official binding."""
    # Lazy import leaves the eager baseline usable without the NPU extension.
    try:
        from cann_ops_transformer import (  # pylint: disable=import-outside-toplevel
            sparse_flash_mla, sparse_flash_mla_grad, sparse_flash_mla_metadata,
        )
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.sparse_flash_mla, sparse_flash_mla_grad "
            "and sparse_flash_mla_metadata. Install the ops-transformer PyTorch extension built for the "
            "runtime torch-npu and CANN versions."
        ) from err
    return sparse_flash_mla, sparse_flash_mla_grad, sparse_flash_mla_metadata


def supports_fused_attention_inputs(query: torch.Tensor, ratio: int, window: int, topk: int = 512) -> bool:
    """Check the validated attention input contract without importing operators.

    Args:
        query: Main Q [batch, heads, queries, dimension].
        ratio: Model ratio, zero for a pure SWA layer.
        window: Raw attention window, including the current token.
        topk: Logical compressed selection width, before padding.
    """
    if query.device.type != "npu" or query.dtype != torch.bfloat16:
        return False
    if query.shape[-1] != 512 or window != 128 or ratio not in (0, 1, 2):
        return False
    if query.shape[1] not in (1, 2, 4, 8, 16, 32, 64, 128) or query.shape[2] == 0:
        return False
    if ratio and not 1 <= topk <= 512:
        return False
    return True


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _attention_metadata(
        device: torch.device, batch: int, length: int, heads: int, dim: int,
        raw_length: int, main_length: int, topk: int, ratio: int, residual: int, window: int,
) -> tuple:
    """Reuse shape metadata across evaluation and gradient-tracked training.

    Evaluation may populate the cache under inference_mode. The backward
    saves the cached residual, so it must be an ordinary tensor regardless
    of which mode first populated the cache.
    """
    # The training backward requires a zero residual tensor even at ratio one;
    # the forward interface instead omits that optional input at ratio one.
    residue = torch.full((batch,), residual, dtype=torch.int32, device=device) if main_length else None
    _, _, metadata_op = _load_attention_ops()
    metadata = metadata_op(
        heads, 1, dim, cmp_residual_kv=residue if ratio > 1 else None, batch_size=batch, max_seqlen_q=length,
        max_seqlen_ori_kv=raw_length, max_seqlen_cmp_kv=main_length, ori_topk=0, cmp_topk=topk,
        cmp_ratio=ratio, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=window-1, ori_win_right=0,
        layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=bool(main_length),
    )
    return residue, metadata


class _SparseMla(torch.autograd.Function):
    """Call the training backward explicitly and discard its unrelated teacher output."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, raw: torch.Tensor, main: torch.Tensor | None,
            indices: torch.Tensor | None, sinks: torch.Tensor, scale: float,
            ratio: int, residual: int, window: int,
    ) -> torch.Tensor:
        """Save the native output and LSE required by the training backward."""
        batch, length, heads, dim = query.shape
        residue, metadata = _attention_metadata(
            query.device, batch, length, heads, dim, raw.shape[1], 0 if main is None else main.shape[1],
            0 if indices is None else indices.shape[-1], ratio, residual, window,
        )
        options = {"softmax_scale": scale, "cmp_ratio": ratio, "ori_mask_mode": 4, "cmp_mask_mode": 3,
                   "ori_win_left": window-1, "ori_win_right": 0, "layout_q": "BSND", "layout_kv": "BSND"}
        attention, _, _ = _load_attention_ops()
        output, lse = attention(
            query, ori_kv=raw, cmp_kv=main, cmp_sparse_indices=indices, sinks=sinks,
            cmp_residual_kv=residue if ratio > 1 else None, metadata=metadata, return_softmax_lse=True, **options,
        )
        ctx.save_for_backward(query, raw, main, indices, sinks, residue, output, lse)
        ctx.options = options
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return main-branch gradients without substituting the KL teacher."""
        query, raw, main, indices, sinks, residue, output, lse = ctx.saved_tensors
        _, attention_grad, _ = _load_attention_ops()
        grad_query, grad_raw, grad_main, grad_sinks, _, _ = attention_grad(
            query, grad_output.contiguous(), output, lse, ori_kv=raw, cmp_kv=main,
            cmp_sparse_indices=indices, sinks=sinks, cmp_residual_kv=residue, **ctx.options,
        )
        return grad_query, grad_raw, grad_main if main is not None else None, None, grad_sinks, None, None, None, None


def fused_sparse_mla_attention(
        query: torch.Tensor, raw_key: torch.Tensor, main_key: torch.Tensor | None,
        indices: torch.Tensor | None, sinks: torch.Tensor, scale: float, ratio: int,
        window: int, query_offset: int,
        segments: tuple[tuple[int, int, int, int, int], ...],
) -> torch.Tensor:
    """Run document-local windows and K prefixes from the original all-gather banks.

    Document boundaries require separate BSND calls; the operator owns
    tiling within each document rather than imposing a Python Q-length cap.

    Args:
        query: Local attention Q in [B, H, Q, D].
        raw_key: Global raw KV in [B, 1, K, D].
        main_key: Global compressed KV in [B, Kc, D], or None for SWA only.
        indices: Global compressed selection [B, Q, topk], invalid slots -1.
        sinks: One scalar logit per head.
        scale: Attention dot-product scale.
        ratio: Compression ratio, zero denotes the SWA-only layer.
        window: Number of raw tokens visible to each query including itself.
        query_offset: Global position of the first local query.
        segments: Cached local Q start/end, compressed document start/prefix end, residual.

    Returns:
        Attention output in [B, Q, H, D].
    """
    if query.dtype != torch.bfloat16 or query.shape[-1] != 512 or window != 128:
        raise ValueError("V4 attention requires BF16, head_dim=512, and window=128")
    outputs = []
    for start, end, key_start, key_end, residual in segments:
        document_start = key_start * max(ratio, 1)
        raw_left = max(document_start, query_offset + start - window + 1)
        raw_right = query_offset + end
        raw = raw_key[:, :, raw_left:raw_right].transpose(1, 2).contiguous()
        main = None
        selected = None
        if ratio and key_end > key_start:
            main = main_key[:, key_start:key_end].unsqueeze(2).contiguous()
            original = indices[:, start:end]
            selected = torch.where(original >= 0, original-key_start, -1).int()
            if selected.shape[-1] < 512:
                selected = torch.nn.functional.pad(selected, (0, 512-selected.shape[-1]), value=-1)
            selected = selected.unsqueeze(2).contiguous()
        part = _SparseMla.apply(
            query[:, :, start:end].transpose(1, 2).contiguous(), raw, main, selected, sinks.float(),
            scale, max(ratio, 1) if main is not None else 1, residual if main is not None else 0, window,
        )
        outputs.append(part)
    return torch.cat(outputs, dim=1)
