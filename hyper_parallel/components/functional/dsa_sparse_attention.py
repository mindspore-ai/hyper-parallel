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
"""DSA sparse-attention function."""

from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from einops import rearrange

from hyper_parallel.tools.logging import get_logger

logger = get_logger()


def _load_sparse_attention_grad():
    """Defer the optional NPU dependency until sparse-attention backward."""
    from cann_ops_transformer.ops import sparse_flash_attention_grad  # pylint: disable=import-outside-toplevel

    return sparse_flash_attention_grad


def _sparse_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
    attention_op: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call torch_npu SparseFlashAttention with the V3.2 DSA contract."""
    kwargs = {
        "query_rope": None if query_rope is None else query_rope.contiguous(),
        "key_rope": None if key_rope is None else key_rope.contiguous(),
        "sparse_block_size": 1,
        "layout_query": layout,
        "layout_kv": layout,
        "sparse_mode": 3,
        "attention_mode": 2,
        "return_softmax_lse": True,
    }
    if layout == "TND":
        kwargs.update({"actual_seq_lengths_query": query_lengths, "actual_seq_lengths_kv": key_lengths})
    key = key.contiguous()
    return attention_op(query.contiguous(), key, key, indices.contiguous(), scale, **kwargs)


def _sparse_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    indices: torch.Tensor,
    grad_output: torch.Tensor,
    output: torch.Tensor,
    maximum: torch.Tensor,
    total: torch.Tensor,
    scale: float,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
    grad_op: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Let CANN GradV2 merge gradients for the shared K/V tensor."""
    key = key.contiguous()
    result = grad_op(
        query.contiguous(), key, None, indices.contiguous(), grad_output.contiguous(),
        output.contiguous(), maximum.contiguous(), total.contiguous(), None, scale, 1,
        query_rope=None if query_rope is None else query_rope.contiguous(),
        key_rope=None if key_rope is None else key_rope.contiguous(),
        actual_seq_qlen=query_lengths,
        actual_seq_kvlen=key_lengths,
        layout=layout,
        sparse_mode=3,
        attention_mode=2,
    )
    grad_query, grad_key, _, grad_query_rope, grad_key_rope, _ = result
    return (
        grad_query,
        grad_key,
        None if query_rope is None else grad_query_rope,
        None if key_rope is None else grad_key_rope,
    )


class _SparseAttention(torch.autograd.Function):
    """Pair torch_npu sparse-attention forward with the available CANN backward API."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        indices: torch.Tensor,
        query_rope: torch.Tensor | None,
        key_rope: torch.Tensor | None,
        qlen: torch.Tensor | None,
        klen: torch.Tensor | None,
        scale: float,
        layout: str,
        attention_op: Any,
        grad_op: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sparse attention and save inputs needed by its backward pass."""
        outputs = _sparse_attention_forward(
            query, key, indices, scale, query_rope, key_rope, qlen, klen, layout, attention_op,
        )
        optional_inputs = (query_rope, key_rope, qlen, klen)
        ctx.optional_input_mask = tuple(tensor is not None for tensor in optional_inputs)
        saved_inputs = [query, key, indices]
        saved_inputs.extend(tensor for tensor in optional_inputs if tensor is not None)
        ctx.save_for_backward(*saved_inputs, *outputs)
        ctx.scale, ctx.layout, ctx.grad_op = scale, layout, grad_op
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        gradmax: torch.Tensor | None,
        gradsum: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute gradients for differentiable sparse-attention inputs."""
        del gradmax, gradsum
        saved = iter(ctx.saved_tensors)
        query, key, indices = (next(saved) for _ in range(3))
        optional_inputs = tuple(next(saved) if present else None for present in ctx.optional_input_mask)
        query_rope, key_rope, qlen, klen = optional_inputs
        output, maximum, total = (next(saved) for _ in range(3))
        gradients = _sparse_attention_backward(
            query, key, indices, grad_output, output, maximum, total, ctx.scale,
            query_rope, key_rope, qlen, klen, ctx.layout, ctx.grad_op(),
        )
        return gradients[0], gradients[1], None, *gradients[2:], None, None, None, None, None, None


def _cann_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    *,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
    attention_op: Any,
    grad_op: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CANN sparse attention with the matching backward API and return statistics."""
    return _SparseAttention.apply(
        query, key, sparse_indices, query_rope, key_rope,
        query_lengths, key_lengths, scale, layout, attention_op, grad_op,
    )


def dsa_sparse_attention(
    query_nope: torch.Tensor,
    compressed_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    *,
    backend: str = "cann",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute NPU DSA sparse flash attention without parameter-sink rescaling.

    Args:
        query_nope: Absorbed MLA query in BSND layout.
        compressed_kv: Compressed MLA key/value in BSND layout.
        query_rope: Query rotary states in BSND layout.
        key_rope: Key rotary states in BSND layout.
        topk_indices: Int32 ``[T, N_kv, K]`` output from :func:`dsa_indexer`.
        scale: Attention score scaling factor.
        actual_seq_qlen: Cumulative query lengths as an int32 NPU tensor.
        actual_seq_kvlen: Cumulative key lengths as an int32 NPU tensor.
        backend: ``"cann"`` (default) or the legacy ``"omni"`` backend.

    Returns:
        Attention output in BSND layout, followed by the softmax maximum and
        sum required by :func:`dsa_kl_loss`. The output head dimension contains
        the compressed-KV dimension followed by zero padding for the RoPE
        dimension required by the sparse-attention backward operator.
    """
    if backend not in ("cann", "omni"):
        raise ValueError(f"backend must be 'cann' or 'omni', got {backend!r}")
    query_tnd = rearrange(query_nope, "b s n d -> (b s) n d")
    key_tnd = rearrange(compressed_kv, "b s n d -> (b s) n d")
    query_rope_tnd = rearrange(query_rope, "b s n d -> (b s) n d")
    key_rope_tnd = rearrange(key_rope, "b s n d -> (b s) n d")
    # Load only the selected optional backend so either can be used independently.
    # pylint: disable=import-outside-toplevel
    if backend == "cann":
        import torch_npu  # pylint: disable=forbidden-backend-import

        output, softmax_max, softmax_sum = _cann_sparse_attention(
            query_tnd,
            key_tnd,
            topk_indices,
            scale,
            query_rope=query_rope_tnd,
            key_rope=key_rope_tnd,
            query_lengths=actual_seq_qlen,
            key_lengths=actual_seq_kvlen,
            layout="TND",
            attention_op=torch_npu.npu_sparse_flash_attention,
            grad_op=_load_sparse_attention_grad,
        )
    else:
        logger.warning(
            "dsa_sparse_attention backend='omni' is deprecated and will be removed in a future release; "
            "use backend='cann'."
        )
        import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import

        output, softmax_max, softmax_sum = torch.ops.custom.npu_sparse_flash_attention_enhance(
            query_tnd,
            key_tnd,
            key_tnd,
            topk_indices,
            scale,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            query_rope=query_rope_tnd,
            key_rope=key_rope_tnd,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
    # pylint: enable=import-outside-toplevel
    if query_rope.size(-1) > 0:
        output = F.pad(output, [0, query_rope.size(-1)])
    output = rearrange(output, "(b s) n d -> b s n d", b=query_nope.shape[0])
    return output, softmax_max, softmax_sum
