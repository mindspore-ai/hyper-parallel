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
"""Fusion-attention rescaling for parameter-sink attention."""

from typing import Any, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch_npu
from einops import rearrange


def _rescale_attention_outputs(
    output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    sink_output: torch.Tensor,
    sink_softmax_max: torch.Tensor,
    sink_softmax_sum: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = rearrange(output, "(b s) n d -> s b n d", b=batch_size, s=sequence_length)
    sink_output = rearrange(sink_output, "s b (n d) -> s b n d", n=num_heads)

    softmax_sum = softmax_sum[:, :, 0].view(batch_size, sequence_length, num_heads).transpose(0, 1)
    softmax_max = softmax_max[:, :, 0].view(batch_size, sequence_length, num_heads).transpose(0, 1)
    sink_softmax_sum = rearrange(sink_softmax_sum[:, :, :, 0], "b n s -> s b n")
    sink_softmax_max = rearrange(sink_softmax_max[:, :, :, 0], "b n s -> s b n")

    combined_max = torch.maximum(softmax_max, sink_softmax_max)
    output_sum = softmax_sum * torch.exp(softmax_max - combined_max)
    sink_sum = sink_softmax_sum * torch.exp(sink_softmax_max - combined_max)
    combined_sum = output_sum + sink_sum
    output_scale = (output_sum / combined_sum).unsqueeze(3)
    sink_scale = (sink_sum / combined_sum).unsqueeze(3)
    return (
        (output * output_scale + sink_output * sink_scale).to(dtype=output.dtype),
        output_scale,
        sink_scale,
    )


def _normal_attention_backward(ctx, grad_rescaled_output):
    """Compute the normal branch's Q/K/V gradients."""
    (
        query, key, value, sink_key, _, attention_mask, softmax_max, softmax_sum,
        _, _, rescaled_output, output_scale, _,
    ) = ctx.saved_tensors
    grad_output = rearrange(output_scale * grad_rescaled_output, "s b n d -> (b s) n d")
    return torch_npu.npu_fusion_attention_grad(
        query, key, value, grad_output.to(sink_key.dtype), ctx.num_heads, "TND",
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=rearrange(rescaled_output, "s b n d -> (b s) n d"),
        scale_value=ctx.scale,
        pre_tockens=ctx.pre_tokens,
        next_tockens=ctx.next_tokens,
        inner_precise=0,
        keep_prob=ctx.keep_prob,
        actual_seq_qlen=ctx.actual_seq_qlen,
        actual_seq_kvlen=ctx.actual_seq_kvlen,
        sparse_mode=ctx.sparse_mode,
        softmax_layout="TND",
    )[:3]


def _sink_attention_backward(ctx, grad_rescaled_output):
    """Compute sink gradients and restore the query gradient's TND layout."""
    (
        query, _, _, sink_key, sink_value, _, _, _,
        sink_softmax_max, sink_softmax_sum, rescaled_output, _, sink_scale,
    ) = ctx.saved_tensors
    sink_query = rearrange(query, "(b s) n d -> s b (n d)", b=ctx.batch_size, s=ctx.sequence_length)
    grad_output = rearrange(sink_scale * grad_rescaled_output, "s b n d -> s b (n d)")
    grad_query, grad_key, grad_value = torch_npu.npu_fusion_attention_grad(
        sink_query, sink_key, sink_value, grad_output.to(sink_key.dtype), ctx.num_heads, "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=None,
        softmax_max=sink_softmax_max,
        softmax_sum=sink_softmax_sum,
        attention_in=rearrange(rescaled_output, "s b n d -> s b (n d)"),
        scale_value=ctx.scale,
        inner_precise=0,
        keep_prob=ctx.keep_prob,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        sparse_mode=0,
    )[:3]
    grad_query = rearrange(
        grad_query, "s b (n d) -> (b s) n d", b=ctx.batch_size, s=ctx.sequence_length, n=ctx.num_heads,
    )
    return grad_query, grad_key, grad_value


class _AttentionRescale(torch.autograd.Function):
    """Autograd bridge for fusion attention with separate sink parameters."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sink_key: torch.Tensor,
        sink_value: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        num_heads: int,
        scale: float,
        pre_tokens: int,
        next_tokens: int,
        keep_prob: float,
        sparse_mode: int,
        actual_seq_qlen: Optional[Sequence[int]],
        actual_seq_kvlen: Optional[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run normal and sink fusion attention and rescale their outputs."""
        output, softmax_max, softmax_sum = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            num_heads,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=scale,
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            keep_prob=keep_prob,
            inner_precise=0,
            sparse_mode=sparse_mode,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            softmax_layout="TND",
        )[:3]
        sink_query = rearrange(query, "(b s) n d -> s b (n d)", b=batch_size, s=sequence_length)
        sink_output, sink_softmax_max, sink_softmax_sum = torch_npu.npu_fusion_attention(
            sink_query,
            sink_key,
            sink_value,
            num_heads,
            "SBH",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            scale=scale,
            keep_prob=keep_prob,
            inner_precise=0,
            sparse_mode=0,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
        )[:3]
        rescaled_output, output_scale, sink_scale = _rescale_attention_outputs(
            output,
            softmax_max,
            softmax_sum,
            sink_output,
            sink_softmax_max,
            sink_softmax_sum,
            sequence_length,
            batch_size,
            num_heads,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            sink_key,
            sink_value,
            attention_mask,
            softmax_max,
            softmax_sum,
            sink_softmax_max,
            sink_softmax_sum,
            rescaled_output,
            output_scale,
            sink_scale,
        )
        ctx.batch_size = batch_size
        ctx.sequence_length = sequence_length
        ctx.num_heads = num_heads
        ctx.scale = scale
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.keep_prob = keep_prob
        ctx.sparse_mode = sparse_mode
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        return rescaled_output, softmax_max

    @staticmethod
    def backward(ctx: Any, grad_rescaled_output: torch.Tensor, _grad_softmax_max: torch.Tensor) -> tuple:
        """Run the explicit fusion-attention backward operators."""
        grad_query, grad_key, grad_value = _normal_attention_backward(ctx, grad_rescaled_output)
        sink_grad_query, sink_grad_key, sink_grad_value = _sink_attention_backward(ctx, grad_rescaled_output)
        return (
            grad_query + sink_grad_query,
            grad_key,
            grad_value,
            sink_grad_key,
            sink_grad_value,
            *((None,) * 11),
        )


def attention_rescale(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink_key: torch.Tensor,
    sink_value: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    scale: float,
    pre_tokens: int,
    next_tokens: int,
    keep_prob: float,
    sparse_mode: int,
    actual_seq_qlen: Optional[Sequence[int]],
    actual_seq_kvlen: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine normal and parameter-sink fusion-attention results.

    Query, key, and value use TND layout. Sink key and value use SBH layout.
    Sequence lengths are cumulative Python sequences, as required by the
    torch-npu TND fusion-attention interface.

    Returns:
        Rescaled attention output in SBND layout and the normal attention
        softmax maximum.
    """
    return _AttentionRescale.apply(
        query,
        key,
        value,
        sink_key,
        sink_value,
        attention_mask,
        batch_size,
        sequence_length,
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        keep_prob,
        sparse_mode,
        actual_seq_qlen,
        actual_seq_kvlen,
    )
